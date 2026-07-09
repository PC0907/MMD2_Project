"""Classification heads over cached representations.

Two tiers:

1. ``LogisticProbe`` -- L2-regularised logistic regression on one
   (pooling, layer) slice. This is the analytical instrument (the probe
   of Appendix A.3) and, swept over layers, produces the layer curve.

2. ``LayerWeightedHead`` -- small torch head that learns a softmax
   mixture over all layers plus a linear classifier. This is the
   *system* configuration: still frozen-backbone, but no manual layer
   choice, and the learned weights are themselves reportable (where does
   the model keep pragmatic information?).

Everything runs on cached features: CPU-fast, multi-seed for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .extract import ActivationCache

# ---------------------------------------------------------------------------
# Tier 1: logistic probe + layer sweep
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    pooling: str
    layer: int
    macro_f1: float
    per_class_f1: list[float]
    predictions: np.ndarray
    seed: int


def fit_logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    c: float = 1.0,
    seed: int = 42,
    class_weight: Optional[str] = None,
) -> tuple[float, list[float], np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(
        C=c, max_iter=3000, random_state=seed, class_weight=class_weight
    )
    clf.fit(scaler.transform(x_train), y_train)
    pred = clf.predict(scaler.transform(x_eval))
    macro = f1_score(y_eval, pred, average="macro")
    per_class = f1_score(y_eval, pred, average=None).tolist()
    return macro, per_class, pred


def layer_sweep(
    cache: ActivationCache,
    task: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    pooling: str = "last",
    c: float = 1.0,
    seed: int = 42,
    class_weight: Optional[str] = None,
    label_filter: Optional[tuple[int, int]] = None,
) -> list[ProbeResult]:
    """Probe every layer for one pooling.

    ``label_filter=(a, b)`` restricts to a binary boundary, e.g.
    (1, 2) = Ambiguous vs Clear Non-Reply, reproducing the v2 probe.
    """
    y = cache.labels(task)
    tr, ev = _apply_filter(train_idx, eval_idx, y, label_filter)
    results = []
    for layer in range(cache.n_layers):
        x = cache.features(pooling, layer)
        macro, per_class, pred = fit_logistic_probe(
            x[tr], y[tr], x[ev], y[ev], c=c, seed=seed, class_weight=class_weight
        )
        results.append(ProbeResult(pooling, layer, macro, per_class, pred, seed))
    return results


def _apply_filter(train_idx, eval_idx, y, label_filter):
    tr = train_idx[y[train_idx] >= 0]
    ev = eval_idx[y[eval_idx] >= 0]
    if label_filter is not None:
        keep = np.isin(y, list(label_filter))
        tr, ev = tr[keep[tr]], ev[keep[ev]]
    return tr, ev


# ---------------------------------------------------------------------------
# Tier 2: learned layer-weighted head
# ---------------------------------------------------------------------------


class LayerWeightedHead(nn.Module):
    """softmax(w) mixture over layers -> optional MLP -> linear classifier."""

    def __init__(self, n_layers: int, dim: int, n_classes: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        if hidden:
            self.head = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
            )
        else:
            self.head = nn.Linear(dim, n_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # feats: [B, L, D]
        w = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.einsum("l,bld->bd", w, feats)
        return self.head(self.norm(mixed))

    @property
    def layer_weights(self) -> np.ndarray:
        return torch.softmax(self.layer_logits.detach(), dim=0).cpu().numpy()


def train_layer_weighted(
    cache: ActivationCache,
    task: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    pooling: str = "mean_answer",
    hidden: int = 256,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-2,
    seed: int = 42,
    class_weighted_loss: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_all = cache.labels(task)
    tr = train_idx[y_all[train_idx] >= 0]
    ev = eval_idx[y_all[eval_idx] >= 0]

    feats = torch.from_numpy(cache.pooled[pooling].astype(np.float32))  # [N, L, D]
    # standardise per (layer, dim) on train only
    mu = feats[tr].mean(dim=0, keepdim=True)
    sd = feats[tr].std(dim=0, keepdim=True).clamp(min=1e-6)
    feats = (feats - mu) / sd

    y = torch.from_numpy(y_all).long()
    n_classes = int(y[tr].max().item()) + 1

    model = LayerWeightedHead(cache.n_layers, feats.shape[-1], n_classes, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if class_weighted_loss:
        counts = np.bincount(y[tr].numpy(), minlength=n_classes).astype(np.float32)
        w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    best = {"macro_f1": -1.0}
    tr_t = torch.from_numpy(tr)
    for epoch in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = criterion(model(feats[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(feats[torch.from_numpy(ev)].to(device))
            pred = logits.argmax(-1).cpu().numpy()
        macro = f1_score(y[ev].numpy(), pred, average="macro")
        if macro > best["macro_f1"]:
            best = {
                "macro_f1": macro,
                "per_class_f1": f1_score(y[ev].numpy(), pred, average=None).tolist(),
                "predictions": pred,
                "epoch": epoch,
                "layer_weights": model.layer_weights.tolist(),
                "pooling": pooling,
                "seed": seed,
            }
    return best
