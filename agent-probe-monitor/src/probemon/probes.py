"""Probe families over pooled activations.

logistic     sklearn logistic regression with L2, standardized inputs.
massmean     difference-of-class-means direction (Marks & Tegmark-style);
             score = x @ w with w = mu_pos - mu_neg on whitened features.
torch_linear single linear layer trained with BCE; useful when the dataset
             stops fitting in memory for sklearn, and for on-GPU inference
             in the runtime monitor.

All probes expose fit(X, y) and score(X) -> np.ndarray of real-valued
scores (higher = positive class). Thresholding lives in evaluate.py.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticProbe:
    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(self.scaler.transform(X))

    @property
    def direction(self) -> np.ndarray:
        return self.clf.coef_[0]


class MassMeanProbe:
    def __init__(self, whiten: bool = True, eps: float = 1e-4):
        self.whiten = whiten
        self.eps = eps
        self.w: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None  # whitening matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MassMeanProbe":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if self.whiten:
            cov = np.cov(Xc, rowvar=False) + self.eps * np.eye(X.shape[1])
            evals, evecs = np.linalg.eigh(cov)
            self.W_ = evecs @ np.diag(evals ** -0.5) @ evecs.T
            Xc = Xc @ self.W_
        self.w = Xc[y == 1].mean(axis=0) - Xc[y == 0].mean(axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean_
        if self.whiten:
            Xc = Xc @ self.W_
        return Xc @ self.w


class TorchLinearProbe:
    def __init__(self, epochs: int = 100, lr: float = 1e-2, batch_size: int = 512,
                 weight_decay: float = 1e-4, device: str | None = None):
        self.epochs, self.lr = epochs, lr
        self.batch_size, self.weight_decay = batch_size, weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.linear: torch.nn.Linear | None = None
        self.mu: torch.Tensor | None = None
        self.sd: torch.Tensor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchLinearProbe":
        Xt = torch.tensor(X, dtype=torch.float32)
        self.mu, self.sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
        Xt = ((Xt - self.mu) / self.sd).to(self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)

        self.linear = torch.nn.Linear(X.shape[1], 1).to(self.device)
        opt = torch.optim.AdamW(self.linear.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        # Class-balance via pos_weight.
        pos = yt.sum().clamp_min(1.0)
        pos_weight = (len(yt) - pos) / pos
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n = len(yt)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.linear(Xt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.tensor(X, dtype=torch.float32)
        Xt = ((Xt - self.mu) / self.sd).to(self.device)
        return self.linear(Xt).squeeze(-1).cpu().numpy()


def build_probe(kind: str, **kwargs):
    if kind == "logistic":
        return LogisticProbe(C=kwargs.get("C", 1.0))
    if kind == "massmean":
        return MassMeanProbe()
    if kind == "torch_linear":
        return TorchLinearProbe(
            epochs=kwargs.get("epochs", 100), lr=kwargs.get("lr", 1e-2),
            batch_size=kwargs.get("batch_size", 512),
        )
    raise ValueError(f"Unknown probe kind: {kind}")
