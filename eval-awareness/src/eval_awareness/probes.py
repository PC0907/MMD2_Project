"""Stage 3: train eval-vs-deploy probes per layer.

Two evaluation regimes:
  1. random k-fold (upper bound; can exploit family-specific vocabulary)
  2. leave-one-family-out (LOFO) — the number that matters for the paper:
     does the probe generalize to scenario families it never saw?

Outputs:
  results/probes/<model>/layer_auroc.csv        both regimes, per layer
  results/probes/<model>/lofo_matrix.csv        per held-out family AUROC (best layer)
  results/probes/<model>/direction_layer{L}.npy unit-norm probe direction (for steering)
  results/probes/<model>/scores.csv             per-example probe score at best layer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir


def _load(npz_path: str | Path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)  # [N, L, D]
    meta = json.loads(str(d["meta"]))
    y = np.array([1 if m["framing"] == "eval" else 0 for m in meta])
    fams = np.array([m["family"] for m in meta])
    return X, y, fams, meta


def _fit_score(Xtr, ytr, Xte, yte, C: float, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    p = clf.predict_proba(scaler.transform(Xte))[:, 1]
    auroc = roc_auc_score(yte, p) if len(set(yte)) > 1 else float("nan")
    w = clf.coef_[0] / (scaler.scale_ + 1e-8)  # map back to raw-activation space
    return auroc, p, w / (np.linalg.norm(w) + 1e-8)


def run(npz_path: str | Path, out_dir: str | Path, C: float = 0.1,
        n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    X, y, fams, meta = _load(npz_path)
    n_layers = X.shape[1]
    out_dir = ensure_dir(out_dir)
    families = sorted(set(fams))

    rows = []
    lofo_scores_per_layer = {}
    for layer in range(n_layers):
        Xl = X[:, layer, :]

        # regime 1: random stratified k-fold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        kfold_aurocs = []
        for tr, te in kf.split(Xl, y):
            a, _, _ = _fit_score(Xl[tr], y[tr], Xl[te], y[te], C, seed)
            kfold_aurocs.append(a)

        # regime 2: leave-one-family-out
        lofo_aurocs, per_family = [], {}
        probs = np.zeros(len(y))
        for fam in families:
            te = fams == fam
            a, p, _ = _fit_score(Xl[~te], y[~te], Xl[te], y[te], C, seed)
            lofo_aurocs.append(a)
            per_family[fam] = a
            probs[te] = p
        lofo_scores_per_layer[layer] = (np.nanmean(lofo_aurocs), per_family, probs)

        rows.append({
            "layer": layer,
            "auroc_kfold": float(np.nanmean(kfold_aurocs)),
            "auroc_lofo": float(np.nanmean(lofo_aurocs)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "layer_auroc.csv", index=False)

    best_layer = int(df.loc[df["auroc_lofo"].idxmax(), "layer"])
    _, per_family, probs = lofo_scores_per_layer[best_layer]
    pd.DataFrame(
        [{"family": f, "auroc": a} for f, a in per_family.items()]
    ).to_csv(out_dir / "lofo_matrix.csv", index=False)

    # per-example LOFO probe score at best layer (used for compliance-gap analysis)
    pd.DataFrame({
        "id": [m["id"] for m in meta],
        "family": fams,
        "framing": [m["framing"] for m in meta],
        "kind": [m["kind"] for m in meta],
        "pair_id": [m["pair_id"] for m in meta],
        "probe_score": probs,
    }).to_csv(out_dir / "scores.csv", index=False)

    # final direction for steering: fit on ALL data at best layer
    _, _, w = _fit_score(X[:, best_layer, :], y, X[:, best_layer, :], y, C, seed)
    np.save(out_dir / f"direction_layer{best_layer}.npy", w)
    (out_dir / "best_layer.txt").write_text(str(best_layer))

    print(df.to_string(index=False))
    print(f"best layer (LOFO): {best_layer}")
    return df
