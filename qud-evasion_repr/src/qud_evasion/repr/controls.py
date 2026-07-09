"""Probe validity controls.

A probe result is only evidence about the *representation* if the probe
could not have achieved it by memorising arbitrary structure. Reviewers
of probing claims expect two controls (Hewitt & Liang, 2019):

1. Control task: refit the identical probe on labels that preserve the
   class marginals but carry no linguistic content. Here labels are
   assigned by a deterministic hash of the example id -- consistent
   across runs (a "task" the probe could in principle memorise), unlike
   naive per-run shuffling.

2. Selectivity: real-task eval F1 minus control-task eval F1, reported
   per layer. High probe F1 with low selectivity means the probe itself
   is doing the work; high selectivity localises the information in the
   representation.

We report the layer curve of both. The paper's claim ("the distinction
is linearly available from layer 1 and saturates at human agreement")
only stands if selectivity is high across that band.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .extract import ActivationCache
from .heads import ProbeResult, fit_logistic_probe, _apply_filter


def control_labels(ids: list[str], y_real: np.ndarray, seed: int = 0) -> np.ndarray:
    """Marginal-preserving pseudo-labels, deterministic per example id."""
    valid = y_real >= 0
    classes, counts = np.unique(y_real[valid], return_counts=True)
    probs = counts / counts.sum()

    y_ctrl = np.full_like(y_real, -1)
    for i, ex_id in enumerate(ids):
        if not valid[i]:
            continue
        h = int(hashlib.sha256(f"{seed}:{ex_id}".encode()).hexdigest(), 16)
        u = (h % 10**9) / 10**9
        y_ctrl[i] = classes[np.searchsorted(np.cumsum(probs), u)]
    return y_ctrl


def selectivity_sweep(
    cache: ActivationCache,
    task: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    pooling: str = "last",
    c: float = 1.0,
    seed: int = 42,
    label_filter=None,
) -> list[dict]:
    """Per-layer: real F1, control F1, selectivity."""
    y_real = cache.labels(task)
    y_ctrl = control_labels(cache.index["ids"], y_real, seed=seed)

    tr_r, ev_r = _apply_filter(train_idx, eval_idx, y_real, label_filter)
    out = []
    for layer in range(cache.n_layers):
        x = cache.features(pooling, layer)
        real_f1, _, _ = fit_logistic_probe(
            x[tr_r], y_real[tr_r], x[ev_r], y_real[ev_r], c=c, seed=seed
        )
        ctrl_f1, _, _ = fit_logistic_probe(
            x[tr_r], y_ctrl[tr_r], x[ev_r], y_ctrl[ev_r], c=c, seed=seed
        )
        out.append(
            {
                "layer": layer,
                "real_macro_f1": real_f1,
                "control_macro_f1": ctrl_f1,
                "selectivity": real_f1 - ctrl_f1,
            }
        )
    return out
