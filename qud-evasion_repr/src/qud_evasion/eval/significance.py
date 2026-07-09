"""Statistical reporting used in every results table.

Three primitives:

  bootstrap_ci       -- 95% CI on macro-F1 for one system, resampling
                        eval examples (grouped by interview, since
                        sibling rows are not independent).
  paired_bootstrap   -- p-value for "system A beats system B" on the
                        same eval set (the comparison the paper's tables
                        make constantly).
  aggregate_seeds    -- mean +/- sd over seeds, plus all per-seed values,
                        matching the reporting style of the fusion
                        experiment.

Grouped resampling matters here: sibling sub-questions share an answer,
so example-level bootstrap understates variance. We resample interviews.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _group_indices(groups: np.ndarray) -> list[np.ndarray]:
    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    return [np.array(v) for v in order.values()]


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    average: str = "macro",
) -> dict:
    rng = np.random.default_rng(seed)
    point = f1_score(y_true, y_pred, average=average)
    units = _group_indices(groups) if groups is not None else [np.array([i]) for i in range(len(y_true))]

    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), size=len(units))
        idx = np.concatenate([units[i] for i in pick])
        stats.append(f1_score(y_true[idx], y_pred[idx], average=average))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    average: str = "macro",
) -> dict:
    """One-sided test that A's macro-F1 > B's, resampling grouped units."""
    rng = np.random.default_rng(seed)
    delta = f1_score(y_true, pred_a, average=average) - f1_score(y_true, pred_b, average=average)
    units = _group_indices(groups) if groups is not None else [np.array([i]) for i in range(len(y_true))]

    wins = 0
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), size=len(units))
        idx = np.concatenate([units[i] for i in pick])
        d = f1_score(y_true[idx], pred_a[idx], average=average) - f1_score(
            y_true[idx], pred_b[idx], average=average
        )
        deltas.append(d)
        if d <= 0:
            wins += 1
    return {
        "delta": float(delta),
        "p_value": (wins + 1) / (n_boot + 1),
        "delta_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }


def aggregate_seeds(per_seed_scores: dict[int, float]) -> dict:
    vals = np.array(list(per_seed_scores.values()), dtype=float)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "per_seed": {int(k): float(v) for k, v in per_seed_scores.items()},
        "n_seeds": len(vals),
    }
