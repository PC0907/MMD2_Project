"""
Derive Level-1 clarity predictions from Level-2 evasion logits via the taxonomy.

This is the core of the reverse hierarchical pipeline. Two modes:

  argmax   : take the top-1 evasion class, map it up to its clarity branch.
  marginal : softmax the evasion logits, SUM probability mass within each clarity
             branch, then argmax over the 3 branch sums.

`marginal` is more robust to within-branch uncertainty. If the model spreads mass
across several Ambivalent sub-strategies (Dodging / Deflection / General) but no
single one is top-1, argmax can leak the prediction into a different clarity branch,
whereas marginal still aggregates correctly to Ambivalent.
"""
from __future__ import annotations

import numpy as np

from ..data.taxonomy import BRANCH_MASK, EVASION_TO_CLARITY_LUT


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def derive_clarity(evasion_logits: np.ndarray, agg: str = "marginal") -> np.ndarray:
    """(N, 9) evasion logits -> (N,) clarity ids."""
    if agg == "argmax":
        return EVASION_TO_CLARITY_LUT[evasion_logits.argmax(-1)]
    if agg == "marginal":
        p = softmax(evasion_logits, axis=-1)        # (N, 9)
        branch_p = p @ BRANCH_MASK.T                # (N, 3)
        return branch_p.argmax(-1)
    raise ValueError(f"unknown aggregation: {agg!r}")
