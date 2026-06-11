"""
Label spaces and the CLARITY taxonomy.

Two-level hierarchy from Thomas et al. (2024):
  Level 1 (clarity): Clear Reply, Ambivalent, Clear Non-Reply
  Level 2 (evasion): 9 fine-grained strategies, each nested under one clarity label.

The evasion -> clarity mapping was verified against every observable
(clarity_label, evasion_label) pair in the dataset (0 violations), confirming the
hierarchy is deterministic. `Clarification` is the one conceptually borderline case
but maps consistently to Clear Non-Reply in this data, so we follow the data.

Naming note: the SemEval/QEvasion data uses "Ambivalent" for the middle clarity
class. Some task descriptions call it "Ambiguous"; `data.dataset` normalises that
alias so nothing is silently dropped.
"""
from __future__ import annotations

import numpy as np

# --- Level 1: clarity -------------------------------------------------------
CLARITY_LABELS: list[str] = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
CLARITY2ID: dict[str, int] = {l: i for i, l in enumerate(CLARITY_LABELS)}
ID2CLARITY: dict[int, str] = {i: l for l, i in CLARITY2ID.items()}

# --- Level 2: evasion -------------------------------------------------------
EVASION_LABELS: list[str] = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Partial/half-answer", "Declining to answer", "Claims ignorance",
    "Clarification",
]
EVASION2ID: dict[str, int] = {l: i for i, l in enumerate(EVASION_LABELS)}
ID2EVASION: dict[int, str] = {i: l for l, i in EVASION2ID.items()}

# --- Hierarchy: evasion label -> clarity label ------------------------------
EVASION_TO_CLARITY: dict[str, str] = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

EVASION_ID_TO_CLARITY_ID: dict[int, int] = {
    EVASION2ID[e]: CLARITY2ID[c] for e, c in EVASION_TO_CLARITY.items()
}

# clarity label -> list of evasion labels nested under it
CLARITY_TO_EVASIONS: dict[str, list[str]] = {c: [] for c in CLARITY_LABELS}
for _e, _c in EVASION_TO_CLARITY.items():
    CLARITY_TO_EVASIONS[_c].append(_e)

# (3, 9) binary mask: row = clarity id, col = evasion id. Used to marginalise
# evasion probabilities into clarity branches.
BRANCH_MASK: np.ndarray = np.zeros((len(CLARITY_LABELS), len(EVASION_LABELS)),
                                   dtype=np.float32)
for _c, _evs in CLARITY_TO_EVASIONS.items():
    for _e in _evs:
        BRANCH_MASK[CLARITY2ID[_c], EVASION2ID[_e]] = 1.0

# (9,) lookup: evasion id -> clarity id
EVASION_TO_CLARITY_LUT: np.ndarray = np.array(
    [EVASION_ID_TO_CLARITY_ID[i] for i in range(len(EVASION_LABELS))], dtype=int)


def validate() -> bool:
    """Sanity-check the taxonomy is internally consistent. Raises on failure."""
    assert set(EVASION_TO_CLARITY) == set(EVASION_LABELS)
    assert set(EVASION_TO_CLARITY.values()) == set(CLARITY_LABELS)
    assert sum(len(v) for v in CLARITY_TO_EVASIONS.values()) == len(EVASION_LABELS)
    assert BRANCH_MASK.sum() == len(EVASION_LABELS)
    return True
