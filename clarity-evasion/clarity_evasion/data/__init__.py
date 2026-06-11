"""
Data package. Taxonomy (pure numpy) is eager; dataset loading (needs `datasets`,
`sklearn`) is lazy so the taxonomy/label spaces can be imported anywhere.
"""
from .taxonomy import (
    CLARITY_LABELS, CLARITY2ID, ID2CLARITY,
    EVASION_LABELS, EVASION2ID, ID2EVASION,
    EVASION_TO_CLARITY, EVASION_ID_TO_CLARITY_ID, CLARITY_TO_EVASIONS,
    BRANCH_MASK, EVASION_TO_CLARITY_LUT, validate,
)
from . import taxonomy

_LAZY = {"Splits", "load_raw", "prepare", "make_splits", "class_weights", "HF_DATASET"}

__all__ = [
    "CLARITY_LABELS", "CLARITY2ID", "ID2CLARITY",
    "EVASION_LABELS", "EVASION2ID", "ID2EVASION",
    "EVASION_TO_CLARITY", "EVASION_ID_TO_CLARITY_ID", "CLARITY_TO_EVASIONS",
    "BRANCH_MASK", "EVASION_TO_CLARITY_LUT", "validate", "taxonomy",
    *_LAZY,
]


def __getattr__(name):
    if name in _LAZY:
        from . import dataset
        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
