"""Frozen-decoder representation classifiers (the paper's core arm).

extract   -> forward passes, all-layer pooled caching (run once per model)
heads     -> logistic layer-sweep probe + learned layer-weighted head
controls  -> Hewitt-Liang control task and selectivity
data_adapter -> canonical Example schema shared by every arm
"""

from .data_adapter import CLARITY_LABELS, EVASION_LABELS, Example, load_examples
from .extract import ActivationCache, ExtractionConfig, extract, load_cache

__all__ = [
    "CLARITY_LABELS",
    "EVASION_LABELS",
    "Example",
    "load_examples",
    "ActivationCache",
    "ExtractionConfig",
    "extract",
    "load_cache",
]
