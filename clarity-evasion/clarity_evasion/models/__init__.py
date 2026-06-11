"""
Model components. Aggregation (pure numpy) is imported eagerly; the torch-backed
components are imported lazily so that `derive_clarity` and friends can be used in
environments without torch (e.g. lightweight analysis or CI logic tests).
"""
from .aggregation import derive_clarity, softmax

__all__ = [
    "derive_clarity", "softmax",
    "WeightedTrainer", "JointClarityEvasionModel", "build_classifier", "tokenize_df",
]


def __getattr__(name):
    if name in {"WeightedTrainer", "JointClarityEvasionModel",
                "build_classifier", "tokenize_df"}:
        from . import components
        return getattr(components, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
