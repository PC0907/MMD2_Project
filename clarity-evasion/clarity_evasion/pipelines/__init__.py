"""Pipeline dispatch: map a pipeline name to its training run() function."""
from . import direct, reverse, evasion, evaluate

TRAIN_PIPELINES = {
    "direct": direct.run,
    "reverse": reverse.run,
    "evasion": evasion.run,
}

__all__ = ["direct", "reverse", "evasion", "evaluate", "TRAIN_PIPELINES"]
