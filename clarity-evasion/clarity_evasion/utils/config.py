"""Typed experiment configuration loaded from YAML (with CLI overrides)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # task / pipeline
    pipeline: str = "reverse"            # direct | reverse | evasion
    agg: str = "marginal"               # reverse only: argmax | marginal
    evasion_mode: str = "baseline"      # evasion pipeline: baseline | joint
    aux_weight: float = 0.5             # joint only

    # model
    model_name: str = "microsoft/deberta-v3-base"
    max_len: int = 384
    num_labels: int | None = None       # set by pipeline from taxonomy

    # data
    dev_size: float = 0.15
    drop_inaudible: bool = False
    use_class_weights: bool = True

    # optimisation
    epochs: float = 6
    lr: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int | None = None
    grad_accum: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    fp16: bool = False

    # bookkeeping
    seed: int = 42
    output_dir: str = "runs/exp"
    logging_steps: int = 25

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size * 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__annotations__}
        extra = {k: v for k, v in data.items() if k not in cls.__annotations__}
        cfg = cls(**known)
        cfg.extra.update(extra)
        return cfg

    def update(self, **overrides) -> "Config":
        """Apply non-None CLI overrides in place and return self."""
        for k, v in overrides.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size * 2
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
