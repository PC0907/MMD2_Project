#!/usr/bin/env python
"""
Training entry point.

    python -m clarity_evasion.cli.train --config configs/reverse.yaml
    python -m clarity_evasion.cli.train --config configs/reverse.yaml --epochs 8 --fp16

CLI flags override the YAML config.
"""
from __future__ import annotations

import argparse
import json

from ..pipelines import TRAIN_PIPELINES
from ..utils import Config, get_logger

log = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description="Train a clarity/evasion model.")
    p.add_argument("--config", required=True, help="Path to a YAML config.")
    # common overrides (None = keep config value)
    p.add_argument("--pipeline", choices=["direct", "reverse", "evasion"])
    p.add_argument("--model_name")
    p.add_argument("--output_dir")
    p.add_argument("--agg", choices=["argmax", "marginal"])
    p.add_argument("--evasion_mode", choices=["baseline", "joint"])
    p.add_argument("--epochs", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--max_len", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--fp16", action="store_true", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.update(**{k: v for k, v in vars(args).items() if k != "config"})
    log.info("pipeline=%s model=%s output_dir=%s", cfg.pipeline, cfg.model_name, cfg.output_dir)

    run = TRAIN_PIPELINES.get(cfg.pipeline)
    if run is None:
        raise SystemExit(f"unknown pipeline: {cfg.pipeline!r}")
    results = run(cfg)
    log.info("RESULTS: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
