#!/usr/bin/env python
"""
Evaluation entry point — runs ONCE on the official test split.

    python -m clarity_evasion.cli.evaluate \
        --config configs/reverse.yaml \
        --model_dir runs/reverse-deberta/best \
        --out_dir runs/reverse-deberta/eval
"""
from __future__ import annotations

import argparse
import json

from ..pipelines import evaluate as eval_pipeline
from ..utils import Config, get_logger

log = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    p.add_argument("--config", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out_dir", default="runs/eval")
    p.add_argument("--pipeline", choices=["direct", "reverse"])
    p.add_argument("--agg", choices=["argmax", "marginal"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.update(pipeline=args.pipeline, agg=args.agg)
    results = eval_pipeline.run(cfg, model_dir=args.model_dir, out_dir=args.out_dir)
    log.info("RESULTS: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
