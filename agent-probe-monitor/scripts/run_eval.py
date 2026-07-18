#!/usr/bin/env python
"""Run black-box baselines on the same splits and merge with probe results.

Usage:
    python scripts/run_eval.py --config configs/default.yaml \
        --split loao --holdout important_instructions \
        --baselines perplexity action_logprob llamaguard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probemon.baselines import run_baselines
from probemon.datasets import load_meta, make_split
from probemon.evaluate import summarize
from probemon.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--baselines", nargs="+",
                    default=["perplexity", "action_logprob"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["collection"]["seed"])
    out_dir = Path(cfg["collection"]["out_dir"])
    res_dir = Path(cfg["eval"]["out_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    split_kind = args.split or cfg["dataset"]["split"]
    meta = load_meta(out_dir).sort_values(["episode_id", "step"])
    _, val, test = make_split(meta, split_kind, holdout=args.holdout,
                              val_fraction=cfg["dataset"]["val_fraction"],
                              seed=cfg["collection"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=getattr(torch, cfg["model"]["dtype"]),
        device_map=cfg["model"]["device_map"],
    )
    model.eval()
    device = next(model.parameters()).device

    results = []
    val_scores = run_baselines(val, model, tokenizer, args.baselines, device)
    test_scores = run_baselines(test, model, tokenizer, args.baselines, device)
    for target in cfg["dataset"]["targets"]:
        y_va, y_te = val[target].to_numpy(), test[target].to_numpy()
        for name in args.baselines:
            row = summarize(
                name=f"baseline/{name}/{target}",
                y_test=y_te, s_test=test_scores[name],
                y_val=y_va, s_val=val_scores[name],
                meta_test=test, fprs=cfg["eval"]["fpr_operating_points"],
            )
            row.update({"target": target, "split": split_kind,
                        "holdout": args.holdout})
            results.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in row.items()})

    tag = f"{split_kind}_{args.holdout or 'na'}"
    pd.DataFrame(results).to_csv(res_dir / f"baselines_{tag}.csv", index=False)
    print(f"wrote {res_dir}/baselines_{tag}.csv")


if __name__ == "__main__":
    main()
