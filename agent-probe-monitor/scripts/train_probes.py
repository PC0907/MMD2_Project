#!/usr/bin/env python
"""Train probes per (target, segment, layer) under a chosen split and
report validation/test metrics. Saves a results CSV plus fitted directions.

Usage:
    python scripts/train_probes.py --config configs/default.yaml \
        --split loao --holdout important_instructions
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from probemon.datasets import load_meta, load_features, make_split
from probemon.probes import build_probe
from probemon.evaluate import summarize
from probemon.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default=None, help="loao | loso | random")
    ap.add_argument("--holdout", default=None,
                    help="attack (loao) or suite (loso) held out for test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["collection"]["seed"])
    out_dir = Path(cfg["collection"]["out_dir"])
    res_dir = Path(cfg["eval"]["out_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    split_kind = args.split or cfg["dataset"]["split"]
    meta = load_meta(out_dir).sort_values(["episode_id", "step"])
    train, val, test = make_split(
        meta, split_kind, holdout=args.holdout,
        val_fraction=cfg["dataset"]["val_fraction"],
        seed=cfg["collection"]["seed"],
    )
    print(f"split={split_kind} holdout={args.holdout} "
          f"train={len(train)} val={len(val)} test={len(test)}")

    results = []
    probes_out = {}
    for target in cfg["dataset"]["targets"]:
        # Injection presence is read off the observation segment; whether the
        # action is attack-furthering is read off the action segment.
        segment = "obs" if target == "injection_present" else "act"
        y_tr, y_va, y_te = (d[target].to_numpy() for d in (train, val, test))
        for layer in cfg["model"]["layers"]:
            X_tr = load_features(out_dir, train, segment, layer)
            X_va = load_features(out_dir, val, segment, layer)
            X_te = load_features(out_dir, test, segment, layer)

            probe = build_probe(cfg["probe"]["kind"], **cfg["probe"])
            probe.fit(X_tr, y_tr)
            s_va, s_te = probe.score(X_va), probe.score(X_te)

            row = summarize(
                name=f"{cfg['probe']['kind']}/{target}/{segment}/L{layer}",
                y_test=y_te, s_test=s_te, y_val=y_va, s_val=s_va,
                meta_test=test, fprs=cfg["eval"]["fpr_operating_points"],
            )
            row.update({"target": target, "segment": segment, "layer": layer,
                        "split": split_kind, "holdout": args.holdout})
            results.append(row)
            probes_out[(target, segment, layer)] = probe
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in row.items()})

    tag = f"{split_kind}_{args.holdout or 'na'}"
    pd.DataFrame(results).to_csv(res_dir / f"probes_{tag}.csv", index=False)
    with open(res_dir / f"probes_{tag}.pkl", "wb") as f:
        pickle.dump(probes_out, f)
    print(f"wrote {res_dir}/probes_{tag}.csv")


if __name__ == "__main__":
    main()
