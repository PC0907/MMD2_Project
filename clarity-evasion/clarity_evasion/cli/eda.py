#!/usr/bin/env python
"""
Exploratory data analysis — distributions + figures for the report.

    python -m clarity_evasion.cli.eda --out_dir runs/eda

Uses train+dev only; never inspects the test set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..data import make_splits, CLARITY_LABELS, EVASION_LABELS
from ..utils import get_logger

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/eda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    s = make_splits(seed=args.seed)
    train = s.train
    lines = ["# QEvasion EDA", f"- {s.sizes()}"]

    dist = train["clarity_label"].value_counts().reindex(CLARITY_LABELS).fillna(0)
    lines.append("\n## Clarity distribution (train)")
    for lbl in CLARITY_LABELS:
        c = int(dist[lbl]); lines.append(f"- {lbl}: {c} ({100*c/len(train):.1f}%)")
    lines.append(f"- imbalance ratio: {dist.max()/max(dist.min(),1):.1f}x")

    fig, ax = plt.subplots(figsize=(6, 4))
    dist.plot(kind="bar", ax=ax, color="#3b6ea5"); ax.set_title("Clarity (train)")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    fig.savefig(out / "clarity_dist.png", dpi=150); plt.close(fig)

    ev = train["evasion_label"].value_counts().reindex(EVASION_LABELS).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ev.plot(kind="bar", ax=ax, color="#a5673b"); ax.set_title("Evasion (train)")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    fig.savefig(out / "evasion_dist.png", dpi=150); plt.close(fig)

    ct = pd.crosstab(train["clarity_label"], train["evasion_label"])
    ct.to_csv(out / "clarity_evasion_crosstab.csv")

    (out / "eda_stats.md").write_text("\n".join(lines))
    log.info("\n%s", "\n".join(lines))
    log.info("figures + stats -> %s/", out)


if __name__ == "__main__":
    main()
