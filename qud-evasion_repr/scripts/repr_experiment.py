#!/usr/bin/env python
"""Phase-1 runner: frozen-decoder representation classifiers.

Stages (each skippable, all reading/writing the activation cache):

  extract   forward passes, cache pooled states for train + test
  probe     logistic layer sweep, all poolings, all tasks, multi-seed
  head      learned layer-weighted head (the system config), multi-seed
  controls  Hewitt-Liang selectivity on the hardest boundary
  report    aggregate everything into outputs/repr/<model_tag>/report.json

Go/no-go criterion (from the project plan): frozen-backbone macro-F1
>= 0.60 on Task 1 dev and >= 0.34 on Task 2 dev means the paper's core
claim is alive.

Usage:
    python scripts/repr_experiment.py --config configs/repr_probe.yaml
    python scripts/repr_experiment.py --config configs/repr_probe.yaml --stages probe,head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qud_evasion.repr.data_adapter import load_examples  # noqa: E402
from qud_evasion.repr.extract import ActivationCache, ExtractionConfig, extract, load_cache  # noqa: E402
from qud_evasion.repr.heads import layer_sweep, train_layer_weighted  # noqa: E402
from qud_evasion.repr.controls import selectivity_sweep  # noqa: E402
from qud_evasion.eval.significance import aggregate_seeds, bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def interview_dev_split(cache: ActivationCache, dev_fraction: float, seed: int):
    """Interview-level train/dev split over the official training cache.

    NOTE: to reproduce the report's exact split, replace this with your
    qud_evasion.data.splits logic -- the interface is just two index
    arrays. Grouping key is interview_id, so sibling sub-questions never
    straddle the boundary.
    """
    groups = cache.groups()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_dev = max(1, int(round(len(uniq) * dev_fraction)))
    dev_set = set(uniq[:n_dev].tolist())
    dev_mask = np.array([g in dev_set for g in groups])
    return np.where(~dev_mask)[0], np.where(dev_mask)[0]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_extract(cfg: dict) -> None:
    for split in cfg["splits"]:
        examples = load_examples(
            split=split,
            loader=cfg["data"].get("loader"),
            loader_kwargs=cfg["data"].get("loader_kwargs"),
        )
        for model_name in cfg["models"]:
            ecfg = ExtractionConfig(
                model_name=model_name,
                output_root=cfg["output_root"],
                **cfg.get("extraction", {}),
            )
            extract(ecfg, examples, split=split)


def stage_probe(cfg: dict, cache: ActivationCache, tr, ev) -> dict:
    results = {}
    for task in cfg["tasks"]:
        for pooling in cfg["poolings"]:
            per_seed_best = {}
            curves = {}
            for seed in cfg["seeds"]:
                sweep = layer_sweep(
                    cache, task, tr, ev, pooling=pooling,
                    c=cfg["probe"]["C"], seed=seed,
                    class_weight=cfg["probe"].get("class_weight"),
                )
                curves[seed] = [r.macro_f1 for r in sweep]
                best = max(sweep, key=lambda r: r.macro_f1)
                per_seed_best[seed] = best.macro_f1
            results[f"{task}/{pooling}"] = {
                "best_layer_per_seed": {
                    s: int(np.argmax(c)) for s, c in curves.items()
                },
                "layer_curves": curves,
                "best_macro_f1": aggregate_seeds(per_seed_best),
            }
            agg = results[f"{task}/{pooling}"]["best_macro_f1"]
            print(f"[probe] {task:8s} {pooling:12s} macro-F1 {agg['mean']:.3f} ± {agg['std']:.3f}")
    return results


def stage_head(cfg: dict, cache: ActivationCache, tr, ev) -> dict:
    results = {}
    for task in cfg["tasks"]:
        for pooling in cfg["poolings"]:
            per_seed = {}
            details = {}
            for seed in cfg["seeds"]:
                best = train_layer_weighted(
                    cache, task, tr, ev, pooling=pooling, seed=seed,
                    class_weighted_loss=(task == "evasion"),  # report finding: weighting helps Task 2 only
                    **cfg.get("head", {}),
                )
                per_seed[seed] = best["macro_f1"]
                details[seed] = {k: v for k, v in best.items() if k != "predictions"}
            results[f"{task}/{pooling}"] = {
                "macro_f1": aggregate_seeds(per_seed),
                "per_seed_details": details,
            }
            agg = results[f"{task}/{pooling}"]["macro_f1"]
            print(f"[head]  {task:8s} {pooling:12s} macro-F1 {agg['mean']:.3f} ± {agg['std']:.3f}")
    return results


def stage_controls(cfg: dict, cache: ActivationCache, tr, ev) -> dict:
    # Hardest boundary: Ambiguous (1) vs Clear Non-Reply (2), plus full 3-way.
    out = {}
    for name, label_filter in [("amb_vs_cnr", (1, 2)), ("full_clarity", None)]:
        out[name] = selectivity_sweep(
            cache, "clarity", tr, ev,
            pooling=cfg["controls"].get("pooling", "last"),
            c=cfg["probe"]["C"], seed=cfg["seeds"][0], label_filter=label_filter,
        )
        best = max(out[name], key=lambda r: r["real_macro_f1"])
        print(f"[controls] {name}: peak real F1 {best['real_macro_f1']:.3f} "
              f"@ layer {best['layer']} (selectivity {best['selectivity']:.3f})")
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default="extract,probe,head,controls,report")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    stages = set(args.stages.split(","))

    if "extract" in stages:
        stage_extract(cfg)

    for model_name in cfg["models"]:
        print(f"\n=== {model_name} ===")
        cache = load_cache(cfg["output_root"], model_name, "train")
        tr, ev = interview_dev_split(cache, cfg["dev_fraction"], cfg["split_seed"])
        print(f"[split] {len(tr)} train / {len(ev)} dev rows "
              f"({len(np.unique(cache.groups()[ev]))} dev interviews)")

        report = {"model": model_name, "n_train": len(tr), "n_dev": len(ev)}
        if "probe" in stages:
            report["probe"] = stage_probe(cfg, cache, tr, ev)
        if "head" in stages:
            report["head"] = stage_head(cfg, cache, tr, ev)
        if "controls" in stages:
            report["controls"] = stage_controls(cfg, cache, tr, ev)

        if "report" in stages:
            out_dir = Path("outputs/repr") / model_name.replace("/", "__")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "report.json").write_text(json.dumps(report, indent=2))
            print(f"[report] wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
