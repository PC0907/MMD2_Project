"""Stage 3 runner: train probes. CPU-light (loads cached activations)."""
import argparse
from pathlib import Path
from eval_awareness.probes import run
from eval_awareness.utils import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg["model_name"]
    safe = model_name.replace("/", "__")
    npz = Path(cfg["activations_dir"]) / f"{safe}.npz"
    out = Path(cfg["results_dir"]) / "probes" / safe
    run(npz, out, C=cfg["probe"]["C"], n_folds=cfg["probe"]["n_folds"], seed=cfg["seed"])
