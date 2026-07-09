"""Stage 2 runner: cache activations. GPU."""
import argparse
from eval_awareness import extract_activations as ea
from eval_awareness.utils import load_config, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="override cfg model_name")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    set_seed(cfg["seed"])
    ea.run(cfg)
