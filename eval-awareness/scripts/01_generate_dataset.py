"""Stage 1 runner: expand seed families into the paired dataset. Fast; login node OK."""
import argparse
from eval_awareness.scenarios import generate_dataset
from eval_awareness.utils import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-variants", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    recs = generate_dataset(
        seed_path=f"{cfg['scenario_dir']}/seed_pairs.json",
        out_path=cfg["generated_dataset"],
        n_variants_per_context=args.n_variants,
        seed=cfg["seed"],
    )
    n_eval = sum(r["framing"] == "eval" for r in recs)
    print(f"wrote {len(recs)} records ({n_eval} eval / {len(recs)-n_eval} deploy) "
          f"-> {cfg['generated_dataset']}")
