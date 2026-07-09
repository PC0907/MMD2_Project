"""Stage 5 runner: steering dose-response + baselines. GPU."""
import argparse
import json
from pathlib import Path
import pandas as pd
from eval_awareness import steering as st
from eval_awareness import behavior as bh
from eval_awareness.extract_activations import load_model
from eval_awareness.utils import load_config, read_jsonl, set_seed, ensure_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    set_seed(cfg["seed"])
    safe = cfg["model_name"].replace("/", "__")

    best_layer_file = Path(cfg["results_dir"]) / "probes" / safe / "best_layer.txt"
    layer = int(best_layer_file.read_text()) if best_layer_file.exists() else cfg["steering"]["layer"]
    direction = st.load_direction(
        Path(cfg["results_dir"]) / "probes" / safe / f"direction_layer{layer}.npy")

    model, tok = load_model(cfg["model_name"], cfg["dtype"], cfg["device_map"])
    rand_dir = st.random_direction(direction.numel(), seed=cfg["seed"])

    dilemmas = [r for r in read_jsonl(cfg["generated_dataset"])
                if r["kind"] == "dilemma" and r["framing"] == "deploy"]

    rows = []
    for alpha in cfg["steering"]["alphas"]:
        for name, vec in [("probe", direction), ("random", rand_dir)]:
            cap = st.capability_check(model, tok, layer, vec, alpha)
            for r in dilemmas:
                resp = st.generate_steered(
                    model, tok, r["prompt"], layer, vec, alpha,
                    max_new_tokens=cfg["steering"]["max_new_tokens"],
                    temperature=cfg["steering"]["temperature"])
                verdict = bh.keyword_judge(r["prompt"] + "\n" + resp)
                safe_flag = verdict.strip().upper().startswith("SAFE")
                rows.append({"alpha": alpha, "direction": name, "family": r["family"],
                             "id": r["id"], "safe": int(safe_flag), "capability": cap})

    out = ensure_dir(Path(cfg["results_dir"]) / "steering" / safe)
    df = pd.DataFrame(rows)
    df.to_csv(out / "steering_results.csv", index=False)
    summ = df.groupby(["direction", "alpha"]).agg(
        safe_rate=("safe", "mean"), capability=("capability", "mean")).reset_index()
    summ.to_csv(out / "dose_response.csv", index=False)
    print(summ.to_string(index=False))
