"""Stage 4 runner: generate + judge + compliance gap. GPU."""
import argparse
from pathlib import Path
from eval_awareness import behavior as bh
from eval_awareness.extract_activations import load_model
from eval_awareness.utils import load_config, read_jsonl, set_seed, write_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge", choices=["keyword"], default="keyword",
                    help="swap in an LLM/API judge for real results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    set_seed(cfg["seed"])

    records = [r for r in read_jsonl(cfg["generated_dataset"]) if r["kind"] == "dilemma"]
    model, tok = load_model(cfg["model_name"], cfg["dtype"], cfg["device_map"])
    gen = bh.generate_responses(model, tok, records,
                                max_new_tokens=cfg["steering"]["max_new_tokens"],
                                temperature=cfg["steering"]["temperature"], n_samples=1)
    judge = bh.keyword_judge  # replace with API judge
    scored = bh.score_with_judge(gen, judge)

    safe = cfg["model_name"].replace("/", "__")
    out = Path(cfg["results_dir"]) / "behavior" / safe
    write_jsonl(out / "responses.jsonl", scored)
    bh.compliance_gap(scored, out)
