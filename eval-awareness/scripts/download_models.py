"""Download HF models on the LOGIN node (compute nodes may lack internet)."""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    AutoTokenizer.from_pretrained(args.model)
    AutoModelForCausalLM.from_pretrained(args.model)
    print(f"cached {args.model}")
