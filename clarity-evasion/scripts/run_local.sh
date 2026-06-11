#!/usr/bin/env bash
# Quick local run (single GPU / CPU) of the full main pipeline: EDA -> train -> eval.
set -euo pipefail
CONFIG="${1:-configs/reverse.yaml}"
OUT="${2:-runs/reverse-deberta}"

python -m clarity_evasion.cli.eda --out_dir "$OUT/eda"
python -m clarity_evasion.cli.train --config "$CONFIG" --output_dir "$OUT"
python -m clarity_evasion.cli.evaluate --config "$CONFIG" \
    --model_dir "$OUT/best" --out_dir "$OUT/eval"
