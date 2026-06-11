#!/usr/bin/env bash
# Pre-download dataset + model on a login node so compute nodes can run offline.
set -euo pipefail
MODEL="${1:-microsoft/deberta-v3-base}"
python - "$MODEL" << 'PY'
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
m = sys.argv[1]
print("fetching dataset ailsntua/QEvasion ...")
load_dataset("ailsntua/QEvasion")
print(f"fetching model {m} ...")
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
print("cached to ~/.cache/huggingface")
PY
