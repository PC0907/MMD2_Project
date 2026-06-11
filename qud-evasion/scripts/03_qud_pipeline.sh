#!/usr/bin/env bash
#SBATCH --job-name=clarity-qud
#SBATCH --account=ag_yourgroup
#SBATCH --partition=sgpu_medium       # full pipeline over train+dev; 1-day limit
#SBATCH --nodes=1
#SBATCH --gpus=4                      # must match tensor_parallel in the config
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE
#
# Full QUD pipeline (Stage 1 reconstruction -> Stage 2 relations ->
# Stage 3 classification).
#   sbatch scripts/03_qud_pipeline.sh                  # dev split, rule head
#   sbatch scripts/03_qud_pipeline.sh dev learned      # learned head
#
# Note: sgpu_medium allows 1 concurrent job per user. For quick prompt
# iterations on a small slice, use the devel queue instead:
#   sbatch --partition=sgpu_devel --time=00:59:00 scripts/03_qud_pipeline.sh
#
# Checkpointing: the sqlite prompt cache (outputs/llm_cache.sqlite) makes
# resubmission after a wall-time kill resume for free, and head ablations
# (rule vs learned) reuse all Stage 1/2 outputs at zero GPU cost.
set -euo pipefail
unset SLURM_EXPORT_ENV
# sbatch runs a spooled copy of this script, so locate the repo via the
# submission directory (submit from the repo root) with a fallback for
# direct `bash scripts/...` runs.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"

SPLIT=${1:-dev}
HEAD=${2:-rule}
CONFIG=configs/qud_pipeline.yaml
if [[ "$HEAD" == "learned" ]]; then
  TMP=$(mktemp --suffix=.yaml)
  sed 's/^head: rule/head: learned/' "$CONFIG" > "$TMP"
  CONFIG="$TMP"
fi
python -m qud_evasion.cli qud-pipeline --config "$CONFIG" --split "$SPLIT"
