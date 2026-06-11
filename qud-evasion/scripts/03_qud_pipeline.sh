#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=20:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=clarity-qud
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Full QUD pipeline (Stage 1 reconstruction -> Stage 2 relations ->
# Stage 3 classification).
#   sbatch scripts/03_qud_pipeline.sh                  # dev split, rule head
#   sbatch scripts/03_qud_pipeline.sh dev learned      # learned head
#
# Notes for Bender:
# - The medium queues allow only ONE concurrent job per user; extra
#   submissions just wait in the queue.
# - Often faster in practice: A100short (8h) + resubmit — the sqlite
#   prompt cache resumes the run for free after a wall-time kill.
# - For quick prompt iterations on a small slice, use the devel queue:
#   sbatch --partition=A100devel --time=00:59:00 scripts/03_qud_pipeline.sh
# - Same GPUs-per-node caveat as 02: keep --gpus == tensor_parallel.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi

SPLIT=${1:-dev}
HEAD=${2:-rule}
CONFIG=configs/qud_pipeline.yaml
if [[ "$HEAD" == "learned" ]]; then
  TMP=$(mktemp --suffix=.yaml)
  sed 's/^head: rule/head: learned/' "$CONFIG" > "$TMP"
  CONFIG="$TMP"
fi
python -m qud_evasion.cli qud-pipeline --config "$CONFIG" --split "$SPLIT"