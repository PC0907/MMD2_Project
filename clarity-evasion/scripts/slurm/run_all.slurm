#!/usr/bin/env bash
#SBATCH --job-name=clarity-all
#SBATCH --partition=gpu              # <-- set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3

# Run all four experiments in parallel as a SLURM array:
#   sbatch scripts/slurm/run_all.slurm
# Each array task trains + evaluates one config.

set -euo pipefail
mkdir -p logs

CONFIGS=(
  "configs/direct.yaml"
  "configs/reverse.yaml"
  "configs/evasion_baseline.yaml"
  "configs/evasion_joint.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUT="runs/$(basename "${CONFIG%.yaml}")"

ENV_DIR="${ENV_DIR:-$HOME/venvs/clarity-evasion}"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export TOKENIZERS_PARALLELISM=false
# export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # if offline

echo ">> [$SLURM_ARRAY_TASK_ID] config=$CONFIG out=$OUT"
python -m clarity_evasion.cli.train --config "$CONFIG" --output_dir "$OUT"

# evasion configs are evaluated by their own pipeline; direct/reverse get test eval
case "$CONFIG" in
  *direct*)  python -m clarity_evasion.cli.evaluate --config "$CONFIG" --pipeline direct  --model_dir "$OUT/best" --out_dir "$OUT/eval" ;;
  *reverse*) python -m clarity_evasion.cli.evaluate --config "$CONFIG" --pipeline reverse --model_dir "$OUT/best" --out_dir "$OUT/eval" ;;
esac
echo ">> done [$SLURM_ARRAY_TASK_ID]."
