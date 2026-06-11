#!/usr/bin/env bash
#SBATCH --job-name=clarity
#SBATCH --partition=gpu              # <-- set to your CAISA/Bonn GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Generic SLURM launcher. Usage:
#   sbatch --job-name=reverse scripts/slurm/train.slurm configs/reverse.yaml runs/reverse-deberta
#
# Args:
#   $1 = config yaml (default configs/reverse.yaml)
#   $2 = output dir  (default runs/<config-stem>)

set -euo pipefail
mkdir -p logs

CONFIG="${1:-configs/reverse.yaml}"
OUT="${2:-runs/$(basename "${CONFIG%.yaml}")}"

# --- environment (adapt module names to your cluster) ---
# module load cuda/12.1
ENV_DIR="${ENV_DIR:-$HOME/venvs/clarity-evasion}"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# If compute nodes are offline, prefetch on a login node first, then uncomment:
# export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo ">> config=$CONFIG  out=$OUT  node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m clarity_evasion.cli.train --config "$CONFIG" --output_dir "$OUT"
python -m clarity_evasion.cli.evaluate --config "$CONFIG" \
    --model_dir "$OUT/best" --out_dir "$OUT/eval"

echo ">> done."
