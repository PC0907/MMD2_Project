#!/bin/bash
#SBATCH --job-name=probemon-collect
#SBATCH --partition=A100          # adjust to Bender's GPU partition names
#SBATCH --gres=gpu:1              # 8B model; use gpu:4 for 70B
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Verify module names with `module avail` on Bender.
module load CUDA 2>/dev/null || true

source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

SUITE=${1:-workspace}
ATTACK=${2:-important_instructions}

python scripts/collect_trajectories.py \
    --config configs/default.yaml \
    --suite "$SUITE" --attack "$ATTACK"
