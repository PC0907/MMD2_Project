#!/bin/bash
#SBATCH --job-name=probemon-train
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

source .venv/bin/activate

SPLIT=${1:-loao}
HOLDOUT=${2:-important_instructions}

python scripts/train_probes.py --config configs/default.yaml \
    --split "$SPLIT" --holdout "$HOLDOUT"
python scripts/run_eval.py --config configs/default.yaml \
    --split "$SPLIT" --holdout "$HOLDOUT" \
    --baselines perplexity action_logprob
