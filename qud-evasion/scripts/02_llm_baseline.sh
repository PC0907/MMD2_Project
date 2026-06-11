#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=clarity-llm-baseline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Direct few-shot CoT prompting baseline via vLLM.
#   sbatch scripts/02_llm_baseline.sh            # dev split (default)
#
# IMPORTANT before the first run: check how many A100s one Bender node
# has (`sinfo` / `scontrol show node`). vLLM tensor parallelism needs
# all GPUs on ONE node; if nodes have fewer than 4 A100s, lower --gpus
# here AND tensor_parallel in configs/baseline_llm.yaml together, and
# pick a backbone that fits (e.g. a ~32B model on 2 GPUs).
#
# Checkpointing: all LLM calls are cached in outputs/llm_cache.sqlite —
# if the job hits the 8h wall, just resubmit; it resumes where it stopped.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"
nvidia-smi

SPLIT=${1:-dev}
python -m qud_evasion.cli llm-baseline --config configs/baseline_llm.yaml --split "$SPLIT"