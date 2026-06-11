#!/usr/bin/env bash
#SBATCH --job-name=clarity-llm-baseline
#SBATCH --account=ag_yourgroup
#SBATCH --partition=sgpu_short        # 4x A100 on one node; 8h limit
#SBATCH --nodes=1
#SBATCH --gpus=4                      # must match tensor_parallel in the config
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE
#
# Direct few-shot CoT prompting baseline via vLLM.
#   sbatch scripts/02_llm_baseline.sh            # dev split (default)
#   sbatch scripts/02_llm_baseline.sh dev
#
# If the queue is full, the A40 nodes also work for smaller backbones:
#   sbatch --partition=mlgpu_short scripts/02_llm_baseline.sh
# (then lower llm_model / raise tensor_parallel in configs/baseline_llm.yaml
#  accordingly — A40s have less VRAM per GPU than A100s).
#
# Checkpointing: every LLM call is cached in outputs/llm_cache.sqlite,
# so if the job hits the 8h wall, simply resubmit — it resumes where it
# stopped instead of recomputing.
set -euo pipefail
unset SLURM_EXPORT_ENV
# sbatch runs a spooled copy of this script, so locate the repo via the
# submission directory (submit from the repo root) with a fallback for
# direct `bash scripts/...` runs.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"

SPLIT=${1:-dev}
python -m qud_evasion.cli llm-baseline --config configs/baseline_llm.yaml --split "$SPLIT"
