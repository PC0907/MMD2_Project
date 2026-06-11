#!/usr/bin/env bash
#SBATCH --job-name=clarity-FINAL
#SBATCH --account=ag_yourgroup
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE
#
# FINAL evaluation on the official 308-row test split. Per the course
# protocol this runs EXACTLY ONCE, after all model and prompt decisions
# are frozen on the internal dev split.
#
# Guard: batch jobs have no TTY, so confirmation is via an environment
# variable instead of a prompt. Submit deliberately with:
#   sbatch --export=ALL,CONFIRM_FINAL=yes scripts/04_final_evaluation.sh
set -euo pipefail
unset SLURM_EXPORT_ENV
if [[ "${CONFIRM_FINAL:-}" != "yes" ]]; then
  echo "Refusing to run: this is the one-time official test evaluation."
  echo "Submit with: sbatch --export=ALL,CONFIRM_FINAL=yes $0"
  exit 1
fi
# sbatch runs a spooled copy of this script, so locate the repo via the
# submission directory (submit from the repo root) with a fallback for
# direct `bash scripts/...` runs.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"

python -m qud_evasion.cli llm-baseline --config configs/baseline_llm.yaml --split official_test
python -m qud_evasion.cli qud-pipeline --config configs/qud_pipeline.yaml --split official_test
