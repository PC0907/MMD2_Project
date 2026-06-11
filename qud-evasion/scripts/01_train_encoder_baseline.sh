#!/usr/bin/env bash
#SBATCH --job-name=clarity-encoder
#SBATCH --account=ag_yourgroup        # <- your group account (sshare -U)
#SBATCH --partition=sgpu_short        # Marvin A100 nodes, 8h limit
#SBATCH --gpus=1                      # proportional CPUs/RAM come with it
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE
##SBATCH --mail-type=END,FAIL         # needs an @uni-bonn.de address
##SBATCH --mail-user=you@uni-bonn.de
#
# Fine-tuned DeBERTa-v3-large baseline (evasion + clarity heads).
# Fits comfortably on a single A100. Submit: sbatch scripts/01_train_encoder_baseline.sh
set -euo pipefail
unset SLURM_EXPORT_ENV               # re-enable env propagation to job steps
# sbatch runs a spooled copy of this script, so locate the repo via the
# submission directory (submit from the repo root) with a fallback for
# direct `bash scripts/...` runs.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/scripts/_env.sh"

python -m qud_evasion.cli train-encoder --config configs/baseline_encoder.yaml
