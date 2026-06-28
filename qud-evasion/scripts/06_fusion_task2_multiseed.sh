#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=fusion-t2-seeds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Multi-seed Task 2 (evasion) fusion A/B to confirm the Dodging gain.
# Runs seeds 13 and 7, both arms each, unattended (~4x 60min = ~4h).
# Seed 42 already done. Submit:
#   sbatch scripts/06_fusion_task2_multiseed.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source scripts/_env.sh
export HF_HOME=/home/s44srizv/MMD2_Project/qud-evasion/.hf_cache

echo "=== GPU ==="; nvidia-smi | head -15

for s in 13 7; do
  echo "############## SEED $s : ARM A (baseline, no structural) ##############"
  python scripts/fusion_experiment_task2.py --use_structural 0 --seed "$s"
  echo "############## SEED $s : ARM B (+ structural) ##############"
  python scripts/fusion_experiment_task2.py --use_structural 1 --seed "$s"
done

echo "############## ALL SEEDS DONE ##############"
echo "Result files:"
ls -1 outputs/fusion_task2/result_struct*_seed*.json
echo
echo "=== Dodging F1 per run (the headline class) ==="
for f in outputs/fusion_task2/result_struct*_seed*.json; do
  python - "$f" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
dod = d["report"].get("Dodging", {}).get("f1-score", float("nan"))
print(f"  struct={d['use_structural']} seed={d['seed']}  "
      f"macroF1={d['macro_f1']:.4f}  Dodging_F1={dod:.4f}")
PY
done