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
# Multi-seed Task 2 (evasion) fusion A/B. Seeds 13 and 7, both arms each.
#   sbatch scripts/06_fusion_task2_multiseed.sh
#
# NOTE: no `pipefail` (it + `head`/SIGPIPE killed the previous run with exit 141).
set -eu
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source scripts/_env.sh
export HF_HOME=/home/s44srizv/MMD2_Project/qud-evasion/.hf_cache

echo "=== GPU ==="
nvidia-smi || true

for s in 13 7; do
  echo "############## SEED $s : ARM A (baseline) ##############"
  python scripts/fusion_experiment_task2.py --use_structural 0 --seed "$s"
  echo "############## SEED $s : ARM B (+ structural) ##############"
  python scripts/fusion_experiment_task2.py --use_structural 1 --seed "$s"
done

echo "############## ALL SEEDS DONE ##############"
ls -1 outputs/fusion_task2/result_struct*_seed*.json || true
echo
echo "=== Dodging F1 + macro per run ==="
python - << 'PY' || true
import json, glob
for f in sorted(glob.glob("outputs/fusion_task2/result_struct*_seed*.json")):
    d = json.load(open(f))
    dod = d["report"].get("Dodging", {}).get("f1-score", float("nan"))
    print(f"  struct={d['use_structural']} seed={d['seed']}  "
          f"macroF1={d['macro_f1']:.4f}  Dodging_F1={dod:.4f}")
PY