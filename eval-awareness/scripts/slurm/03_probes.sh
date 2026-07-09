#!/bin/bash
#SBATCH --job-name=ea-probes
#SBATCH --partition=A40short
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/probes-%j.out
set -euo pipefail
mkdir -p logs
module load Python
source ~/venvs/evalaware/bin/activate
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
python scripts/03_train_probes.py --model "$MODEL"
