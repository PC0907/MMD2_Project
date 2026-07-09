#!/bin/bash
#SBATCH --job-name=ea-behavior
#SBATCH --partition=A100short
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/behavior-%j.out
set -euo pipefail
mkdir -p logs
module load Python
source ~/venvs/evalaware/bin/activate
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
python scripts/04_behavior_gap.py --model "$MODEL"
