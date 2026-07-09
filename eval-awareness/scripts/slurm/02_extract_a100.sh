#!/bin/bash
#SBATCH --job-name=ea-extract
#SBATCH --partition=A100short
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/extract-%j.out

# Bender: no --account needed. Debug first on A100devel (--time=00:30:00).
set -euo pipefail
mkdir -p logs
module load Python
source ~/venvs/evalaware/bin/activate
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python scripts/02_extract_activations.py --model "$MODEL"
