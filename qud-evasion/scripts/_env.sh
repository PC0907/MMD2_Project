#!/usr/bin/env bash
# Shared environment setup, sourced by every job script.
# Bender cluster: no SLURM account needed (unlike Marvin/Bonna).

# --- CONFIGURE ME (1/2): modules --------------------------------------
# Check exact names with `module avail Python` / `module avail CUDA`.
# Uncomment / adapt as needed (or source your existing setup_env.sh):
# module purge
# module load Python
# module load CUDA
# source ~/NLP_Lab/setup_env.sh

# --- CONFIGURE ME (2/2): storage ---------------------------------------
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"

# Compute jobs run strictly offline; 00_prepare_data.sh pre-downloads
# data and model weights on the login node.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$PROJECT_DIR"
source ~/mmd2/bin/activate
mkdir -p logs outputs