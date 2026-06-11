#!/usr/bin/env bash
# Shared environment setup, sourced by every job script.
# Adjust the three CONFIGURE-ME blocks once for your cluster account.

# --- CONFIGURE ME (1/3): your SLURM account on Marvin -----------------
# Find it with: sshare -U
export SLURM_ACCOUNT_NAME="${SLURM_ACCOUNT_NAME:-ag_yourgroup}"

# --- CONFIGURE ME (2/3): modules -------------------------------------
# Check exact names with `module avail Python` / `module avail CUDA`.
# Uncomment / adapt as needed:
# module purge
# module load Python
# module load CUDA

# --- CONFIGURE ME (3/3): storage -------------------------------------
# Keep HF caches and outputs off your (quota-limited) $HOME if your
# group has project/scratch storage; otherwise this default is fine.
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"

# Compute nodes often have no (full) internet access. Everything is
# pre-downloaded by 00_prepare_data.sh on the login node, so jobs run
# strictly offline:
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p logs outputs
