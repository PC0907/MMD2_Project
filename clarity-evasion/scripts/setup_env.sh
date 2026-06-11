#!/usr/bin/env bash
# One-time environment setup. Run on a login node (needs internet).
set -euo pipefail

ENV_DIR="${1:-$HOME/venvs/clarity-evasion}"
echo ">> creating venv at $ENV_DIR"
python -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo ">> done. activate with: source $ENV_DIR/bin/activate"
