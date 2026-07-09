#!/usr/bin/env bash
# Phase 1: extraction + probes + learned head + controls.
# Extraction needs the GPU (~2-4 h for 4B+8B on one A100);
# everything after runs from the cache in minutes.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

CONFIG="${1:-configs/repr_probe.yaml}"
STAGES="${2:-extract,probe,head,controls,report}"

python scripts/repr_experiment.py --config "$CONFIG" --stages "$STAGES"
