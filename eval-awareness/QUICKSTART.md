# Quickstart

## 0. One-time setup (login node)
```bash
module load Python
python -m venv ~/venvs/evalaware && source ~/venvs/evalaware/bin/activate
pip install -r requirements.txt && pip install -e .
export HF_HOME=$HOME/hf_cache          # add to ~/.bashrc
python scripts/download_models.py --model Qwen/Qwen2.5-7B-Instruct
```

## 1. Generate dataset (login node, seconds)
```bash
python scripts/01_generate_dataset.py
```
Inspect `results/dataset.jsonl`. **Before real runs, expand
`data/scenarios/seed_pairs.json` to 25-40 families** — 8 is only enough to debug.

## 2. Debug on the devel queue FIRST (1h limit, fast to schedule)
```bash
sbatch --partition=A100devel --time=00:20:00 scripts/slurm/02_extract_a100.sh
squeue --me                            # watch it; check logs/extract-<id>.out
```
Confirm the GPU line prints a real device and activations land in
`results/activations/`. Only then submit the full short-queue job.

## 3. Full pipeline
```bash
sbatch scripts/slurm/02_extract_a100.sh
sbatch scripts/slurm/03_probes.sh          # after (2) finishes
sbatch scripts/slurm/04_behavior_a100.sh
sbatch scripts/slurm/05_steering_a100.sh   # after (3): needs the probe direction
python scripts/make_figures.py             # login node
```

## 4. Sweep models
Pass a model as the first arg: `sbatch scripts/slurm/02_extract_a100.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

## Order dependency
`02 -> 03 -> 05` is a hard chain (05 loads the probe direction from 03).
`04` is independent of 03/05. Use `sbatch --dependency=afterok:<jobid>` to chain
automatically instead of waiting manually.

## First things to make real (the keyword judge is a placeholder!)
- Swap `keyword_judge` in `behavior.py`/`05_...py` for a real LLM judge (a separate
  local model, or a frontier API). The keyword heuristic is only for plumbing tests.
- Add realism-matched deploy controls to the seed file (see README hygiene section).
