# RUNBOOK — step-by-step guide with smoke tests

This is the operational guide: exact commands, what to verify after each step, and
what to do when something breaks. Follow it in order the first time through. The
guiding rule: **never submit a long job whose next analysis step you haven't already
run successfully on small/fake data.** Most "weird results" in week one are bugs.

Phases:

```
Phase 0   One-time setup (login node)                          ~30 min
Phase 1   CPU smoke test — no GPU, no model                    ~2 min
Phase 2   GPU smoke test — tiny 0.5B model, devel queue        ~10 min
Phase 3   Pilot run — 7B model, small dataset, short queue     ~1-2 h
Phase 4   Dataset expansion + full runs                        days
Phase 5   Analysis, figures, and result logging                ongoing
```

---

## Phase 0 — One-time setup (login node)

```bash
ssh <user>@bender.hpc.uni-bonn.de          # your usual login
git clone <your-repo-url> && cd eval-awareness

module load Python
python -m venv ~/venvs/evalaware
source ~/venvs/evalaware/bin/activate
pip install -r requirements.txt
pip install -e .                            # makes `eval_awareness` importable
```

Set the HuggingFace cache location **once** (models are tens of GB; check where you
have quota — home vs. scratch — with your cluster's quota command):

```bash
echo 'export HF_HOME=$HOME/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

Download models on the login node (compute nodes may not have internet):

```bash
python scripts/download_models.py --model Qwen/Qwen2.5-0.5B-Instruct   # smoke-test model
python scripts/download_models.py --model Qwen/Qwen2.5-7B-Instruct     # first real model
```

For gated models (Llama family): create a HF account, accept the license on the
model page, run `huggingface-cli login` on the login node, then download.

**Verify Phase 0** — all three must succeed before continuing:

```bash
python -c "import eval_awareness; print('package OK')"
python -c "import torch; print('torch', torch.__version__)"
ls $HF_HOME/hub | grep -i qwen        # both Qwen snapshots present
```

---

## Phase 1 — CPU smoke test (login node, ~2 min, no GPU)

This tests dataset generation and the probe code with synthetic activations — no
model involved, so failures here are pure code/env problems, cheap to debug.

```bash
python scripts/01_generate_dataset.py --n-variants 2
```

**Verify:** it prints a record count with a 50/50 eval/deploy split, and
`results/dataset.jsonl` exists. Open it and read 5 records — do the eval and deploy
framings of a pair share the same task text? They must.

Then run the probe stage on fake data:

```bash
python - << 'EOF'
import json, numpy as np
from eval_awareness.utils import read_jsonl
from eval_awareness.probes import run
recs = read_jsonl("results/dataset.jsonl")
X = np.random.RandomState(0).randn(len(recs), 12, 64).astype(np.float16)
sig = np.random.RandomState(1).randn(64)
for i, r in enumerate(recs):
    if r["framing"] == "eval": X[i, 6:, :] += 0.8 * sig      # plant signal, layers 6+
import os; os.makedirs("results/smoke_cpu", exist_ok=True)
np.savez_compressed("results/smoke_cpu/acts.npz", X=X, meta=json.dumps(recs))
run("results/smoke_cpu/acts.npz", "results/smoke_cpu/probes", C=0.1, n_folds=3, seed=0)
EOF
```

**Verify:** layers 0–5 show AUROC ≈ 0.5 and layers 6–11 show AUROC ≈ 1.0, with best
layer = 6. This confirms the probe pipeline *finds signal exactly where it exists
and nowhere else* — your leakage detector works. If layers 0–5 score high on fake
random data, something is broken; stop and debug.

---

## Phase 2 — GPU smoke test (devel queue, ~10 min)

The devel partitions (`A40devel`, `A100devel`) have a 1-hour limit and schedule
fast — they exist exactly for this. The smoke test runs ALL five stages end-to-end
with a 0.5B model and asserts on shapes, files, NaNs, and generation output.

```bash
mkdir -p logs
sbatch --partition=A40devel --time=00:30:00 --gpus=1 --ntasks=1 \
       --job-name=ea-smoke --output=logs/smoke-%j.out \
       --wrap="module load Python && source ~/venvs/evalaware/bin/activate && \
               export HF_HOME=\$HOME/hf_cache && python scripts/smoke_test.py"

squeue --me                      # wait for it (email notifications don't work on Bender)
tail -f logs/smoke-<jobid>.out   # watch live once running
```

**Verify:** the log ends with `ALL SMOKE CHECKS PASSED`. Each `[CHECK] PASS` line
tells you which stage's plumbing is confirmed. Also eyeball the two `[info]`
response snippets — they should be coherent text, not repeated tokens or garbage
(garbage at alpha=0 means a padding/template bug; garbage only at alpha=6 is fine,
that's just a tiny model being steered hard).

Common Phase-2 failures → fixes are in the Troubleshooting table at the end.

---

## Phase 3 — Pilot run (7B model, small dataset, A100short/A40short)

Purpose: get real numbers on a small scale, exercise the exact scripts you'll use
for full runs, and calibrate runtimes so you can size `--time` correctly later.

Keep the dataset small for the pilot (the Phase-1 dataset with `--n-variants 2` is
fine). Then chain the jobs with dependencies so you can submit once and walk away:

```bash
# Stage 2: extraction (~20-40 min for a few hundred prompts on A100)
jid2=$(sbatch --parsable scripts/slurm/02_extract_a100.sh Qwen/Qwen2.5-7B-Instruct)

# Stage 3: probes, only if extraction succeeded
jid3=$(sbatch --parsable --dependency=afterok:$jid2 scripts/slurm/03_probes.sh Qwen/Qwen2.5-7B-Instruct)

# Stage 4: behavior (independent of probes; only needs the model)
jid4=$(sbatch --parsable --dependency=afterok:$jid2 scripts/slurm/04_behavior_a100.sh Qwen/Qwen2.5-7B-Instruct)

# Stage 5: steering, needs the probe direction from stage 3
jid5=$(sbatch --parsable --dependency=afterok:$jid3 scripts/slurm/05_steering_a100.sh Qwen/Qwen2.5-7B-Instruct)

echo "chain: $jid2 -> ($jid3, $jid4) -> $jid5"
squeue --me
```

If a job fails, its dependents stay pending forever with reason `DependencyNeverSatisfied`;
cancel them with `scancel <jobid>` after fixing the issue.

**Verify each stage of the pilot before trusting anything:**

Stage 2 (extraction):
```bash
python - << 'EOF'
import numpy as np, json
d = np.load("results/activations/Qwen__Qwen2.5-7B-Instruct.npz", allow_pickle=True)
X = d["X"]; meta = json.loads(str(d["meta"]))
print("shape", X.shape)                      # (N, 28, 3584) for Qwen2.5-7B
print("NaNs:", np.isnan(X.astype(np.float32)).any())
print("mean |act| per layer head/tail:", np.abs(X.astype(np.float32)).mean(axis=(0,2))[[0,14,27]])
EOF
```
Expect: N = your dataset size, 28 layers, hidden 3584, no NaNs, and activation
norms that grow smoothly with depth (a sudden 100x jump at one layer = hook bug).

Stage 3 (probes): open `results/probes/<model>/layer_auroc.csv`.
Sanity expectations for a real 7B model on a *small* pilot dataset:
- early layers near 0.5, rising through middle layers — a plausible curve
- `auroc_kfold >= auroc_lofo` almost everywhere (if LOFO > k-fold consistently,
  something is wrong with the split logic or the dataset is degenerate)
- if LOFO ≈ 1.0 everywhere: vocabulary leakage — see the expectations discussion;
  fix the dataset before scaling up, do NOT proceed to steering on a leaky probe.

Stage 4 (behavior): read `results/behavior/<model>/responses.jsonl` — actually read
20 transcripts yourself. The keyword judge is a placeholder; your #1 post-pilot task
is replacing it (see Phase 4). Check `compliance_gap.csv` exists and counts match.

Stage 5 (steering): open `results/steering/<model>/dose_response.csv`.
- capability at alpha=0 should be 1.0 (all four sanity questions right); if not,
  your generation setup is broken, not the steering
- capability should degrade monotonically-ish as |alpha| grows; find the largest
  alpha keeping capability ≥ 0.75 — that's your usable steering range
- the `random` direction rows are your control; if random matches probe at every
  alpha, the effect is nonspecific (see expectations discussion).

Also record wall-clock times from `sacct -j <jobid> --format=JobID,Elapsed,State`
— you'll need them to size Phase-4 `--time` requests (ask for ~1.5x measured).

---

## Phase 4 — Scale up

Only after the pilot passes all verifications:

1. **Expand the dataset** to 25–40 families in `data/scenarios/seed_pairs.json`,
   including (a) implicit-cue eval framings with no eval vocabulary, and
   (b) realism-matched deploy controls. Re-run Phase 1 verification after editing
   (the pairing-integrity check catches JSON mistakes).
   Then regenerate at full size: `python scripts/01_generate_dataset.py --n-variants 4`
2. **Replace the keyword judge** with a real LLM judge before any result you intend
   to keep. Options on-cluster: load a second local model as judge inside
   `04_behavior_gap.py` (e.g., judge with Qwen-14B while testing Qwen-7B — never
   judge a model with itself). Hand-label 50 transcripts yourself and report
   judge-human agreement; reviewers will ask.
3. **Sweep models** by passing the model as the first sbatch argument:
   ```bash
   for M in Qwen/Qwen2.5-7B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-7B meta-llama/Llama-3.1-8B-Instruct; do
     jid2=$(sbatch --parsable scripts/slurm/02_extract_a100.sh $M)
     jid3=$(sbatch --parsable --dependency=afterok:$jid2 scripts/slurm/03_probes.sh $M)
     sbatch --dependency=afterok:$jid2 scripts/slurm/04_behavior_a100.sh $M
     sbatch --dependency=afterok:$jid3 scripts/slurm/05_steering_a100.sh $M
   done
   ```
   Note Bender's `A100medium` allows only 1 concurrent job per user — keep long jobs
   on `A100short` (8 h) where possible and split work rather than queueing mediums.
4. For 14B+ models: stay on A100, drop `batch_size` to 2–4 in the config, and if
   memory is still tight add `--gpus=2` and keep `device_map: auto` (it will shard).

---

## Phase 5 — Analysis hygiene

- `python scripts/make_figures.py --model <model>` after stages 3/5.
- Keep a `results/LOG.md`: date, git commit hash (`git rev-parse --short HEAD`),
  model, dataset size, key numbers, anything surprising. Two months from now this
  file is the difference between "I can reproduce Figure 2" and panic.
- Commit code before every submission batch; never edit code while its jobs run.
- Everything under `results/` is gitignored — back it up (rsync to your machine or
  the university storage) after each milestone.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: eval_awareness` inside a job | venv not activated in that shell / `pip install -e .` skipped | check the sbatch script sources `~/venvs/evalaware/bin/activate`; rerun `pip install -e .` |
| Job pending forever with `(Resources)` or `(Priority)` | queue busy | normal; devel queues clear fastest. Check `squeue -p A100short` to gauge load |
| Pending with `DependencyNeverSatisfied` | upstream job failed | fix upstream, `scancel` the orphan, resubmit the chain |
| `CUDA out of memory` | batch too big / model too big for the card | halve `batch_size` in config; move 14B+ to A100 (80 GB nodes if available); as a last resort load in 8-bit |
| Model download attempted inside a compute job (hangs or connection error) | compute nodes offline | download on the login node first; add `export HF_HUB_OFFLINE=1` to sbatch scripts once everything is cached |
| Gated-repo 403 for Llama | license not accepted / not logged in | accept on the HF model page, `huggingface-cli login` on login node |
| Batched generations are garbage/repetitive for short prompts | right-padding during generation | already fixed in `behavior.py` (left-padding); if you write new generation code, set `tok.padding_side="left"` |
| Probe AUROC ≈ 1.0 even in LOFO | vocabulary leakage | remove explicit eval words from framings; add bag-of-words baseline; see README hygiene |
| A single layer has exploding activation norms | hook capturing the wrong tensor for that architecture | print `type(out)` in the hook; adjust `_decoder_layers`/hook for that model family |
| Steered outputs are word salad at all alphas | alpha too large in absolute terms | scale alpha relative to mean residual norm at that layer (compute it from the cached activations: `np.linalg.norm(X[:,L,:],axis=1).mean()`), try 0.5–2x fractions |
| `sacct` shows `OUT_OF_MEMORY` (RAM, not VRAM) | loading full activation tensor on a small-memory node | request more with `--mem=64G` in the probes sbatch script |

---

## The one-page checklist (print this)

```
[ ] Phase 0: package imports, torch imports, both models cached
[ ] Phase 1: pairing integrity OK; planted-signal test localizes to layers 6+ only
[ ] Phase 2: smoke_test.py prints ALL SMOKE CHECKS PASSED on devel queue
[ ] Phase 3: activations have right shape, no NaNs, smooth norms
[ ] Phase 3: AUROC curve plausible; LOFO <= k-fold; not ~1.0 (leakage)
[ ] Phase 3: read 20 behavior transcripts with my own eyes
[ ] Phase 3: capability = 1.0 at alpha=0; random-direction control present
[ ] Phase 3: recorded wall-clock times for sizing full runs
[ ] Phase 4: dataset >= 25 families incl. implicit-cue + realism controls
[ ] Phase 4: real LLM judge installed; judge-human agreement on 50 items measured
[ ] Ongoing: results/LOG.md updated with commit hash after every batch
```
