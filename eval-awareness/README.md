# Evaluation Awareness in Open-Weight Reasoning Models

**Research question:** Can linear probes on internal activations detect when an LLM
believes it is being *evaluated* vs. *deployed* — and does steering that internal
direction causally change safety-relevant behavior?

Why it matters: safety evaluations assume models behave the same under test and in
the wild. If models internally represent "this is a test" and modulate behavior on
it, evals systematically overestimate safety. We measure this **internally**
(probes), not just behaviorally (prompt framing), and then intervene **causally**
(activation steering).

## Pipeline

```
Stage 1  Dataset          Matched scenario pairs: same task, eval-framed vs deploy-framed
Stage 2  Activations      Forward passes with hooks, cache residual stream per layer
Stage 3  Probes           Per-layer logistic probes (eval vs deploy)
                          → layerwise AUROC + leave-one-family-out generalization
Stage 4  Compliance gap   Behavior difference on dilemma tasks, conditioned on probe score
Stage 5  Steering         Add/subtract the probe direction during generation
                          → does suppressing "test awareness" change behavior? (causal)
```

Stages 1–3 are the direct analogue of probe-based error detection for extraction
tasks (train a probe on activations to detect a latent property). Stages 4–5 are the
new, safety-specific parts.

## Repo layout

```
configs/default.yaml              All experiment settings
data/scenarios/seed_pairs.json    Hand-written matched scenario families (EXTEND THESE)
src/eval_awareness/
    scenarios.py                  Templating → paired dataset
    extract_activations.py        Hooked forward passes → cached activations
    probes.py                     Probe training + cross-family generalization
    behavior.py                   Behavioral scoring / compliance gap
    steering.py                   Activation steering with the probe direction
    utils.py
scripts/
    01_generate_dataset.py … 05_steering_experiment.py
    download_models.py            Run on LOGIN node (compute nodes may lack internet)
    slurm/                        sbatch scripts for Bender A40/A100 queues
```

## Setup on Bender (Uni Bonn HPC)

```bash
# login node
module load Python
python -m venv ~/venvs/evalaware && source ~/venvs/evalaware/bin/activate
pip install -r requirements.txt
pip install -e .

export HF_HOME=$HOME/hf_cache          # add to ~/.bashrc; check disk quota
python scripts/download_models.py --model Qwen/Qwen2.5-7B-Instruct
```

Run:

```bash
python scripts/01_generate_dataset.py                    # fast, login node is fine
sbatch scripts/slurm/02_extract_a100.sh                  # GPU
sbatch scripts/slurm/03_probes.sh                        # CPU-light
sbatch scripts/slurm/04_behavior_a100.sh                 # GPU (generation)
sbatch scripts/slurm/05_steering_a100.sh                 # GPU (generation + hooks)
```

Bender queue cheat-sheet: debug on `A100devel`/`A40devel` (1 h limit), real runs on
`A100short` (8 h) or `A100medium` (24 h, max 1 concurrent job/user). A40 (48 GB)
fits 7–8B models in bf16; use A100 for 14B+ or bigger batches. No `--account`
needed on Bender. E-mail notifications don't work on Bender — use `squeue --me`.

## Model ladder

| Model | Size | Notes |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 7B | fast iteration, fits A40 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | reasoning model → CoT analyses |
| meta-llama/Llama-3.1-8B-Instruct | 8B | family diversity (gated: accept license) |
| Qwen/Qwen2.5-14B-Instruct | 14B | scale trend, A100 |

## Scientific hygiene (reviewers WILL ask)

- **Leakage:** probes are evaluated leave-one-family-out, never only on held-out
  samples of seen families — otherwise the probe may detect vocabulary, not the
  eval/deploy concept. Default in `probes.py`.
- **Artificiality confound:** eval-framed prompts may just be weirder. Extend
  `seed_pairs.json` with realism-matched controls and report them separately.
- **Steering sanity:** run the capability sanity set (built into
  `05_steering_experiment.py`) so "behavior changed" ≠ "model broke".
- **Random-direction baseline:** included in the steering script.

## Paper milestones

1. Layerwise probe AUROC across ≥3 models (Fig. 1)
2. Leave-one-family-out generalization matrix (Fig. 2)
3. Compliance gap vs. probe score (Fig. 3)
4. Steering dose–response curves (Fig. 4 — the headline causal result)

## Position against

Abdelnabi & Salem 2025 (linear control of test awareness); LURE 2026 (live-usage
replay evals); Greenblatt et al. 2024 (alignment faking); Lynch et al. 2025 (agentic
misalignment); Meinke et al. 2024 (in-context scheming); Chen et al. 2025 (reasoning
models don't always say what they think). Contribution: systematic multi-model
internal measurement + compliance-gap quantification + causal steering.
