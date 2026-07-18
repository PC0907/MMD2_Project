# agent-probe-monitor

White-box runtime monitoring for LLM agents: linear probes on residual-stream
activations that detect, *during* a multi-step agent trajectory, (a) that the
current observation contains a prompt injection, and (b) that the agent's next
action is attack-furthering, before the action executes.

Core research questions:

1. Do activations encode "I am being injected" / "this action violates my
   instructions" with enough signal for a linear readout?
2. Do such probes generalize across attack types (leave-one-attack-out) and
   task suites (leave-one-suite-out) they were never trained on?
3. Do white-box monitors beat black-box baselines (LlamaGuard-style output
   monitors, perplexity/log-prob heuristics) on detection AUROC and
   latency-to-detection (how many steps before the harmful action)?

## Pipeline

```
collect_trajectories.py   run AgentDojo episodes with a hooked HF model,
                          cache per-step residual activations + labels
build_dataset.py          assemble activation shards into probe datasets
train_probes.py           train logistic / mass-mean / torch probes per layer
run_eval.py               LOAO / LOSO evaluation, baselines, plots
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

AgentDojo pins move fast; if imports break, check the current API at
https://github.com/ethz-spylab/agentdojo and adjust `probemon/agentdojo_env.py`
(that file is the only place the benchmark API is touched).

## Bender HPC

See `slurm/`. Activation collection for a 70B model wants 4x A100 80GB
(device_map="auto"); probe training runs on a single GPU or CPU.

## Repo layout

```
configs/            YAML experiment configs
src/probemon/
  hooks.py          residual-stream capture via forward hooks
  agentdojo_env.py  AgentDojo integration (the only benchmark-coupled file)
  labeling.py       step- and action-level label construction
  datasets.py       activation shard I/O, split logic (LOAO / LOSO)
  probes.py         logistic, mass-mean, and torch linear probes
  baselines.py      perplexity, action log-prob, LlamaGuard output monitor
  evaluate.py       AUROC, TPR@FPR, latency-to-detection
scripts/            entry points
slurm/              sbatch templates for Bender
```

## Status

Research scaffold. Verify AgentDojo API calls against the installed version
before the first collection run.
