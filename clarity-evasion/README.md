# Political Clarity & Evasion Detection

Response-clarity and evasion-strategy classification on political interview
question–answer pairs, built on the **QEvasion** dataset
([`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion)) from
the SemEval-2026 Task 6 *CLARITY* shared task (Thomas et al., 2024, 2026).

Given a `(question, answer)` pair from a presidential interview, the system predicts:

- **Subtask 1 (Clarity, Level 1):** `Clear Reply`, `Ambivalent`, or `Clear Non-Reply`.
- **Subtask 2 (Evasion, Level 2, bonus):** one of 9 fine-grained evasion strategies
  (`Explicit`, `Implicit`, `Dodging`, `General`, `Deflection`, `Partial/half-answer`,
  `Declining to answer`, `Claims ignorance`, `Clarification`).

Primary metric: **macro-F1** (the classes are heavily imbalanced).

---

## Approaches

Two clarity systems plus a bonus evasion system, all sharing one codebase:

| Pipeline   | Idea | Role |
|------------|------|------|
| `direct`   | Fine-tune an encoder to predict clarity directly (3-way). | Baseline / ablation. The known encoder ceiling. |
| `reverse`  | Fine-tune a 9-way evasion classifier, then **derive** clarity by mapping evasion predictions up the taxonomy. | **Main system.** Matches the SemEval-2026 winning strategy. |
| `evasion`  | Level-2 classification, in `baseline` (flat 9-way) or `joint` (multitask with a clarity auxiliary head) mode. | Bonus / Subtask 2. |

**Why reverse?** The SemEval-2026 overview reports that coarse clarity labels are
more reliably *derived* from fine-grained evasion reasoning than learned directly,
and that hierarchical decomposition was the single most consistent differentiator
between strong and weak systems. The `reverse` pipeline implements that, and selects
the model on *derived clarity* macro-F1 (the Subtask 1 metric).

The reverse pipeline supports two ways to turn evasion logits into a clarity label:
- `argmax` — take the top-1 evasion class, map it to its clarity branch.
- `marginal` — softmax the evasion logits, sum probability mass within each clarity
  branch, then argmax over the 3 branches. More robust to within-branch uncertainty.

---

## Repository layout

```
clarity-evasion/
├── configs/                  # YAML experiment configs (one per run)
├── clarity_evasion/          # installable package
│   ├── data/                 # taxonomy (label spaces + hierarchy) and dataset loading
│   ├── models/               # aggregation (numpy) + torch components (trainer, joint model)
│   ├── pipelines/            # direct / reverse / evasion / evaluate orchestration
│   ├── cli/                  # train, evaluate, eda entry points
│   └── utils/                # config, seeding, metrics
├── scripts/                  # setup, prefetch, local runner, SLURM jobs
├── tests/                    # unit tests (taxonomy, aggregation, data — no GPU/network)
└── docs/                     # design notes
```

Each leaf module pulls in only what it needs: the taxonomy and aggregation logic
import with just NumPy, so they (and the tests for them) run anywhere; torch and
`datasets` are loaded lazily.

---

## Setup

```bash
# one-time, on a machine/login-node with internet
bash scripts/setup_env.sh                 # creates a venv, installs deps + this package
source ~/venvs/clarity-evasion/bin/activate

# if your GPU/compute nodes are offline, cache dataset+model first:
bash scripts/prefetch.sh microsoft/deberta-v3-base
```

## Quickstart

```bash
make eda          # distributions + figures -> runs/eda/
make reverse      # train + evaluate the main system
make direct       # train + evaluate the baseline
make evasion      # train evasion baseline + joint
make test         # run unit tests (no GPU/network needed)
```

Or call the CLI directly (flags override the YAML):

```bash
python -m clarity_evasion.cli.train    --config configs/reverse.yaml --fp16
python -m clarity_evasion.cli.evaluate --config configs/reverse.yaml \
    --pipeline reverse --model_dir runs/reverse-deberta/best --out_dir runs/reverse-deberta/eval
```

## Running on the cluster (SLURM)

```bash
# single experiment
sbatch --job-name=reverse scripts/slurm/train.slurm configs/reverse.yaml runs/reverse-deberta

# all four experiments as an array job
sbatch scripts/slurm/run_all.slurm
```

Edit `--partition` in the SLURM headers to your CAISA/Bonn partition, and set
`ENV_DIR` if your venv lives elsewhere.

---

## Data protocol (important)

- QEvasion ships **train (3,448)** and **test (308)** splits only — there is **no
  official dev split**. We carve a stratified internal dev split out of train; the
  test split is the untouched final-evaluation set, read exactly once by `evaluate`.
- The classification unit is the extracted **sub-question** (`question`), not the
  raw interviewer turn; multi-part questions are already decomposed into rows.
- **Two test sets exist in the literature.** The HF `test` split (308 rows) is the
  *original* Thomas et al. 2024 test set. The SemEval-2026 leaderboard used a *new*
  237-pair evaluation set. Numbers produced here are on the HF split and are **not**
  directly comparable to the 0.89 / 0.68 leaderboard figures.

## Reference points (SemEval-2026 overview)

- Plain encoder fine-tuning on Subtask 1 saturates around **0.75–0.81** macro-F1;
  top LLM systems reach **0.89**. Treat the DeBERTa numbers here as a floor.
- Subtask 2 is much harder: best system **0.68**, best encoder-only **~0.50**.
- Standard imbalance fixes (class weights, focal loss, oversampling) help but do not
  close the gap alone.

## Outputs

Each run writes to `runs/<name>/`:
- `best/` — best checkpoint (by dev macro-F1) + tokenizer + `run_config.yaml`.
- `eval/` — test-set reports and `hardest_errors_*.csv` (the required Ambivalent vs
  Clear Non-Reply error analysis).

## Tests

```bash
pytest -q tests/
```
The taxonomy and aggregation tests run without GPU or network. The data tests need
the `datasets`/`sklearn` packages (but not network — they exercise the cleaning
helpers on synthetic rows).

## Citation

If you use this code or the dataset, cite the dataset/task papers:

```bibtex
@inproceedings{thomas-etal-2024-never,
  title     = {{``}I Never Said That{''}: A dataset, taxonomy and baselines on response clarity classification},
  author    = {Thomas, Konstantinos and Filandrianos, Giorgos and Lymperaiou, Maria and Zerva, Chrysoula and Stamou, Giorgos},
  booktitle = {Findings of the ACL: EMNLP 2024},
  year      = {2024},
  pages     = {5204--5233},
}
```

## License

MIT — see [LICENSE](LICENSE). (Set the copyright holder name before submitting.)
