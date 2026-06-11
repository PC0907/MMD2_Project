# Design notes

Short rationale for the structure, for anyone reading or extending the code.

## Why config-driven
Every run is fully specified by a YAML file in `configs/`. CLI flags override
individual fields for quick sweeps. The resolved config is saved to
`<output_dir>/best/run_config.yaml` so every checkpoint records exactly how it was
produced — reproducibility without digging through shell history.

## Why the package is split the way it is
- `data/taxonomy.py` — label spaces and the evasion→clarity hierarchy. Pure NumPy,
  zero heavy deps, so it imports anywhere. This is the one place the taxonomy is
  defined; everything else imports from here.
- `data/dataset.py` — loading, cleaning, splits, class weights. Isolated from the
  taxonomy so the latter stays import-light.
- `models/aggregation.py` — the evasion→clarity derivation (argmax / marginal).
  Pure NumPy and unit-tested, because it is the conceptual core of the reverse
  pipeline and must be provably correct.
- `models/components.py` — torch-backed pieces (`WeightedTrainer`, the joint model,
  tokenization). Imported lazily so non-torch code paths stay usable.
- `pipelines/*` — thin orchestration. Each `run(cfg)` wires data + model + trainer
  and returns a small results dict. No business logic lives in the CLI.
- `cli/*` — argument parsing only; delegates to pipelines.

## Why lazy imports
`from clarity_evasion.data import taxonomy` and `from clarity_evasion.models import
derive_clarity` work without torch or `datasets` installed. This keeps the unit tests
fast and dependency-free, and lets analysis notebooks use the taxonomy without
spinning up the full stack. Implemented via module-level `__getattr__`.

## The marginal aggregation, concretely
If the model is confident it's *some* Ambivalent sub-strategy but unsure which
(Dodging vs Deflection vs General), the probability mass is split three ways. A
single Clear-Non-Reply label can then win top-1 despite the branch as a whole being
more probable. `argmax` would leak across the clarity boundary; `marginal` sums the
branch mass first and recovers the correct clarity label. `eval` reports both modes,
and what fraction of evasion errors stay within-branch (harmless to Subtask 1) vs
cross a boundary (the ones that actually cost clarity macro-F1).

## Extending
- New model: change `model_name` in a config. Anything `AutoModelForSequenceClassification`
  supports works.
- New pipeline: add `pipelines/<name>.py` with a `run(cfg)`, register it in
  `pipelines/__init__.py::TRAIN_PIPELINES`.
- New aggregation: add a branch to `models/aggregation.derive_clarity` and a test.
