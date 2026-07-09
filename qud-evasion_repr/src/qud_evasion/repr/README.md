# `repr/` — frozen-decoder representation classifiers

Phase 1 of the paper plan: extract hidden states once per model, then run
all probing/classification experiments from the disk cache (CPU-fast,
multi-seed for free).

## Run order

```bash
# 1. Everything, end to end (GPU needed only for the extract stage):
bash scripts/09_repr_experiment.sh configs/repr_probe.yaml

# 2. Iterate on probes/heads without touching the GPU again:
bash scripts/09_repr_experiment.sh configs/repr_probe.yaml probe,head,controls,report
```

Outputs land in `outputs/activations/<model>/<split>/` (caches) and
`outputs/repr/<model>/report.json` (results).

## Integration points (do these before trusting any number)

1. **Loader** — set `data.loader` in the config to your existing
   `qud_evasion.data.load` entry point so preprocessing is identical to
   the encoder/LLM arms. The fallback HF loader normalises column names
   via `COLUMN_CANDIDATES` in `data_adapter.py`; if it raises, the fix
   is a one-line candidate addition.
2. **Split** — `interview_dev_split()` in `scripts/repr_experiment.py`
   is a GroupShuffle over `interview_id`. To reproduce the report's
   exact partition, swap in `qud_evasion.data.splits`; the interface is
   just two index arrays.
3. **Regression test** — before anything new, reproduce Appendix A.3:
   run the sweep with `label_filter=(1, 2)` (Ambiguous vs Clear
   Non-Reply), pooling `last`, on Qwen3-4B. You should recover the
   ~0.43 embedding-layer / ~0.64 layer-1 / ~0.68 peak curve. If not,
   the discrepancy is in the loader or the split — resolve it before
   Phase 1 proper.

## Sanity checklist before believing results

- [ ] `truncated_examples` in `index.json` is under ~3%; otherwise raise
      `max_length` (answers in this dataset can be long).
- [ ] Probe scores on the **control task** are near chance and
      selectivity is high across the claimed layer band.
- [ ] Dev interviews in the split have zero overlap with train
      (`test_repr.py` covers the mechanism; check the count printed at
      startup is plausible, ~10% of interviews).
- [ ] The official **test** cache exists but is not read by any stage
      except the final one-shot evaluation.

## Go/no-go gate

Frozen-backbone dev macro-F1 **≥ 0.60 (clarity)** and **≥ 0.34
(evasion)** — i.e., at or above the fine-tuned DeBERTa baselines from
the report — means the "representations beat generation" thesis holds
beyond the binary boundary and Phase 2 (LoRA, 32B, transfer) is a go.
