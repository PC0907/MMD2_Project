"""
Final TEST-set evaluation. Handles both direct (clarity head) and reverse (evasion
head) checkpoints, selected by `--pipeline`. Produces:
  - Clarity macro-F1 (Subtask 1) — for reverse, both aggregation modes.
  - Evasion macro-F1 (Subtask 2) — reverse only.
  - The required hardest-case analysis (Ambivalent vs Clear Non-Reply), with
    misclassified examples written to CSV.
  - For reverse: the fraction of evasion errors that stay within a clarity branch
    (harmless to Subtask 1) vs cross a boundary (what costs clarity macro-F1).

The test split is loaded and scored ONCE here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding)

from ..data import (make_splits, CLARITY_LABELS, CLARITY2ID, EVASION_LABELS,
                    ID2EVASION, EVASION_TO_CLARITY_LUT)
from ..models import derive_clarity
from ..utils import (Config, get_logger, macro_f1, full_report, confusion_df)

log = get_logger()


@torch.no_grad()
def _predict_logits(df, model_dir, max_len, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    ds = Dataset.from_pandas(df[["text"]], preserve_index=False).map(
        lambda b: tok(b["text"], truncation=True, max_length=max_len),
        batched=True, remove_columns=["text"])
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tok))
    out = [model(**{k: v.to(device) for k, v in b.items()}).logits.cpu().numpy()
           for b in loader]
    return np.concatenate(out, 0)


def _hardest_cases(df, cl_true, cl_pred, out_dir: Path, tag: str):
    amb, cnr = CLARITY2ID["Ambivalent"], CLARITY2ID["Clear Non-Reply"]
    m = np.isin(cl_true, [amb, cnr])
    log.info("HARDEST CASES (gold in {Ambivalent, Clear Non-Reply}, n=%d):\n%s",
             m.sum(), full_report(cl_true[m], cl_pred[m], CLARITY_LABELS))
    hd = df[m].copy()
    hd["true"] = [CLARITY_LABELS[i] for i in cl_true[m]]
    hd["pred"] = [CLARITY_LABELS[i] for i in cl_pred[m]]
    err = hd[hd["true"] != hd["pred"]]
    cols = [c for c in ["q_clean", "a_clean", "true", "pred", "evasion_label"] if c in err]
    path = out_dir / f"hardest_errors_{tag}.csv"
    err[cols].to_csv(path, index=False)
    log.info("%d hard-case errors -> %s", len(err), path)


def run(cfg: Config, model_dir: str, out_dir: str = "runs/eval") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    test = make_splits(dev_size=cfg.dev_size, seed=cfg.seed,
                       drop_inaudible=cfg.drop_inaudible).test
    cl_true = test["clarity_id"].to_numpy()
    logits = _predict_logits(test, model_dir, cfg.max_len)

    if cfg.pipeline == "direct":
        cl_pred = logits.argmax(-1)
        log.info("TEST clarity macro-F1 = %.4f", macro_f1(cl_true, cl_pred))
        log.info("clarity report:\n%s", full_report(cl_true, cl_pred, CLARITY_LABELS))
        log.info("confusion:\n%s", confusion_df(cl_true, cl_pred, CLARITY_LABELS).to_string())
        _hardest_cases(test, cl_true, cl_pred, out, "direct")
        return {"test_clarity_macro_f1": macro_f1(cl_true, cl_pred)}

    # reverse: logits are 9-way evasion
    ev_true = test["evasion_id"].to_numpy()
    has_ev = ~pd.isna(test["evasion_id"])
    for a in ("argmax", "marginal"):
        cp = derive_clarity(logits, agg=a)
        sel = " <- selected" if a == cfg.agg else ""
        log.info("TEST derived-clarity macro-F1 [%s] = %.4f%s", a, macro_f1(cl_true, cp), sel)
    cl_pred = derive_clarity(logits, agg=cfg.agg)
    log.info("clarity report (%s):\n%s", cfg.agg, full_report(cl_true, cl_pred, CLARITY_LABELS))
    log.info("clarity confusion:\n%s", confusion_df(cl_true, cl_pred, CLARITY_LABELS).to_string())

    ev_pred = logits.argmax(-1)
    if has_ev.any():
        sel = has_ev.to_numpy()
        log.info("TEST evasion macro-F1 = %.4f", macro_f1(ev_true[sel], ev_pred[sel]))
        log.info("evasion report:\n%s", full_report(ev_true[sel], ev_pred[sel], EVASION_LABELS))
        wrong = (ev_pred != ev_true) & sel
        if wrong.any():
            same = (EVASION_TO_CLARITY_LUT[ev_pred[wrong].astype(int)]
                    == EVASION_TO_CLARITY_LUT[ev_true[wrong].astype(int)])
            log.info("Of %d evasion errors: %.1f%% stay within-branch (harmless to "
                     "Subtask 1), %.1f%% cross a clarity boundary.",
                     wrong.sum(), 100*same.mean(), 100*(1-same.mean()))

    _hardest_cases(test, cl_true, cl_pred, out, "reverse")
    return {"test_clarity_macro_f1": macro_f1(cl_true, cl_pred)}
