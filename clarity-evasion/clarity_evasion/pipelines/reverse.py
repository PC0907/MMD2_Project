"""
Reverse hierarchical pipeline: fine-tune a 9-way evasion classifier, then DERIVE the
clarity label by mapping evasion predictions up the taxonomy.

Matches the SemEval-2026 winning strategy: clarity is more reliably derived from
fine-grained evasion reasoning than learned directly. Model selection is on derived
clarity macro-F1 (the Subtask 1 metric), not evasion macro-F1.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

from ..data import (make_splits, class_weights, EVASION_LABELS, ID2EVASION,
                    CLARITY_LABELS, EVASION_TO_CLARITY_LUT)
from ..models import WeightedTrainer, build_classifier, tokenize_df, derive_clarity
from ..utils import Config, set_seed, get_logger, macro_f1, full_report, confusion_df
from .training_args import build_training_args

log = get_logger()


def _metrics_fn(agg: str):
    """Report derived-clarity macro-F1 (selection metric) and evasion macro-F1."""
    def compute(eval_pred):
        logits, labels = eval_pred                   # labels = evasion ids
        clarity_true = EVASION_TO_CLARITY_LUT[labels.astype(int)]
        clarity_pred = derive_clarity(logits, agg=agg)
        return {
            "macro_f1": f1_score(clarity_true, clarity_pred, average="macro"),
            "clarity_weighted_f1": f1_score(clarity_true, clarity_pred, average="weighted"),
            "evasion_macro_f1": f1_score(labels, logits.argmax(-1), average="macro"),
        }
    return compute


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    cfg.num_labels = len(EVASION_LABELS)
    splits = make_splits(dev_size=cfg.dev_size, seed=cfg.seed,
                         drop_inaudible=cfg.drop_inaudible)
    log.info("splits: %s", splits.sizes())

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = tokenize_df(splits.train, tok, cfg.max_len, "evasion_id")
    dev_ds = tokenize_df(splits.dev, tok, cfg.max_len, "evasion_id")

    model = build_classifier(cfg.model_name, len(EVASION_LABELS), ID2EVASION,
                             {v: k for k, v in ID2EVASION.items()})

    cw = None
    if cfg.use_class_weights:
        cw = torch.tensor(class_weights(splits.train, "evasion_id", len(EVASION_LABELS)))
        log.info("evasion class weights: %s", [round(x, 2) for x in cw.tolist()])

    trainer = WeightedTrainer(
        class_weights=cw, model=model,
        args=build_training_args(cfg, metric="macro_f1"),  # = derived clarity macro-F1
        train_dataset=train_ds, eval_dataset=dev_ds, tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics_fn(cfg.agg),
        callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience)])
    trainer.train()

    pred = trainer.predict(dev_ds)
    ev_logits, ev_true = pred.predictions, pred.label_ids
    cl_true = EVASION_TO_CLARITY_LUT[ev_true.astype(int)]
    cl_pred = derive_clarity(ev_logits, agg=cfg.agg)

    log.info("DEV derived-clarity report (%s):\n%s", cfg.agg,
             full_report(cl_true, cl_pred, CLARITY_LABELS))
    log.info("DEV clarity confusion:\n%s",
             confusion_df(cl_true, cl_pred, CLARITY_LABELS).to_string())
    for a in ("argmax", "marginal"):
        cp = derive_clarity(ev_logits, agg=a)
        log.info("  [%s] dev clarity macro-F1 = %.4f", a, macro_f1(cl_true, cp))
    log.info("DEV evasion macro-F1 = %.4f", macro_f1(ev_true, ev_logits.argmax(-1)))

    trainer.save_model(f"{cfg.output_dir}/best")
    tok.save_pretrained(f"{cfg.output_dir}/best")
    cfg.save(f"{cfg.output_dir}/best/run_config.yaml")
    return {"dev_clarity_macro_f1": macro_f1(cl_true, cl_pred),
            "dev_evasion_macro_f1": macro_f1(ev_true, ev_logits.argmax(-1))}
