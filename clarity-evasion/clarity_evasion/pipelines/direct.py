"""
Direct pipeline: fine-tune an encoder to predict the 3-way clarity label directly.

This is the honest baseline and the known encoder ceiling (~0.75-0.81 macro-F1 on
Subtask 1 per the SemEval-2026 overview). Use it as the ablation that the reverse
pipeline is compared against.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

from ..data import (make_splits, class_weights, CLARITY_LABELS, ID2CLARITY,
                    CLARITY2ID)
from ..models import WeightedTrainer, build_classifier, tokenize_df
from ..utils import Config, set_seed, get_logger, macro_f1, weighted_f1, full_report, confusion_df
from .training_args import build_training_args

log = get_logger()


def _metrics_fn():
    from sklearn.metrics import f1_score
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        out = {"macro_f1": f1_score(labels, preds, average="macro"),
               "weighted_f1": f1_score(labels, preds, average="weighted")}
        for i, name in ID2CLARITY.items():
            out[f"f1_{name.replace(' ', '_')}"] = f1_score(
                labels, preds, labels=[i], average="macro")
        return out
    return compute


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    cfg.num_labels = len(CLARITY_LABELS)
    splits = make_splits(dev_size=cfg.dev_size, seed=cfg.seed,
                         drop_inaudible=cfg.drop_inaudible)
    log.info("splits: %s", splits.sizes())

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = tokenize_df(splits.train, tok, cfg.max_len, "clarity_id")
    dev_ds = tokenize_df(splits.dev, tok, cfg.max_len, "clarity_id")

    model = build_classifier(cfg.model_name, len(CLARITY_LABELS), ID2CLARITY,
                             {v: k for k, v in ID2CLARITY.items()})

    cw = None
    if cfg.use_class_weights:
        cw = torch.tensor(class_weights(splits.train, "clarity_id", len(CLARITY_LABELS)))
        log.info("class weights: %s", [round(x, 3) for x in cw.tolist()])

    trainer = WeightedTrainer(
        class_weights=cw, model=model,
        args=build_training_args(cfg, metric="macro_f1"),
        train_dataset=train_ds, eval_dataset=dev_ds, tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics_fn(),
        callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience)])
    trainer.train()

    pred = trainer.predict(dev_ds)
    y_pred = pred.predictions.argmax(-1)
    log.info("DEV clarity report:\n%s",
             full_report(pred.label_ids, y_pred, CLARITY_LABELS))
    log.info("DEV confusion:\n%s",
             confusion_df(pred.label_ids, y_pred, CLARITY_LABELS).to_string())

    trainer.save_model(f"{cfg.output_dir}/best")
    tok.save_pretrained(f"{cfg.output_dir}/best")
    cfg.save(f"{cfg.output_dir}/best/run_config.yaml")
    return {"dev_macro_f1": macro_f1(pred.label_ids, y_pred),
            "dev_weighted_f1": weighted_f1(pred.label_ids, y_pred)}
