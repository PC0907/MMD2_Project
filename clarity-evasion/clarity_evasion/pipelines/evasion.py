"""
Bonus pipeline: Level-2 fine-grained evasion classification.

Modes:
  baseline : 9-way evasion classification from text alone.
  joint    : multitask shared encoder + clarity auxiliary head (joint learning).
             Reports the evasion head; the clarity head regularises the shared
             representation. Compare baseline vs joint evasion macro-F1 to answer
             whether clarity labels help strategy detection.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer

from ..data import make_splits, EVASION_LABELS, ID2EVASION, CLARITY_LABELS
from ..models import JointClarityEvasionModel, build_classifier, tokenize_df
from ..utils import Config, set_seed, get_logger, macro_f1, full_report
from .training_args import build_training_args

log = get_logger()


def _metrics_fn():
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"macro_f1": f1_score(labels, preds, average="macro"),
                "weighted_f1": f1_score(labels, preds, average="weighted")}
    return compute


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    joint = cfg.evasion_mode == "joint"
    cfg.num_labels = len(EVASION_LABELS)
    splits = make_splits(dev_size=cfg.dev_size, seed=cfg.seed,
                         drop_inaudible=cfg.drop_inaudible)
    log.info("splits: %s | mode=%s", splits.sizes(), cfg.evasion_mode)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    extra = ["clarity_id"] if joint else None
    train_ds = tokenize_df(splits.train, tok, cfg.max_len, "evasion_id", extra)
    dev_ds = tokenize_df(splits.dev, tok, cfg.max_len, "evasion_id", extra)
    if joint:  # rename the extra label col the model expects
        train_ds = train_ds.rename_column("clarity_id", "clarity_labels")
        dev_ds = dev_ds.rename_column("clarity_id", "clarity_labels")

    if joint:
        model = JointClarityEvasionModel(cfg.model_name, len(EVASION_LABELS),
                                         len(CLARITY_LABELS), aux_weight=cfg.aux_weight)
        label_names = ["labels", "clarity_labels"]
    else:
        model = build_classifier(cfg.model_name, len(EVASION_LABELS), ID2EVASION,
                                 {v: k for k, v in ID2EVASION.items()})
        label_names = ["labels"]

    trainer = Trainer(
        model=model, args=build_training_args(cfg, "macro_f1", label_names),
        train_dataset=train_ds, eval_dataset=dev_ds, tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics_fn(),
        callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience)])
    trainer.train()

    pred = trainer.predict(dev_ds)
    y_pred = pred.predictions.argmax(-1)
    log.info("DEV evasion report (%s):\n%s", cfg.evasion_mode,
             full_report(pred.label_ids, y_pred, EVASION_LABELS))
    score = macro_f1(pred.label_ids, y_pred)
    log.info("DEV evasion macro-F1 (%s) = %.4f", cfg.evasion_mode, score)

    trainer.save_model(f"{cfg.output_dir}/best")
    tok.save_pretrained(f"{cfg.output_dir}/best")
    cfg.save(f"{cfg.output_dir}/best/run_config.yaml")
    return {"dev_evasion_macro_f1": score, "mode": cfg.evasion_mode}
