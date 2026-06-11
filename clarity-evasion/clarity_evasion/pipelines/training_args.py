"""Shared HF TrainingArguments construction from a Config."""
from __future__ import annotations

from transformers import TrainingArguments

from ..utils.config import Config


def build_training_args(cfg: Config, metric: str = "macro_f1",
                        label_names: list[str] | None = None) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=True,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16,
        report_to="none",
        seed=cfg.seed,
        label_names=label_names or ["labels"],
    )
