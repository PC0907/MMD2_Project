"""
Shared model components: a class-weighted Trainer, the joint multitask model,
and tokenization helpers. Factored here so the three pipelines do not duplicate them.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModel, AutoModelForSequenceClassification, Trainer,
)


class WeightedTrainer(Trainer):
    """HF Trainer with class-weighted cross-entropy for imbalanced labels."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self._cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self._cw.to(outputs.logits.device) if self._cw is not None else None
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


class JointClarityEvasionModel(nn.Module):
    """
    Shared encoder with two linear heads (evasion + clarity) for multitask
    'joint learning'. Primary task is evasion; clarity is an auxiliary signal that
    regularises the shared representation. Loss = evasion_CE + aux_weight * clarity_CE.
    Mean-pools token states (robust across encoders without a pooler output).
    """

    def __init__(self, backbone: str, n_evasion: int, n_clarity: int,
                 aux_weight: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.evasion_head = nn.Linear(h, n_evasion)
        self.clarity_head = nn.Linear(h, n_clarity)
        self.aux_weight = aux_weight

    def forward(self, input_ids=None, attention_mask=None,
                labels=None, clarity_labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        ev_logits = self.evasion_head(pooled)
        cl_logits = self.clarity_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(ev_logits, labels)
            if clarity_labels is not None:
                loss = loss + self.aux_weight * nn.functional.cross_entropy(
                    cl_logits, clarity_labels)
        return {"loss": loss, "logits": ev_logits, "clarity_logits": cl_logits}


def build_classifier(model_name: str, num_labels: int,
                     id2label: dict, label2id: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)


def tokenize_df(df, tokenizer, max_len: int, label_col: str,
                extra_label_cols: list[str] | None = None) -> Dataset:
    """DataFrame -> tokenized HF Dataset with 'labels' (+ optional extra labels)."""
    cols = ["text", label_col] + (extra_label_cols or [])
    sub = df.dropna(subset=[label_col])[cols].copy()
    sub[label_col] = sub[label_col].astype(int)
    rename = {label_col: "labels"}
    for c in (extra_label_cols or []):
        sub[c] = sub[c].astype(int)
    sub = sub.rename(columns=rename)
    ds = Dataset.from_pandas(sub, preserve_index=False)
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
        batched=True, remove_columns=["text"])
