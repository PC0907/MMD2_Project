"""Metric computation and human-readable reports."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted"))


def per_class_f1(y_true, y_pred, labels) -> dict[int, float]:
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {int(l): float(s) for l, s in zip(labels, scores)}


def full_report(y_true, y_pred, target_names, digits: int = 4) -> str:
    present = sorted(set(np.asarray(y_true).astype(int)) | set(np.asarray(y_pred).astype(int)))
    names = [target_names[i] for i in present]
    return classification_report(y_true, y_pred, labels=present,
                                 target_names=names, digits=digits, zero_division=0)


def confusion_df(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return pd.DataFrame(cm, index=labels, columns=labels)
