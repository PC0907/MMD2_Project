"""
Load and prepare the QEvasion dataset.

Key facts (verified from the dataset card / viewer):
  - Official splits: train (3448) and test (308) ONLY. No official dev split.
    -> We carve a stratified internal dev split out of train; test is the
       untouched final-evaluation set, read exactly once by the eval pipelines.
  - The classification unit is the extracted sub-question (`question`), not the
    raw interviewer turn. Model input = question + interview_answer.
  - Heavy class imbalance (Ambivalent dominates; Clear Non-Reply is rare).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .taxonomy import (
    CLARITY_LABELS, CLARITY2ID, EVASION2ID,
)

HF_DATASET = "ailsntua/QEvasion"

_CLARITY_ALIASES = {
    "clear reply": "Clear Reply",
    "ambivalent": "Ambivalent",
    "ambivalent reply": "Ambivalent",
    "ambiguous": "Ambivalent",          # task-description name -> dataset name
    "clear non-reply": "Clear Non-Reply",
    "clear nonreply": "Clear Non-Reply",
    "clear non reply": "Clear Non-Reply",
}


@dataclass
class Splits:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "dev": len(self.dev), "test": len(self.test)}


def _norm_clarity(x) -> str | None:
    if not isinstance(x, str):
        return None
    return _CLARITY_ALIASES.get(x.strip().lower(), x.strip())


def _clean_text(x) -> str:
    if not isinstance(x, str):
        return ""
    x = x.replace("\u2014", "-").replace("\u2013", "-")
    x = re.sub(r"\[(inaudible|laughter|applause)\]", " ", x, flags=re.I)
    return re.sub(r"\s+", " ", x).strip()


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official (train, test) splits as DataFrames."""
    ds = load_dataset(HF_DATASET)
    return ds["train"].to_pandas(), ds["test"].to_pandas()


def prepare(df: pd.DataFrame, drop_inaudible: bool = False,
            sep: str = " [SEP] ") -> pd.DataFrame:
    """Clean text, normalise labels, build model input, map labels to ids."""
    df = df.copy()
    df["clarity_label"] = df["clarity_label"].map(_norm_clarity)
    df = df.dropna(subset=["question", "interview_answer", "clarity_label"])
    df = df[df["clarity_label"].isin(CLARITY_LABELS)]

    if drop_inaudible and "inaudible" in df.columns:
        df = df[df["inaudible"] != True]  # noqa: E712

    df["q_clean"] = df["question"].map(_clean_text)
    df["a_clean"] = df["interview_answer"].map(_clean_text)
    df = df[(df["q_clean"].str.len() > 0) & (df["a_clean"].str.len() > 0)]

    # Question placed first so it is never truncated away on long answers.
    df["text"] = "Question: " + df["q_clean"] + sep + "Answer: " + df["a_clean"]
    df["clarity_id"] = df["clarity_label"].map(CLARITY2ID).astype(int)
    if "evasion_label" in df.columns:
        df["evasion_id"] = df["evasion_label"].map(EVASION2ID)
    return df.reset_index(drop=True)


def make_splits(dev_size: float = 0.15, seed: int = 42,
                drop_inaudible: bool = False, sep: str = " [SEP] ") -> Splits:
    """train / stratified internal-dev / untouched test."""
    raw_train, raw_test = load_raw()
    train_full = prepare(raw_train, drop_inaudible, sep)
    test = prepare(raw_test, drop_inaudible, sep)
    tr, dev = train_test_split(
        train_full, test_size=dev_size, random_state=seed,
        stratify=train_full["clarity_id"])
    return Splits(tr.reset_index(drop=True), dev.reset_index(drop=True),
                  test.reset_index(drop=True))


def class_weights(df: pd.DataFrame, label_col: str, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1.0, for weighted CE loss."""
    labels = df[label_col].dropna().astype(int).to_numpy()
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)
