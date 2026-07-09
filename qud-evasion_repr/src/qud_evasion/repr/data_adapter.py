"""Canonical example schema shared by every experimental arm.

Every arm of the paper (prompting, representation classifier, LoRA,
encoder) must consume *identical* inputs, otherwise arm comparisons are
confounded by preprocessing. This module defines that single schema and
adapts whatever loader you already have onto it.

Integration
-----------
Preferred path: point ``loader`` in the YAML config at your existing
function, e.g. ``qud_evasion.data.load:load_qevasion``. It may return
either a list of dicts or a HuggingFace ``Dataset``; we normalise column
names defensively via ``COLUMN_CANDIDATES``.

Fallback path: if no loader is configured we pull
``ailsntua/QEvasion`` from the HF hub directly.

If a required column cannot be found under any candidate name, we raise
with the observed columns so the fix is a one-line edit to
``COLUMN_CANDIDATES`` -- never a silent misalignment.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CLARITY_LABELS = ["Clear Reply", "Ambiguous", "Clear Non-Reply"]

EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]


@dataclass
class Example:
    """One (sub-question, answer) unit -- the dataset's unit of labelling."""

    id: str
    interview_id: str            # grouping key for leakage-free splits
    sub_question: str            # the isolated sub-question being scored
    full_question: str           # the interviewer's full turn (may be empty)
    answer: str                  # the full interview answer
    clarity_label: Optional[int] = None   # index into CLARITY_LABELS
    evasion_label: Optional[int] = None   # index into EVASION_LABELS
    position_in_turn: int = 0    # 0-based rank among sibling sub-questions
    turn_size: int = 1           # number of sibling sub-questions
    annotator_labels: list = field(default_factory=list)  # per-annotator, if any
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Candidate raw-column names, tried in order. Extend if your columns differ;
# the error message on failure prints what was actually found.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "sub_question": ["question", "sub_question", "subquestion", "isolated_question"],
    "full_question": ["interview_question", "full_question", "original_question"],
    "answer": ["interview_answer", "answer", "response"],
    "clarity_label": ["clarity_label", "clarity", "label", "level1", "level_1"],
    "evasion_label": ["evasion_label", "evasion", "strategy", "level2", "level_2"],
    "interview_id": ["interview_id", "interview", "url", "source", "doc_id", "title"],
}

# Surface variants observed in the raw annotation, normalised to canonical.
CLARITY_ALIASES = {
    "clear reply": "Clear Reply",
    "clear": "Clear Reply",
    "direct reply": "Clear Reply",
    "ambiguous": "Ambiguous",
    "ambivalent": "Ambiguous",
    "ambiguous reply": "Ambiguous",
    "clear non-reply": "Clear Non-Reply",
    "clear non reply": "Clear Non-Reply",
    "non-reply": "Clear Non-Reply",
}


def _resolve(row: dict, key: str, required: bool = True) -> Any:
    for cand in COLUMN_CANDIDATES[key]:
        if cand in row and row[cand] is not None:
            return row[cand]
    if required:
        raise KeyError(
            f"Could not resolve schema field '{key}' from columns "
            f"{sorted(row.keys())}. Add the right column name to "
            f"COLUMN_CANDIDATES['{key}'] in repr/data_adapter.py."
        )
    return None


def _norm_clarity(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    canonical = CLARITY_ALIASES.get(text)
    if canonical is None:
        # try prefix matching on the canonical labels themselves
        for lab in CLARITY_LABELS:
            if text == lab.lower():
                canonical = lab
                break
    if canonical is None:
        raise ValueError(
            f"Unrecognised clarity label {value!r}. Add it to CLARITY_ALIASES."
        )
    return CLARITY_LABELS.index(canonical)


def _norm_evasion(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    for i, lab in enumerate(EVASION_LABELS):
        if text == lab.lower() or text.split("/")[0] == lab.lower().split("/")[0]:
            return i
    raise ValueError(
        f"Unrecognised evasion label {value!r}. Extend _norm_evasion / "
        f"EVASION_LABELS in repr/data_adapter.py."
    )


def _load_via_dotted_path(path: str) -> Callable[..., Iterable[dict]]:
    """Resolve 'package.module:function' into a callable."""
    module_name, _, func_name = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def load_examples(
    split: str = "train",
    loader: Optional[str] = None,
    loader_kwargs: Optional[dict] = None,
) -> list[Example]:
    """Load one split as a list of canonical Examples.

    Parameters
    ----------
    split: "train" or "test" (the official HF splits).
    loader: optional dotted path to your own loader,
        e.g. "qud_evasion.data.load:load_qevasion". It receives
        ``split=<split>`` plus ``loader_kwargs`` and must return an
        iterable of dict-like rows.
    """
    if loader:
        rows = list(_load_via_dotted_path(loader)(split=split, **(loader_kwargs or {})))
    else:  # fallback: straight from the hub
        from datasets import load_dataset

        rows = list(load_dataset("ailsntua/QEvasion", split=split))

    examples: list[Example] = []
    for i, row in enumerate(rows):
        row = dict(row)
        ex = Example(
            id=str(row.get("id", f"{split}-{i}")),
            interview_id=str(_resolve(row, "interview_id")),
            sub_question=str(_resolve(row, "sub_question")).strip(),
            full_question=str(_resolve(row, "full_question", required=False) or "").strip(),
            answer=str(_resolve(row, "answer")).strip(),
            clarity_label=_norm_clarity(_resolve(row, "clarity_label", required=False)),
            evasion_label=_norm_evasion(_resolve(row, "evasion_label", required=False)),
            meta={k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))},
        )
        examples.append(ex)

    _annotate_turn_structure(examples)
    return examples


def _annotate_turn_structure(examples: list[Example]) -> None:
    """Fill position_in_turn / turn_size.

    Sibling sub-questions share the same (interview_id, answer). This is
    the allocation structure documented in the report; we recompute it
    here so the schema is self-contained.
    """
    groups: dict[tuple, list[Example]] = {}
    for ex in examples:
        groups.setdefault((ex.interview_id, ex.answer), []).append(ex)
    for members in groups.values():
        for rank, ex in enumerate(members):
            ex.position_in_turn = rank
            ex.turn_size = len(members)
