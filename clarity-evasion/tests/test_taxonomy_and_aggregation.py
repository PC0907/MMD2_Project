"""
Unit tests for the parts that don't need a GPU or network:
the taxonomy mapping and the clarity-derivation aggregation logic.

    pytest -q tests/
"""
import numpy as np
import pytest

from clarity_evasion.data import (
    taxonomy, CLARITY_LABELS, EVASION_LABELS, EVASION2ID, CLARITY2ID,
    EVASION_TO_CLARITY, CLARITY_TO_EVASIONS,
)
from clarity_evasion.models import derive_clarity


def test_taxonomy_is_consistent():
    assert taxonomy.validate()
    assert len(CLARITY_LABELS) == 3
    assert len(EVASION_LABELS) == 9
    # every evasion label maps to a valid clarity label
    assert set(EVASION_TO_CLARITY.values()) == set(CLARITY_LABELS)


def test_branch_partition_is_complete():
    total = sum(len(v) for v in CLARITY_TO_EVASIONS.values())
    assert total == len(EVASION_LABELS)
    assert CLARITY_TO_EVASIONS["Clear Reply"] == ["Explicit"]


def test_known_pairs_map_correctly():
    # pairs verified from the dataset viewer
    known = {
        "Explicit": "Clear Reply",
        "Dodging": "Ambivalent",
        "Deflection": "Ambivalent",
        "General": "Ambivalent",
        "Implicit": "Ambivalent",
        "Partial/half-answer": "Ambivalent",
        "Declining to answer": "Clear Non-Reply",
        "Claims ignorance": "Clear Non-Reply",
        "Clarification": "Clear Non-Reply",
    }
    for ev, cl in known.items():
        assert EVASION_TO_CLARITY[ev] == cl


def test_argmax_aggregation_confident_case():
    logits = np.full((1, 9), -5.0)
    logits[0, EVASION2ID["Explicit"]] = 10.0
    pred = derive_clarity(logits, agg="argmax")
    assert CLARITY_LABELS[pred[0]] == "Clear Reply"


def test_marginal_rescues_within_branch_uncertainty():
    """
    Mass spread across three Ambivalent sub-labels, with a single Clear-Non-Reply
    label edging them out on top-1. argmax leaks across the boundary; marginal
    correctly aggregates to Ambivalent.
    """
    logits = np.full((1, 9), -10.0)
    for lbl in ["Dodging", "Deflection", "General"]:
        logits[0, EVASION2ID[lbl]] = 1.0
    logits[0, EVASION2ID["Declining to answer"]] = 1.05

    argmax_pred = CLARITY_LABELS[derive_clarity(logits, agg="argmax")[0]]
    marginal_pred = CLARITY_LABELS[derive_clarity(logits, agg="marginal")[0]]
    assert argmax_pred == "Clear Non-Reply"
    assert marginal_pred == "Ambivalent"


def test_aggregation_batch_shape():
    logits = np.random.randn(17, 9)
    for agg in ("argmax", "marginal"):
        out = derive_clarity(logits, agg=agg)
        assert out.shape == (17,)
        assert out.min() >= 0 and out.max() < len(CLARITY_LABELS)


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        derive_clarity(np.zeros((1, 9)), agg="nonsense")
