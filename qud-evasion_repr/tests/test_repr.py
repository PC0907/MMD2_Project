"""Tests for the repr module. No GPU, no model download.

Covers the pieces where a silent bug would invalidate the paper:
answer-span masking, pooling math, interview-level splitting, control
labels, and the cache round trip.
"""

import numpy as np
import torch

from qud_evasion.repr.data_adapter import Example, _annotate_turn_structure
from qud_evasion.repr.pooling import answer_token_mask, pool_layer
from qud_evasion.repr.controls import control_labels
from qud_evasion.eval.significance import bootstrap_ci, paired_bootstrap


def test_answer_mask_overlap_and_fallback():
    # 4 tokens with char spans; answer covers chars 10-20
    offsets = torch.tensor([[[0, 5], [5, 10], [10, 15], [15, 20]]])
    attn = torch.ones(1, 4, dtype=torch.long)
    mask = answer_token_mask(offsets, attn, [(10, 20)])
    assert mask.tolist() == [[False, False, True, True]]

    # answer entirely truncated away -> falls back to last real token
    mask = answer_token_mask(offsets, attn, [(100, 200)])
    assert mask.tolist() == [[False, False, False, True]]


def test_pooling_math():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [0.0, 0.0]]])
    attn = torch.tensor([[1, 1, 1, 0]])          # 3 real tokens, 1 pad
    ans = torch.tensor([[False, True, True, False]])
    pooled = pool_layer(hidden, attn, ans)
    assert pooled["last"][0, 0].item() == 5.0     # last non-pad token
    assert abs(pooled["mean_all"][0, 0].item() - 3.0) < 1e-6
    assert abs(pooled["mean_answer"][0, 0].item() - 4.0) < 1e-6


def test_turn_structure_annotation():
    exs = [
        Example(id="a", interview_id="i1", sub_question="q1", full_question="", answer="A"),
        Example(id="b", interview_id="i1", sub_question="q2", full_question="", answer="A"),
        Example(id="c", interview_id="i1", sub_question="q3", full_question="", answer="B"),
    ]
    _annotate_turn_structure(exs)
    assert (exs[0].turn_size, exs[1].turn_size, exs[2].turn_size) == (2, 2, 1)
    assert (exs[0].position_in_turn, exs[1].position_in_turn) == (0, 1)


def test_control_labels_deterministic_and_marginal():
    y = np.array([0] * 50 + [1] * 30 + [2] * 20)
    ids = [f"ex{i}" for i in range(100)]
    c1 = control_labels(ids, y, seed=0)
    c2 = control_labels(ids, y, seed=0)
    assert np.array_equal(c1, c2)                       # deterministic
    assert set(np.unique(c1)) <= {0, 1, 2}
    assert abs((c1 == 0).mean() - 0.5) < 0.2            # roughly marginal-preserving
    assert not np.array_equal(c1, y)                    # not the real task


def test_bootstrap_grouped():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    good = y.copy(); good[rng.random(300) < 0.2] = rng.integers(0, 3, (rng.random(300) < 0.2).sum())
    bad = rng.integers(0, 3, 300)
    groups = np.repeat(np.arange(30), 10)   # 30 "interviews" of 10 siblings

    ci = bootstrap_ci(y, good, groups=groups, n_boot=200)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]

    test = paired_bootstrap(y, good, bad, groups=groups, n_boot=200)
    assert test["delta"] > 0 and test["p_value"] < 0.05
