"""Tests for the benchmark-independent core (probes, splits, metrics).

Run: pytest tests/ -q
These intentionally avoid AgentDojo and any model download.
"""

import numpy as np
import pandas as pd
import pytest

from probemon.probes import build_probe
from probemon.datasets import make_split
from probemon.evaluate import (
    auroc, latency_to_detection, threshold_at_fpr,
)


def _toy_data(n=400, d=32, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d)) + 1.5 * np.outer(y, w)
    return X, y


@pytest.mark.parametrize("kind", ["logistic", "massmean", "torch_linear"])
def test_probe_learns_separable_signal(kind):
    X, y = _toy_data()
    probe = build_probe(kind)
    probe.fit(X, y)
    assert auroc(y, probe.score(X)) > 0.9


def _toy_meta():
    rows = []
    for attack in ["benign", "a1", "a2"]:
        for ep in range(6):
            eid = f"m/{attack}/{ep}"
            for step in range(4):
                rows.append({
                    "episode_id": eid, "step": step,
                    "suite": "s1" if ep % 2 else "s2",
                    "attack": attack,
                    "injection_present": int(attack != "benign" and step >= 1),
                    "attack_furthering": int(attack != "benign" and step >= 2),
                    "security_violated": attack != "benign",
                })
    return pd.DataFrame(rows)


def test_loao_split_holds_out_attack_episodes():
    meta = _toy_meta()
    train, val, test = make_split(meta, "loao", holdout="a1", seed=0)
    assert (train["attack"] != "a1").all()
    assert (val["attack"] != "a1").all()
    assert (test["attack"] == "a1").any()
    assert (test["attack"] == "benign").any(), "test needs benign negatives"
    # No episode leaks across splits.
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not (set(a["episode_id"]) & set(b["episode_id"]))


def test_latency_to_detection_pre_execution():
    meta = _toy_meta()
    test = meta[meta["attack"] == "a1"].reset_index(drop=True)
    # Score fires exactly at the first injected step (step 1), one step
    # before the attack executes (step 2) -> caught_before_execution.
    scores = (test["step"] >= 1).astype(float).to_numpy()
    lat = latency_to_detection(test, scores, threshold=0.5)
    assert len(lat) == test["episode_id"].nunique()
    assert lat["caught_before_execution"].all()
    assert (lat["latency"] == -1).all()


def test_threshold_at_fpr_monotone():
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    t1 = threshold_at_fpr(y, s, 0.01)
    t5 = threshold_at_fpr(y, s, 0.05)
    assert t1 >= t5
