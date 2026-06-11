"""
Tests for the text-cleaning and label-normalisation logic in data.dataset.
These run without network (they call the private helpers directly on synthetic rows).
"""
import pandas as pd

from clarity_evasion.data.dataset import _norm_clarity, _clean_text, prepare


def test_clarity_alias_ambiguous_maps_to_ambivalent():
    assert _norm_clarity("Ambiguous") == "Ambivalent"
    assert _norm_clarity("ambivalent reply") == "Ambivalent"
    assert _norm_clarity("Clear Non-Reply") == "Clear Non-Reply"


def test_clean_text_strips_inaudible_and_dashes():
    out = _clean_text("Well, the [inaudible] situation\u2014is calm.")
    assert "[inaudible]" not in out.lower()
    assert "\u2014" not in out


def test_prepare_builds_input_and_ids():
    rows = pd.DataFrame([
        {"question": "Will you commit?", "interview_answer": "We support families.",
         "clarity_label": "Clear Non-Reply", "evasion_label": "Dodging", "inaudible": False},
        {"question": "Alias test", "interview_answer": "ans",
         "clarity_label": "Ambiguous", "evasion_label": "General", "inaudible": False},
    ])
    out = prepare(rows)
    assert len(out) == 2
    assert out.iloc[1]["clarity_label"] == "Ambivalent"   # alias normalised
    assert out.iloc[0]["text"].startswith("Question:")
    assert "Answer:" in out.iloc[0]["text"]
    assert set(out["clarity_id"]).issubset({0, 1, 2})


def test_prepare_drops_missing_and_empty():
    rows = pd.DataFrame([
        {"question": None, "interview_answer": "x", "clarity_label": "Ambivalent"},
        {"question": "q", "interview_answer": "", "clarity_label": "Ambivalent"},
        {"question": "q", "interview_answer": "a", "clarity_label": "Ambivalent"},
    ])
    out = prepare(rows)
    assert len(out) == 1
