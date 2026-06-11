"""Stage 2: relate the asked question to each reconstructed QUD.

Two complementary signal sources:

  (a) LLM relation judgment (primary): structured relation label +
      overlap score per (asked, addressed) pair.
  (b) Cheap symmetric features (for the learned classifier and for
      analysis): sentence-embedding cosine similarity and bidirectional
      NLI scores between the questions.

The per-example output aggregates over the (<= max_quds) addressed
questions by keeping the best-overlap relation, and carries forward all
pairwise features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # headless fallback
    def tqdm(it, **kwargs):
        return it

from ..data.taxonomy import QUDRelation
from .llm_client import LLMClient, parse_json
from .prompts import RELATE_SYSTEM, RELATE_USER

logger = logging.getLogger(__name__)

# Ordering used to pick the "dominant" relation when several QUDs are
# addressed: a question that is fully answered dominates partial coverage.
RELATION_PRIORITY = [
    QUDRelation.EQUIVALENT,
    QUDRelation.SPECIFICATION,
    QUDRelation.GENERALIZATION,
    QUDRelation.TOPIC_SHIFT,
    QUDRelation.UNRELATED,
]


def _valid_relation(value: str) -> str:
    try:
        return QUDRelation(value).value
    except ValueError:
        return QUDRelation.UNRELATED.value


def llm_relations(
    df: pd.DataFrame,
    recon: pd.DataFrame,
    client: LLMClient,
    out_path: str | Path,
    question_col: str = "question",
    batch_size: int = 256,
) -> pd.DataFrame:
    """LLM relation judgment for every (asked, addressed) pair."""
    merged = df[["example_id", question_col]].merge(recon, on="example_id")

    pairs: list[tuple[str, str, str]] = []  # (example_id, asked, addressed)
    for _, row in merged.iterrows():
        for q in row["addressed_questions"]:
            pairs.append((row["example_id"], row[question_col], q))
    logger.info("Stage 2: %d (asked, addressed) pairs", len(pairs))

    results = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="relate"):
        chunk = pairs[start:start + batch_size]
        users = [RELATE_USER.format(asked=a, addressed=b) for _, a, b in chunk]
        raw = client.chat_batch(RELATE_SYSTEM, users)
        for (ex_id, asked, addressed), text in zip(chunk, raw):
            rec = parse_json(text, default={})
            results.append({
                "example_id": ex_id,
                "asked": asked,
                "addressed": addressed,
                "relation": _valid_relation(rec.get("relation", "unrelated")),
                "overlap": float(rec.get("overlap", 0.0) or 0.0),
                "rationale": rec.get("rationale", ""),
            })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_json(out_path, orient="records", lines=True)
    logger.info("Stage 2 wrote %d pair relations to %s", len(results), out_path)
    return pd.DataFrame(results)


def embedding_features(
    pair_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cuda",
) -> pd.DataFrame:
    """Cosine similarity between asked and addressed questions."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    asked_emb = model.encode(pair_df["asked"].tolist(), normalize_embeddings=True,
                             show_progress_bar=True, batch_size=128)
    addr_emb = model.encode(pair_df["addressed"].tolist(), normalize_embeddings=True,
                            show_progress_bar=True, batch_size=128)
    pair_df = pair_df.copy()
    pair_df["emb_cosine"] = np.sum(asked_emb * addr_emb, axis=1)
    return pair_df


def nli_features(
    pair_df: pd.DataFrame,
    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    device: str = "cuda",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Bidirectional NLI entailment probabilities between declarative
    paraphrases of the two questions. Asymmetry between the directions is
    a specificity cue: if asked => addressed but not vice versa, the
    addressed question is more general."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    ent_idx = _entailment_index(model)

    def _probs(premises, hypotheses):
        out = []
        for s in tqdm(range(0, len(premises), batch_size), desc="nli", leave=False):
            batch = tok(premises[s:s + batch_size], hypotheses[s:s + batch_size],
                        truncation=True, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**batch).logits
            out.append(torch.softmax(logits, dim=-1)[:, ent_idx].cpu().numpy())
        return np.concatenate(out)

    a = pair_df["asked"].tolist()
    b = pair_df["addressed"].tolist()
    pair_df = pair_df.copy()
    pair_df["nli_asked_to_addr"] = _probs(a, b)
    pair_df["nli_addr_to_asked"] = _probs(b, a)
    pair_df["nli_asymmetry"] = pair_df["nli_asked_to_addr"] - pair_df["nli_addr_to_asked"]
    return pair_df


def _entailment_index(model) -> int:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    for i, name in id2label.items():
        if "entail" in name:
            return i
    return 0


def aggregate_per_example(pair_df: pd.DataFrame, recon: pd.DataFrame) -> pd.DataFrame:
    """Collapse pairwise relations to one record per example:
    dominant relation, best overlap, mean/max similarity features, plus
    the Stage-1 speech act."""
    prio = {r.value: i for i, r in enumerate(RELATION_PRIORITY)}

    rows = []
    for ex_id, grp in pair_df.groupby("example_id"):
        grp = grp.sort_values(by="relation", key=lambda s: s.map(prio))
        best = grp.iloc[0]
        rows.append({
            "example_id": ex_id,
            "dominant_relation": best["relation"],
            "qud_overlap": float(grp["overlap"].max()),
            "mean_overlap": float(grp["overlap"].mean()),
            "n_addressed": len(grp),
            "emb_cosine_max": float(grp.get("emb_cosine", pd.Series(np.nan)).max()),
            "nli_asym_best": float(best.get("nli_asymmetry", np.nan)),
        })
    agg = pd.DataFrame(rows)

    # Examples whose Stage-1 output had no addressed questions never enter
    # pair_df; re-attach them with relation NONE.
    agg = recon[["example_id", "speech_act"]].merge(agg, on="example_id", how="left")
    agg["dominant_relation"] = agg["dominant_relation"].fillna(QUDRelation.NONE.value)
    agg["qud_overlap"] = agg["qud_overlap"].fillna(0.0)
    agg["n_addressed"] = agg["n_addressed"].fillna(0).astype(int)
    return agg
