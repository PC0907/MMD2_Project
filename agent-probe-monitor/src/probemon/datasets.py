"""Activation shard I/O and dataset assembly with generalization splits.

On-disk layout (one shard per episode):

    data/activations/
      meta.parquet                 one row per (episode, step) with labels
                                   and provenance (suite, attack, model)
      shards/{episode_id}.pt       dict: "{step}/{segment}_{layer}" -> tensor

Splits:
    loao   leave-one-attack-out: train on all attacks but one, test on the
           held-out attack (benign episodes appear in both, split by episode)
    loso   leave-one-suite-out: same, over task suites
    random episode-level random split (sanity ceiling, not a headline number)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def save_episode(out_dir: Path, episode_id: str, act_store: dict[str, torch.Tensor],
                 meta_rows: list[dict]) -> None:
    shards = out_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    torch.save(act_store, shards / f"{episode_id.replace('/', '_')}.pt")
    meta_path = out_dir / "meta.parquet"
    df_new = pd.DataFrame(meta_rows)
    if meta_path.exists():
        df = pd.concat([pd.read_parquet(meta_path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(meta_path, index=False)


def load_meta(out_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / "meta.parquet")


def load_features(out_dir: Path, meta: pd.DataFrame, segment: str, layer: int
                  ) -> np.ndarray:
    """Return (n_rows, d_model) features aligned to meta's row order."""
    shards = Path(out_dir) / "shards"
    feats: list[np.ndarray] = []
    cache: dict[str, dict] = {}
    for _, row in meta.iterrows():
        eid = row["episode_id"].replace("/", "_")
        if eid not in cache:
            cache.clear()  # keep memory bounded; meta is sorted by episode
            cache[eid] = torch.load(shards / f"{eid}.pt", map_location="cpu")
        key = f"{row['episode_id']}/{row['step']}/{segment}_{layer}"
        feats.append(cache[eid][key].float().numpy())
    return np.stack(feats)


def make_split(meta: pd.DataFrame, kind: str, holdout: str | None = None,
               val_fraction: float = 0.15, seed: int = 0
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) metadata frames.

    kind='loao': holdout is an attack name; test = that attack + a disjoint
    slice of benign episodes. kind='loso': holdout is a suite name.
    """
    rng = np.random.default_rng(seed)
    if kind == "random":
        eids = np.asarray(meta["episode_id"].unique(), dtype=object)
        rng.shuffle(eids)
        n_test = max(1, int(0.2 * len(eids)))
        test_ids = set(eids[:n_test])
        rest = eids[n_test:]
    elif kind in ("loao", "loso"):
        col = "attack" if kind == "loao" else "suite"
        assert holdout is not None, f"{kind} split needs a holdout {col}"
        test_mask = meta[col] == holdout
        # Give the test side its own benign episodes for calibrated FPRs.
        benign = np.asarray(
            meta[meta["attack"] == "benign"]["episode_id"].unique(), dtype=object
        )
        rng.shuffle(benign)
        benign_test = set(benign[: max(1, len(benign) // 5)])
        test_ids = set(meta[test_mask]["episode_id"]) | benign_test
        rest = np.array([e for e in meta["episode_id"].unique() if e not in test_ids])
    else:
        raise ValueError(f"Unknown split kind: {kind}")

    rng.shuffle(rest)
    n_val = max(1, int(val_fraction * len(rest)))
    val_ids = set(rest[:n_val])
    train_ids = set(rest[n_val:])

    by = meta["episode_id"]
    return (
        meta[by.isin(train_ids)].reset_index(drop=True),
        meta[by.isin(val_ids)].reset_index(drop=True),
        meta[by.isin(test_ids)].reset_index(drop=True),
    )
