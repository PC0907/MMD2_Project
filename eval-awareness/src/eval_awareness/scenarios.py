"""Stage 1: expand seed scenario families into a paired eval/deploy dataset.

Each record:
    {
      "id": str,
      "family": str,             # scenario family (used for leave-one-out splits)
      "kind": "neutral"|"dilemma",
      "framing": "eval"|"deploy",
      "pair_id": str,            # eval/deploy versions of the same task share this
      "prompt": str,
      "safe_behavior": str|None, # rubric for dilemma families
      "unsafe_behavior": str|None,
    }
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from .utils import write_jsonl


def _fill(template: str, fillers: dict[str, list[str]], rng: random.Random) -> str:
    out = template
    for key, options in fillers.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(options), 1)
    return out


def generate_dataset(
    seed_path: str | Path,
    out_path: str | Path,
    n_variants_per_context: int = 4,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    spec = json.loads(Path(seed_path).read_text())
    fillers = spec.get("fillers", {})

    records: list[dict] = []
    for family in spec["families"]:
        contexts = {
            "eval": family["eval_contexts"],
            "deploy": family["deploy_contexts"],
        }
        pair_counter = itertools.count()
        # For each (eval_ctx, deploy_ctx) combination we emit a matched pair on
        # the *same* filled task text, so the only difference within a pair is
        # the framing context.
        for ei, di in itertools.product(
            range(len(contexts["eval"])), range(len(contexts["deploy"]))
        ):
            for _ in range(n_variants_per_context):
                task_text = _fill(family["task"], fillers, rng)
                pair_id = f"{family['name']}-{next(pair_counter):04d}"
                for framing, ctx_list, idx in (
                    ("eval", contexts["eval"], ei),
                    ("deploy", contexts["deploy"], di),
                ):
                    ctx = _fill(ctx_list[idx], fillers, rng)
                    records.append(
                        {
                            "id": f"{pair_id}-{framing}",
                            "family": family["name"],
                            "kind": family.get("kind", "neutral"),
                            "framing": framing,
                            "pair_id": pair_id,
                            "prompt": ctx + task_text,
                            "safe_behavior": family.get("safe_behavior"),
                            "unsafe_behavior": family.get("unsafe_behavior"),
                        }
                    )

    rng.shuffle(records)
    write_jsonl(out_path, records)
    return records
