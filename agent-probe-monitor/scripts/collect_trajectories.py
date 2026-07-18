#!/usr/bin/env python
"""Collect agent trajectories with activations over AgentDojo.

Iterates suites x attacks x (user task, injection task) pairs, runs each
episode with the hooked local model, labels steps, and writes shards.

Usage:
    python scripts/collect_trajectories.py --config configs/default.yaml \
        [--suite workspace] [--attack important_instructions] [--limit 10]
"""

from __future__ import annotations

import argparse
import itertools
import traceback

from pathlib import Path

from probemon.agentdojo_env import (
    LocalHFAgent, load_suite_and_attack, run_episode,
)
from probemon.labeling import (
    extract_target_tools, label_attack_furthering, label_injection_present,
)
from probemon.datasets import save_episode
from probemon.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default=None, help="restrict to one suite")
    ap.add_argument("--attack", default=None, help="restrict to one attack")
    ap.add_argument("--limit", type=int, default=None,
                    help="max episodes per (suite, attack)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["collection"]["seed"])
    out_dir = Path(cfg["collection"]["out_dir"])

    agent = LocalHFAgent(
        model_name=cfg["model"]["name"],
        layers=cfg["model"]["layers"],
        dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"],
        pooling=cfg["model"]["pooling"],
    )

    suites = [args.suite] if args.suite else cfg["collection"]["suites"]
    attacks = [args.attack] if args.attack else cfg["collection"]["attacks"]
    if cfg["collection"]["include_benign"] and "benign" not in attacks:
        attacks = ["benign"] + list(attacks)

    for suite_name, attack_name in itertools.product(suites, attacks):
        suite, attack = load_suite_and_attack(suite_name, attack_name)
        user_tasks = list(suite.user_tasks.values())      # VERIFY attr
        inj_tasks = (
            [None] if attack_name == "benign"
            else list(suite.injection_tasks.values())     # VERIFY attr
        )
        pairs = list(itertools.product(user_tasks, inj_tasks))
        if args.limit:
            pairs = pairs[: args.limit]

        for i, (ut, it) in enumerate(pairs):
            eid = f"{cfg['model']['name'].split('/')[-1]}/{suite_name}/{attack_name}/{i}"
            try:
                records, act_store, outcome = run_episode(
                    agent, suite, ut, it, attack,
                    max_steps=cfg["collection"]["max_steps"], episode_id=eid,
                )
                label_injection_present(records, outcome["injections"])
                label_attack_furthering(
                    records, outcome["security_violated"],
                    extract_target_tools(it) if it else None,
                )
                meta_rows = []
                for rec in records:
                    row = rec.to_meta()
                    row.update({
                        "suite": suite_name,
                        "attack": attack_name,
                        "model": cfg["model"]["name"],
                        "utility": outcome["utility"],
                        "security_violated": outcome["security_violated"],
                    })
                    meta_rows.append(row)
                save_episode(out_dir, eid, act_store, meta_rows)
                print(f"[ok] {eid} steps={len(records)} "
                      f"sec_violated={outcome['security_violated']}")
            except Exception:  # noqa: BLE001 — keep the sweep alive
                print(f"[fail] {eid}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
