#!/usr/bin/env python3
"""Freeze one seeded ordering of all 378 task IDs from the supplied records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("protocol/task_order.json"))
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    data = json.loads(args.baseline.read_text())
    ids = [str(task["task_id"]) for task in data["tasks"]]
    if len(ids) != 378 or len(set(ids)) != 378:
        raise SystemExit(f"Expected 378 unique task IDs, found {len(ids)}")
    random.Random(args.seed).shuffle(ids)
    payload = {
        "seed": args.seed,
        "source": args.baseline.name,
        "n_total": len(ids),
        "task_ids": ids,
        "smoke_ids": ids[:25],
        "principal_ids": ids[:100],
        "extension_blocks": [ids[i : i + 25] for i in range(100, 378, 25)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}: first 25 smoke tasks, first 100 principal tasks")


if __name__ == "__main__":
    main()
