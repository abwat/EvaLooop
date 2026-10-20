#!/usr/bin/env python3
"""Measure deterministic noise and cycle-1 agreement on the 25-task gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaloop.evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset
from evaloop.extensions.spanish_loop import GENERATION_PROMPT, extract_code
from run_assignment4 import TransformersBackend, test_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("assignment4_config.json"))
    parser.add_argument("--task-order", type=Path, default=Path("protocol/task_order.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen2.5-coder-7b-instruct")
    parser.add_argument("--output", type=Path, default=Path("protocol/cycle1_gate.json"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    model = next(m for m in cfg["models"] if m["model_id"] == args.model_id)
    task_ids = json.loads(args.task_order.read_text())["smoke_ids"]
    supplied = json.loads((args.data_dir / "loops" / model["baseline_file"]).read_text())
    historical = {t["task_id"]: bool(t["cycles"][0]["passed"]) for t in supplied["tasks"]}
    problems, ground_truth = load_mbpp_dataset()
    tester = EvalPlusCodeTester(problems, ground_truth)
    backend = TransformersBackend(model, args.max_new_tokens)
    rows = []
    try:
        for index, task_id in enumerate(task_ids, 1):
            prompt = GENERATION_PROMPT.format(description=problems[task_id]["prompt"])
            repetitions = []
            for _ in range(3):
                generation = backend.generate(prompt)
                code = extract_code(generation["text"])
                tested = test_code(tester, problems, task_id, code)
                repetitions.append(
                    {
                        "passed": tested["passed"],
                        "code": code,
                        "hit_token_cap": generation["hit_token_cap"],
                    }
                )
            rows.append(
                {
                    "task_id": task_id,
                    "historical_passed": historical[task_id],
                    "repetitions": repetitions,
                    "pass_fail_disagreement": len({r["passed"] for r in repetitions}) > 1,
                    "first_vs_historical_disagreement": repetitions[0]["passed"] != historical[task_id],
                }
            )
            print(f"[{index}/25] {task_id}", flush=True)
    finally:
        backend.close()
    repeat_disagreements = sum(r["pass_fail_disagreement"] for r in rows)
    historical_disagreements = sum(r["first_vs_historical_disagreement"] for r in rows)
    result = {
        "model_id": args.model_id,
        "n_tasks": len(rows),
        "repetitions": 3,
        "batch_size": 1,
        "repeat_pass_fail_disagreements": repeat_disagreements,
        "halt_threshold": 2,
        "passes_noise_gate": repeat_disagreements <= 2,
        "first_vs_historical_disagreements": historical_disagreements,
        "historical_comparison_note": (
            "Historical chat template/system prompt/backend/precision are undocumented; this is "
            "a diagnostic, not a determinism test."
        ),
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, indent=2))
    if not result["passes_noise_gate"]:
        raise SystemExit("HALT: deterministic-noise threshold exceeded (more than 2/25 tasks)")


if __name__ == "__main__":
    main()
