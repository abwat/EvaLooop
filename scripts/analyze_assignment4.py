#!/usr/bin/env python3
"""Build derived baselines, summary.csv, statistics, and failure sample."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Mapping

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.extensions.assignment4_metrics import (
    cycles_by_task,
    paired_asl_difference,
    paired_bootstrap_ci,
    summarize_cycles,
)


def load(path: Path) -> Dict:
    return json.loads(path.read_text())


def save(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def derive_baseline(source: Path, task_ids: list[str]) -> Dict:
    original = load(source)
    by_id = {t["task_id"]: t for t in original["tasks"]}
    missing = sorted(set(task_ids) - set(by_id))
    if missing:
        raise ValueError(f"Baseline {source} is missing {missing}")
    result = {
        "schema_version": "assignment4-derived-baseline-v1",
        "model_id": original["model_id"],
        "name": original.get("name"),
        "configuration": "standard-supplied-derived",
        "provenance": f"assignment2-data/loops/{source.name}",
        "note": "Derived from the supplied records on the selected 100 task IDs; no new inference.",
        "task_ids": task_ids,
        "max_cycles": 10,
        "tasks": [by_id[i] for i in task_ids],
    }
    return result


def assert_complete(result: Mapping, task_ids: list[str]) -> None:
    found = [t["task_id"] for t in result["tasks"]]
    if len(found) != len(task_ids) or set(found) != set(task_ids):
        raise ValueError(
            f"Incomplete {result.get('model_id')} {result.get('configuration')}: "
            f"{len(found)}/{len(task_ids)} tasks"
        )


def sample_failures(
    results: Mapping[tuple[str, str], Dict], output: Path, seed: int, sample_size: int = 20
) -> None:
    """Stratify Spanish post-cycle-1 failures across models and failure cycles."""
    candidates = []
    for (model, arm), result in results.items():
        if arm != "spanish":
            continue
        control = {t["task_id"]: t for t in results[(model, "english-control")]["tasks"]}
        baseline = {t["task_id"]: t for t in results[(model, "standard-supplied-derived")]["tasks"]}
        for task in result["tasks"]:
            s = int(task["successful_cycles"])
            if 1 <= s < 10:  # At least one Spanish summary preceded the failure.
                fail_cycle = s + 1
                failed = task["cycles"][-1]
                last_summary = task["cycles"][-2].get("summary_raw", "") if s > 1 else task["cycles"][0].get("summary_raw", "")
                candidates.append(
                    {
                        "model": model,
                        "task_id": task["task_id"],
                        "failed_cycle": fail_cycle,
                        "spanish_successful_cycles": s,
                        "english_control_successful_cycles": control[task["task_id"]]["successful_cycles"],
                        "standard_successful_cycles": baseline[task["task_id"]]["successful_cycles"],
                        "spanish_compliant_last_summary": task["cycles"][s - 1]
                        .get("language_check", {})
                        .get("spanish_compliant", ""),
                        "last_summary_excerpt": last_summary.replace("\n", " ")[:240],
                        "failed_code_excerpt": failed.get("code", "").replace("\n", " ")[:240],
                        "failure_mode": "TO_CODE_MANUALLY",
                        "notes": "",
                    }
                )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Cover distinct model/cycle strata, then fill from the seeded order.
    selected = []
    used = set()
    covered_strata = set()
    target = min(sample_size, len(candidates))
    for row in candidates:
        key = (row["model"], row["failed_cycle"])
        if key not in covered_strata:
            selected.append(row)
            used.add(id(row))
            covered_strata.add(key)
            if len(selected) == target:
                break
    if len(selected) < target:
        for row in candidates:
            if id(row) not in used:
                selected.append(row)
                used.add(id(row))
                if len(selected) == target:
                    break
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0]) if selected else [
        "model", "task_id", "failed_cycle", "failure_mode", "notes"
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("assignment4_config.json"))
    parser.add_argument("--task-order", type=Path, default=Path("protocol/task_order.json"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild only six performed-run rows; no course data or SciPy needed")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_assignment4"))
    args = parser.parse_args()
    summary_only = args.summary_only or args.data_dir is None

    config = load(args.config)
    task_ids = load(args.task_order)["principal_ids"]
    results: Dict[tuple[str, str], Dict] = {}
    for model in config["models"]:
        model_id = model["model_id"]
        if not summary_only:
            baseline = derive_baseline(
                args.data_dir / "loops" / model["baseline_file"], task_ids
            )
            results[(model_id, "standard-supplied-derived")] = baseline
        for arm in config["principal_arms"]:
            result = load(args.results_dir / f"{model_id}__{arm}.json")
            assert_complete(result, task_ids)
            results[(model_id, arm)] = result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (model_id, arm), result in results.items():
        values = cycles_by_task(result)
        metrics = summarize_cycles(list(values.values()))
        rows.append(
            {
                "model": model_id,
                "configuration": arm,
                "n_tasks": metrics["n_tasks"],
                "success_rate": f"{metrics['success_rate']:.6f}",
                "asl_score": f"{metrics['asl_score']:.6f}",
            }
        )
    rows.sort(key=lambda row: (row["model"], row["configuration"]))
    summary_path = args.output_dir / "summary.csv"
    for filename, historical in [("summary.csv", False), ("baseline_summary.csv", True)]:
        if summary_only and historical:
            continue
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["model", "configuration", "n_tasks", "success_rate", "asl_score"]
            )
            writer.writeheader()
            writer.writerows(row for row in rows if (row["configuration"] == "standard-supplied-derived") == historical)

    if summary_only:
        print(f"Wrote {summary_path}: six performed runs, using only saved result JSONs.")
        return

    save(args.output_dir / "baseline_record_references.json", {
        "source": "course-provided assignment2-data/loops",
        "records": [
            {"model_id": model, "task_id": task["task_id"],
             "cycles": [cycle["cycle"] for cycle in task["cycles"]]}
            for (model, arm), result in results.items()
            if arm == "standard-supplied-derived"
            for task in result["tasks"]
        ],
    })

    comparisons = {}
    pooled_task_effects: Dict[str, list[float]] = {task_id: [] for task_id in task_ids}
    from scipy.stats import wilcoxon

    for model in config["models"]:
        model_id = model["model_id"]
        spanish = cycles_by_task(results[(model_id, "spanish")])
        control = cycles_by_task(results[(model_id, "english-control")])
        baseline = cycles_by_task(results[(model_id, "standard-supplied-derived")])
        diffs = [(spanish[i] ** 2 - control[i] ** 2) / 10 for i in task_ids]
        for task_id, diff in zip(task_ids, diffs):
            pooled_task_effects[task_id].append(diff)
        try:
            test = wilcoxon(diffs, zero_method="pratt", alternative="two-sided")
            wilcoxon_result = {"statistic": float(test.statistic), "pvalue": float(test.pvalue)}
        except ValueError as error:
            wilcoxon_result = {"statistic": None, "pvalue": None, "note": str(error)}
        comparisons[model_id] = {
            "paired_task_counts": {
                "shorter": sum(spanish[i] < control[i] for i in task_ids),
                "equal": sum(spanish[i] == control[i] for i in task_ids),
                "longer": sum(spanish[i] > control[i] for i in task_ids),
            },
            "control_vs_historical_cycle1_disagreements": sum(
                (control[i] >= 1) != (baseline[i] >= 1) for i in task_ids
            ),
            "token_caps_by_arm": {
                arm: {"capped": sum(g["hit_token_cap"] for g in generations),
                      "total": len(generations)}
                for arm in ["english-control", "spanish"]
                for generations in [[g for task in results[(model_id, arm)]["tasks"]
                                     for cycle in task["cycles"]
                                     for g in [cycle.get("code_generation"), cycle.get("summary_generation")]
                                     if g]]
            },
            "spanish_minus_english_control_asl": paired_asl_difference(spanish, control),
            "bootstrap_95_ci": paired_bootstrap_ci(spanish, control),
            "wilcoxon_pratt": wilcoxon_result,
            "cycle1_disagreement_rate": sum((spanish[i] >= 1) != (control[i] >= 1) for i in task_ids)
            / len(task_ids),
            "spanish_language_compliance": sum(
                c.get("language_check", {}).get("spanish_compliant", False)
                for t in results[(model_id, "spanish")]["tasks"]
                for c in t["cycles"]
                if c.get("passed") and "summary_raw" in c
            )
            / max(
                1,
                sum(
                    1
                    for t in results[(model_id, "spanish")]["tasks"]
                    for c in t["cycles"]
                    if c.get("passed") and "summary_raw" in c
                ),
            ),
        }

    per_task = {task_id: sum(ds) / len(ds) for task_id, ds in pooled_task_effects.items()}
    rng = random.Random(config["seed"])
    boots = []
    for _ in range(10_000):
        draw = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        boots.append(sum(per_task[i] for i in draw) / len(draw))
    boots.sort()
    comparisons["pooled_fixed_models"] = {
        "estimate": sum(per_task.values()) / len(per_task),
        "bootstrap_95_ci": [boots[250], boots[9750]],
        "note": "The bootstrap resamples tasks. Models are fixed and equally weighted.",
    }
    save(args.output_dir / "comparisons.json", comparisons)
    sample_failures(
        results, args.output_dir / "failure_analysis_sample.csv", config["seed"], sample_size=20
    )
    print(f"Wrote {summary_path}, comparisons.json, and failure_analysis_sample.csv")


if __name__ == "__main__":
    main()
