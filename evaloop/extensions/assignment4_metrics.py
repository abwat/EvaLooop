"""Exact coursework metrics and paired uncertainty estimates."""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def task_scores(successful_cycles: Iterable[int], max_cycles: int = 10) -> List[float]:
    return [(int(s) ** 2) / max_cycles for s in successful_cycles]


def summarize_cycles(successful_cycles: Sequence[int], max_cycles: int = 10) -> Dict[str, float]:
    if not successful_cycles:
        raise ValueError("At least one task is required")
    n = len(successful_cycles)
    return {
        "n_tasks": n,
        "success_rate": sum(s >= 1 for s in successful_cycles) / n,
        "asl_score": sum(task_scores(successful_cycles, max_cycles)) / n,
        "mean_successful_cycles": sum(successful_cycles) / n,
    }


def cycles_by_task(result: Mapping) -> Dict[str, int]:
    tasks = result.get("tasks", result.get("prompt_results", []))
    return {str(t["task_id"]): int(t["successful_cycles"]) for t in tasks}


def paired_asl_difference(
    treatment: Mapping[str, int], control: Mapping[str, int], max_cycles: int = 10
) -> float:
    if set(treatment) != set(control):
        raise ValueError("Treatment and control must contain identical task IDs")
    ids = sorted(treatment)
    return sum((treatment[i] ** 2 - control[i] ** 2) / max_cycles for i in ids) / len(ids)


def paired_bootstrap_ci(
    treatment: Mapping[str, int],
    control: Mapping[str, int],
    *,
    repetitions: int = 10_000,
    seed: int = 20261017,
    max_cycles: int = 10,
) -> Tuple[float, float]:
    """Percentile CI; the resampling unit is the shared MBPP+ task."""
    if set(treatment) != set(control):
        raise ValueError("Treatment and control must contain identical task IDs")
    ids = sorted(treatment)
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        draw = [ids[rng.randrange(len(ids))] for _ in ids]
        values.append(
            sum((treatment[i] ** 2 - control[i] ** 2) / max_cycles for i in draw)
            / len(draw)
        )
    values.sort()
    return values[int(0.025 * repetitions)], values[int(0.975 * repetitions)]


def cycle_histogram(successful_cycles: Iterable[int]) -> Dict[int, int]:
    return dict(sorted(Counter(map(int, successful_cycles)).items()))
