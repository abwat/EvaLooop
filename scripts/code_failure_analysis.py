#!/usr/bin/env python3
"""Restore AI-assisted codes confirmed by the author on 20 October 2026."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "analysis_assignment4" / "failure_analysis_sample.csv"


CODES = {
    ("opencoder-8b-instruct", "Mbpp/555"): (
        "wrong implementation of retained description",
        "Summary keeps cubes minus integers; code reverses the subtraction.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/251"): (
        "wrong implementation of retained description",
        "Summary requires a prefix before each item; code prefixes the list only once.",
    ),
    ("codeqwen1.5-7b-chat", "Mbpp/251"): (
        "requirement dropped",
        "Summary loses the required prefix ordering; code repeats the original items.",
    ),
    ("opencoder-8b-instruct", "Mbpp/290"): (
        "requirement dropped",
        "Summary invents a None result for empty inputs; code follows that change.",
    ),
    ("opencoder-8b-instruct", "Mbpp/758"): (
        "requirement dropped",
        "Summary makes equality order-insensitive; code sorts each sublist.",
    ),
    ("codeqwen1.5-7b-chat", "Mbpp/67"): (
        "wrong implementation of retained description",
        "Summary still specifies Bell numbers; the dynamic program is incorrect.",
    ),
    ("opencoder-8b-instruct", "Mbpp/643"): (
        "requirement dropped",
        "Requirement changes from an interior z to z at the exact middle index.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/265"): (
        "requirement dropped",
        "Stride-based partitioning becomes contiguous chunks.",
    ),
    ("opencoder-8b-instruct", "Mbpp/93"): (
        "requirement dropped",
        "Summary drops negative exponents; repeated multiplication fails those tests.",
    ),
    ("opencoder-8b-instruct", "Mbpp/732"): (
        "wrong implementation of retained description",
        "Summary says one colon; code inserts two for commas and periods.",
    ),
    ("codeqwen1.5-7b-chat", "Mbpp/11"): (
        "requirement dropped",
        "Summary drops the single-occurrence case; code leaves that input unchanged.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/245"): (
        "requirement dropped",
        "Bitonic subsequence becomes two separated subsequences with excluded endpoints.",
    ),
    ("codeqwen1.5-7b-chat", "Mbpp/245"): (
        "requirement dropped",
        "Maximum-sum bitonic subsequence becomes a contiguous-subarray problem.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/440"): (
        "wrong implementation of retained description",
        "Summary retains first-adverb matching; regex misses boundary cases.",
    ),
    ("opencoder-8b-instruct", "Mbpp/633"): (
        "requirement dropped",
        "Pairwise XOR sum becomes an operand sum limited to XOR equal to one.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/758"): (
        "wrong implementation of retained description",
        "Summary is correct; code looks up tuple keys in a list of lists and returns zeros.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/643"): (
        "wrong implementation of retained description",
        "Summary excludes boundary z; regex accepts it at word starts and ends.",
    ),
    ("qwen2.5-coder-7b-instruct", "Mbpp/283"): (
        "wrong implementation of retained description",
        "Summary is correct; code zips ten frequency counts with the input digits.",
    ),
    ("opencoder-8b-instruct", "Mbpp/273"): (
        "wrong implementation of retained description",
        "Summary preserves indexed subtraction; code discards nonpositive pairs.",
    ),
    ("opencoder-8b-instruct", "Mbpp/14"): (
        "requirement dropped",
        "Triangular prism becomes a parallelepiped, dropping the one-half factor.",
    ),
}


def main() -> None:
    with SAMPLE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise ValueError(f"Expected frozen 20-row sample, found {len(rows)}")
    found = {(row["model"], row["task_id"]) for row in rows}
    if found != set(CODES):
        raise ValueError("Failure sample no longer matches the manually reviewed frozen sample")
    for row in rows:
        row["failure_mode"], row["notes"] = CODES[(row["model"], row["task_id"])]
        row["draft_reviewer"] = "Codex GPT-6"
        row["human_reviewer"] = "Aniruddh S Bhagwat"
        row["human_confirmed"] = "2026-10-20"
    with SAMPLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Coded {len(rows)} failures in {SAMPLE}")


if __name__ == "__main__":
    main()
