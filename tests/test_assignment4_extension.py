import json
from pathlib import Path

import pytest

from evaloop.extensions.assignment4_metrics import summarize_cycles
from evaloop.extensions.spanish_loop import (
    ARMS,
    EXPECTED_HASHES,
    GENERATION_PROMPT,
    STANDARD_SUMMARY_PROMPT,
    extract_code,
    original_assert_line,
    sha256,
    verify_standard_prompt_hashes,
    wrap_summary,
)
from scripts.run_assignment4 import run_task


DATA = Path(__file__).resolve().parents[2] / "assignment2-data"


def test_standard_prompts_are_byte_pinned():
    verify_standard_prompt_hashes()
    assert sha256(GENERATION_PROMPT) == EXPECTED_HASHES["generation"]
    assert sha256(STANDARD_SUMMARY_PROMPT) == EXPECTED_HASHES["standard_summary"]
    assert not GENERATION_PROMPT.endswith("\n")
    assert not STANDARD_SUMMARY_PROMPT.endswith("\n")


def test_standard_prompts_match_supplied_markdown_byte_for_byte():
    prompt_document = (DATA / "prompts/task_prompts.md").read_text()
    generation_block = f"### Code generation prompt\n\n```\n{GENERATION_PROMPT}\n```"
    summary_block = f"### Code summarization prompt\n\n```\n{STANDARD_SUMMARY_PROMPT}\n```"
    assert generation_block in prompt_document
    assert summary_block in prompt_document


def test_wrapper_matches_supplied_record_exactly():
    supplied = json.loads((DATA / "loops/qwen2.5-coder-7b-instruct.json").read_text())
    task = supplied["tasks"][0]
    stored = task["cycles"][0]["summary"]
    body = stored.splitlines()[1]
    assertion = original_assert_line(task["initial_prompt"])
    assert wrap_summary(body, assertion) == stored


def test_first_code_block_extraction():
    raw = "text\n```python\ndef f():\n    return 1\n```\n```python\nBAD\n```"
    assert extract_code(raw) == "def f():\n    return 1"
    assert extract_code("  def f(): return 1  ") == "def f(): return 1"


def test_control_arms_differ_only_by_language_slot():
    code = "def f(x): return x"
    english = ARMS["english-control"].prompt(code)
    spanish = ARMS["spanish"].prompt(code)
    assert english.replace("English", "Spanish") == spanish


def test_metric_matches_supplied_leaderboard():
    loops = json.loads((DATA / "loops/gemini-2.5-flash-lite.json").read_text())
    leaderboard = json.loads((DATA / "leaderboard.json").read_text())
    row = next(
        r for r in leaderboard["models"] if r["model_id"] == "gemini-2.5-flash-lite"
    )
    metrics = summarize_cycles([t["successful_cycles"] for t in loops["tasks"]])
    assert metrics["success_rate"] == pytest.approx(row["successRate"], abs=5e-4)
    assert metrics["asl_score"] == pytest.approx(row["aslScore"], abs=5e-4)


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        text = "def f(x):\n    return x" if len(self.calls) % 2 else "Return the input unchanged."
        return {
            "text": text,
            "input_tokens": 1,
            "output_tokens": 1,
            "hit_token_cap": False,
            "seconds": 0.0,
        }


class FakeTester:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def test_code(self, code, task_id, entry_point):
        passed = next(self.outcomes)
        return {
            "success": passed,
            "base_success": passed,
            "plus_success": passed,
            "base_details": [],
            "plus_details": [],
        }


class FakeLanguageChecker:
    def check_spanish(self, text):
        return {"language": "spanish", "spanish_confidence": 1.0, "spanish_compliant": True}


PROBLEM = {
    "prompt": '"""\nReturn the input.\nassert f(1) == 1\n"""\n',
    "entry_point": "f",
}


def test_stops_immediately_on_first_failure():
    backend = FakeBackend()
    result = run_task(
        "Mbpp/1", PROBLEM, FakeTester([False]), backend, "english-control", 10, FakeLanguageChecker()
    )
    assert result["successful_cycles"] == 0
    assert len(result["cycles"]) == 1
    assert len(backend.calls) == 1


def test_ten_cycle_termination_and_assertion_preservation():
    backend = FakeBackend()
    result = run_task(
        "Mbpp/1", PROBLEM, FakeTester([True] * 10), backend, "spanish", 10, FakeLanguageChecker()
    )
    assert result["successful_cycles"] == 10
    assert result["max_cycles_reached"] is True
    assert len(result["cycles"]) == 10
    assert len(backend.calls) == 20
    assert all("assert f(1) == 1" in cycle["summary"] for cycle in result["cycles"])
