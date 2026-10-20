"""Prompts and wrapper logic for the Assignment 4 language conditions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional


GENERATION_PROMPT = (
    "Generate Python code for the following task:\n\n"
    "{description}\n\n"
    "Return only the code, without explanations or comments."
)

STANDARD_SUMMARY_PROMPT = (
    "Summarize what the following code does in a detailed description:\n\n"
    "```\n{code}\n```\n\n"
    "Return only the summary, without explanations or comments."
)

CONTROL_SUMMARY_PROMPT = (
    "Summarize what the following code does in a detailed {language} description. "
    "Preserve every behavioral requirement, edge case, input/output constraint, "
    "and function-signature requirement.\n\n"
    "```\n{code}\n```\n\n"
    "Return only the summary in {language}, without explanations, comments, or code."
)

EXPECTED_HASHES = {
    "generation": "8fd73b0cf0084b48a2bbba7b0e81827266b1b67819164926d6b2d00203f25190",
    "standard_summary": "f705d73dafce40b2d17cf03fc03f8db146e617e490e1ebe39050dfe7b55fe3da",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_standard_prompt_hashes() -> None:
    """Fail if either copied package prompt changes."""
    actual = {
        "generation": sha256(GENERATION_PROMPT),
        "standard_summary": sha256(STANDARD_SUMMARY_PROMPT),
    }
    if actual != EXPECTED_HASHES:
        raise AssertionError(f"Standard prompt drift: expected {EXPECTED_HASHES}, got {actual}")


def extract_code(raw_response: str) -> str:
    """Apply the data package's first-code-block extraction rule."""
    matches = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", raw_response)
    return (matches[0] if matches else raw_response).strip()


def original_assert_line(initial_prompt: str) -> str:
    """Return the last assertion from the original MBPP+ prompt."""
    assertions = [line for line in initial_prompt.splitlines() if line.lstrip().startswith("assert ")]
    if not assertions:
        raise ValueError("Initial prompt contains no assertion line")
    return assertions[-1]


def wrap_summary(summary: str, assert_line: str) -> str:
    """Create exactly the next-cycle wrapper used by the supplied records."""
    return f'"""\n{summary.strip()}\n{assert_line}\n"""\n'


@dataclass(frozen=True)
class SummaryArm:
    name: str
    language: Optional[str]

    def prompt(self, code: str) -> str:
        if self.name == "standard":
            return STANDARD_SUMMARY_PROMPT.format(code=code)
        if self.name not in {"english-control", "spanish"}:
            raise ValueError(f"Unknown arm: {self.name}")
        if not self.language:
            raise ValueError(f"Arm {self.name} requires a language")
        return CONTROL_SUMMARY_PROMPT.format(code=code, language=self.language)


ARMS = {
    "standard": SummaryArm("standard", None),
    "english-control": SummaryArm("english-control", "English"),
    "spanish": SummaryArm("spanish", "Spanish"),
}
