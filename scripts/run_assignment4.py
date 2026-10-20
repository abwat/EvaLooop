#!/usr/bin/env python3
"""Run one checkpointed Assignment 4 arm on a CUDA cloud VM."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from evaloop.evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset
from evaloop.extensions.spanish_loop import (
    ARMS,
    GENERATION_PROMPT,
    extract_code,
    original_assert_line,
    verify_standard_prompt_hashes,
    wrap_summary,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LanguageChecker:
    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self.spanish = Language.SPANISH
        self.detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.SPANISH
        ).build()

    def check_spanish(self, text: str) -> Dict[str, Any]:
        detected = self.detector.detect_language_of(text)
        confidence = self.detector.compute_language_confidence(text, self.spanish)
        return {
            "language": detected.name.lower() if detected else "unknown",
            "spanish_confidence": float(confidence),
            "spanish_compliant": detected == self.spanish,
        }


class TransformersBackend:
    """Batch-size-one, greedy CUDA inference with a repository chat template."""

    def __init__(self, model_cfg: Mapping[str, str], max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; this script is for the cloud VM")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.repo = model_cfg["hf_repo"]
        self.revision = model_cfg["revision"]
        self.trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repo,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        ).eval()
        # OpenCoder's tokenizer EOS differs from its generation-config EOS.
        self.eos_token_id = self.model.generation_config.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.model.config.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id

    def generate(self, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        # CodeQwen emits token_type_ids, but its causal LM rejects them.
        encoded.pop("token_type_ids", None)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        generated = output[0, input_tokens:]
        generated_tokens = int(generated.shape[-1])
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "input_tokens": input_tokens,
            "output_tokens": generated_tokens,
            "hit_token_cap": generated_tokens >= self.max_new_tokens,
            "seconds": time.monotonic() - started,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def fresh_result(model_cfg: Mapping[str, str], arm: str, task_ids: list[str], args: Any) -> Dict:
    import torch
    import transformers

    return {
        "schema_version": "assignment4-v1",
        "model_id": model_cfg["model_id"],
        "hf_repo": model_cfg["hf_repo"],
        "revision": model_cfg["revision"],
        "configuration": arm,
        "task_ids": task_ids,
        "max_cycles": args.max_cycles,
        "inference": {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "batch_size": 1,
            "max_new_tokens": args.max_new_tokens,
            "dtype": "bfloat16",
            "system_prompt": None,
            "chat_template": "tokenizer repository default",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "host": {"python": sys.version, "platform": platform.platform()},
        "started_at": utc_now(),
        "completed_at": None,
        "tasks": [],
    }


def atomic_json(path: Path, payload: Mapping) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def test_code(tester: EvalPlusCodeTester, problems: Mapping, task_id: str, code: str) -> Dict:
    raw = tester.test_code(code, task_id, problems[task_id]["entry_point"])
    return {
        "passed": bool(raw.get("success", False)),
        "base_passed": bool(raw.get("base_success", False)),
        "plus_passed": bool(raw.get("plus_success", False)),
        "error": raw.get("error"),
        "base_details": str(raw.get("base_details", "")),
        "plus_details": str(raw.get("plus_details", "")),
    }


def run_task(
    task_id: str,
    problem: Mapping,
    tester: EvalPlusCodeTester,
    backend: TransformersBackend,
    arm_name: str,
    max_cycles: int,
    language_checker: LanguageChecker,
) -> Dict:
    initial = problem["prompt"]
    assertion = original_assert_line(initial)
    current = initial
    cycles = []
    successful = 0
    for cycle_no in range(1, max_cycles + 1):
        generation_prompt = GENERATION_PROMPT.format(description=current)
        generated = backend.generate(generation_prompt)
        code = extract_code(generated["text"])
        tested = test_code(tester, {task_id: problem}, task_id, code)
        cycle = {
            "cycle": cycle_no,
            "cycle_input": current,
            "generation_prompt": generation_prompt,
            "code_raw": generated["text"],
            "code": code,
            "code_generation": {k: v for k, v in generated.items() if k != "text"},
            **tested,
        }
        if not tested["passed"]:
            cycles.append(cycle)
            break
        successful = cycle_no
        summary_prompt = ARMS[arm_name].prompt(code)
        summarized = backend.generate(summary_prompt)
        summary_raw = summarized["text"].strip()
        wrapped = wrap_summary(summary_raw, assertion)
        cycle.update(
            {
                "summary_prompt": summary_prompt,
                "summary_raw": summary_raw,
                "summary": wrapped,
                "summary_generation": {k: v for k, v in summarized.items() if k != "text"},
            }
        )
        if arm_name == "spanish":
            cycle["language_check"] = language_checker.check_spanish(summary_raw)
        cycles.append(cycle)
        current = wrapped
    return {
        "task_id": task_id,
        "initial_prompt": initial,
        "successful_cycles": successful,
        "max_cycles_reached": successful == max_cycles,
        "cycles": cycles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("assignment4_config.json"))
    parser.add_argument("--task-order", type=Path, default=Path("protocol/task_order.json"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--arm", choices=["standard", "english-control", "spanish"], required=True)
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    verify_standard_prompt_hashes()
    config = json.loads(args.config.read_text())
    model_cfg = next((m for m in config["models"] if m["model_id"] == args.model_id), None)
    if model_cfg is None:
        raise SystemExit(f"Model {args.model_id!r} is not in {args.config}")
    order = json.loads(args.task_order.read_text())
    task_ids = order["task_ids"][: args.n_tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.model_id}__{args.arm}.json"

    problems, ground_truth = load_mbpp_dataset()
    missing = sorted(set(task_ids) - set(problems))
    if missing:
        raise SystemExit(f"Frozen IDs missing from installed EvalPlus dataset: {missing}")
    tester = EvalPlusCodeTester(problems, ground_truth)
    checker = LanguageChecker()
    backend = TransformersBackend(model_cfg, args.max_new_tokens)
    if output.exists():
        result = json.loads(output.read_text())
        frozen = result.get("task_ids", [])
        invariant = (
            result.get("model_id") == model_cfg["model_id"]
            and result.get("revision") == model_cfg["revision"]
            and result.get("configuration") == args.arm
            and result.get("max_cycles") == args.max_cycles
            and result.get("inference", {}).get("max_new_tokens") == args.max_new_tokens
        )
        if not invariant or task_ids[: len(frozen)] != frozen:
            raise SystemExit(
                "Checkpoint protocol differs from this run. Preserve it, choose a new output "
                "directory, and rerun affected tasks as required by the frozen protocol."
            )
        result["task_ids"] = task_ids  # The smoke set is a prefix of the main set.
        result["completed_at"] = None
    else:
        result = fresh_result(model_cfg, args.arm, task_ids, args)
    done = {t["task_id"] for t in result["tasks"]}
    try:
        for index, task_id in enumerate(task_ids, start=1):
            if task_id in done:
                continue
            print(f"[{index}/{len(task_ids)}] {model_cfg['model_id']} {args.arm} {task_id}", flush=True)
            task_result = run_task(
                task_id, problems[task_id], tester, backend, args.arm, args.max_cycles, checker
            )
            result["tasks"].append(task_result)
            atomic_json(output, result)
    finally:
        backend.close()
    result["completed_at"] = utc_now()
    atomic_json(output, result)
    print(f"Completed {output}")


if __name__ == "__main__":
    main()
