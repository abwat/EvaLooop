# Assignment 4 protocol

Version: `1.0-frozen`
Frozen: 2026-10-17

## Research question

Does Spanish code summarization reduce quadratic Average Sustainable Loops (ASL) beyond the
effect of changing the summary prompt?

## Conditions

- `standard-supplied-derived`: supplied standard-loop records restricted to the selected tasks.
  These are historical records, not new inference.
- `english-control`: a detail-preserving summary prompt requesting English.
- `spanish`: the same prompt with `English` replaced by `Spanish`.

All conditions use the package code-generation prompt. After a passing cycle, the raw summary is
placed in a docstring with the final assertion from the original MBPP+ prompt. The resulting text
is the next cycle's input. A task stops at the first Base or Plus test failure or after ten passing
cycles.

Only the generated prose changes language. Identifiers, literals copied from code, the retained
assertion, generation prompt, and wrapper remain unchanged. Lingua checks the raw summary before
wrapping, using only its English and Spanish detectors. Noncompliant summaries remain in the
primary analysis.

## Models and inference

The model repositories and revisions are fixed in `assignment4_config.json`:

- Qwen2.5-Coder-7B-Instruct
- CodeQwen1.5-7B-Chat
- OpenCoder-8B-Instruct

The three models are a fixed comparison set. Inference uses BF16, batch size 1, each repository's
chat template, no system prompt, `do_sample=False`, and at most 1,024 new tokens. The supplied
historical records do not state their backend, precision, chat template, or system prompt, so
historical-to-new differences are descriptive. The English-Spanish comparison uses the same
backend and settings.

OpenCoder requires repository code at its pinned revision. Remote code is disabled for both Qwen
models. OpenCoder generation uses the EOS token from the model's generation configuration because
its tokenizer exposes a different legacy EOS token.

## Tasks

`protocol/task_order.json` contains a seed-20261017 permutation of the 378 supplied task IDs. The
first 100 IDs form the principal sample; the first 25 of those form the gate and smoke sample.
Any extension follows the same order in blocks of 25.

Smoke results count toward the principal runs only when the prompt, model revision, chat template,
precision, decoding settings, token cap, and tests are unchanged. Otherwise, the affected arm is
rerun in a new output directory.

## Pre-run checks

- Generate the same 25 cycle-1 prompts three times for Qwen2.5-Coder-7B at batch size 1. Stop if
  more than 2 of 25 tasks have inconsistent pass/fail outcomes.
- Report disagreement between the first repetition and the supplied cycle-1 result separately.
- Review generations that reach 1,024 tokens. Increase the common cap and rerun affected arms if
  more than 5% are capped or code blocks are visibly incomplete.
- Record a ceiling warning if English-control mean cycles are at least 9.5 or at least 20 of 25
  tasks reach cycle 10. Replace the model before principal runs if the ceiling prevents comparison.
- Report Spanish compliance. If it is below 80%, keep the intention-to-treat result and add a
  compliant-summary subgroup analysis.
- Check whether cycle-10 completions are concentrated in historically easy tasks.

## Metrics and statistics

For task `i`, `s_i` is the number of successful cycles from 0 to 10.

- Success rate: `mean(s_i >= 1)`
- Quadratic ASL: `mean(s_i^2 / 10)`

A unit test reconstructs the supplied Gemini 2.5 Flash-Lite leaderboard values from its JSON
record. The primary contrast is Spanish minus English-control ASL, paired by task and reported
separately for each model. Uncertainty is a 95% percentile interval from 10,000 task bootstrap
samples. The paired test is two-sided Wilcoxon signed-rank with Pratt zero handling.

A secondary pooled estimate averages the three model effects within each task and then
bootstraps tasks. Models are fixed and equally weighted. Cycle-1 pass/fail disagreement is
reported as a noise check.

## Failure analysis

Sample 20 Spanish runs that pass cycle 1 and fail before cycle 10. Use seeded sampling across
models and failure cycles. Compare each sampled task with its English-control and supplied
standard record. Use these labels:

- requirement dropped
- wrong implementation of retained description
- signature/name drift
- assert over-reliance
- formatting/extraction
- language mixing
- other

If fewer than 20 Spanish failures qualify, add English-control failures with the same procedure.
If the sample is still short, run the next 25 tasks in the frozen order.

## Cost and stopping rule

The planned platform is one Google Cloud `g2-standard-4` L4 VM. Stop at 20 GPU-hours or USD 10,
including disk charges. Record the SKU, zone, provisioning type, measured generation time, and
effective out-of-pocket cost. No external LLM judge is used for test outcomes or quantitative
metrics. Retrospective AI-assisted qualitative checks and the author's subsequent review of
the 20 Spanish failures are documented in POST_RUN_AUDIT.md.

## Anticipated limitations

The sample contains 100 short MBPP+ tasks and three 7-8B models, two from the Qwen family. Tests
may miss semantic errors, benchmark contamination is possible, and ASL weights long survival
quadratically. The retained assertion may reduce language effects. Language detection can be
uncertain for code-heavy text. Historical records are not execution-matched to the new runs.
