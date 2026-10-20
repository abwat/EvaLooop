# Assignment 4: Spanish summaries in EvaLoop

Public fork: https://github.com/abwat/EvaLooop
Assignment branch: https://github.com/abwat/EvaLooop/tree/assignment4

Upstream: https://github.com/evalooop/EvaLooop, branch `dev`.
Patch base commit: `ce22fe2f80c8f36da85cc8a0cca49c2bcc8b1cb9`.
The fork was verified as public and as a fork of `evalooop/EvaLooop` on 20 October 2026.

## Question

Does requiring Spanish intermediate summaries change Average Sustainable Loops (ASL) after
controlling for the wording of the summary prompt?

I compare three conditions on the same 100 MBPP+ tasks:

1. `standard-supplied-derived`: the supplied EvaLoop records, restricted to the selected tasks;
2. `english-control`: a detail-preserving summary prompt in English; and
3. `spanish`: the same prompt with Spanish as the requested output language.

The English and Spanish runs use three models. Each task stops at the first failed Base or Plus
test, or after ten passing cycles. Model revisions and inference settings are recorded in
`assignment4_config.json` and `PROTOCOL_ASSIGNMENT4.md`.

## Files

- `report.pdf`: three-page report.
- `summary.csv`: six performed runs; byte-identical working copy in `analysis_assignment4/`.
- `task_ids.json`: the 100 evaluated task IDs, in run order.
- `protocol/task_order.json`: the full seeded permutation and selected subsets.
- `protocol/prompts.json`: exact generation, summary, and wrapper templates exported from the extension.
- `analysis_assignment4/baseline_summary.csv`: three aggregate historical comparisons.
- `analysis_assignment4/baseline_record_references.json`: supplied model, task, and cycle references.
- `results/`: six new runs. Supplied per-cycle records are not copied into the archive.
- `analysis_assignment4/comparisons.json`: paired effects, confidence intervals, tests, and
  language checks.
- `analysis_assignment4/failure_analysis_sample.csv`: 20 Spanish-loop failures manually reviewed by the author on 20 October 2026, following AI-assisted draft coding.
- `Assignment4_Cloud_Run.ipynb`: cloud commands used for the experiment.
- `assignment4.patch`: changes relative to `evalooop/EvaLooop` branch `dev`.
- `evaloop/extensions/`, `scripts/`, and `tests/`: implementation and tests.

The private `assignment2-data` directory is not included.

## Environment

The accepted runs used a Google Cloud `g2-standard-4` VM with one NVIDIA L4 in
`us-central1-a`. The VM began as Spot and was later changed to standard provisioning after
preemptions. Inference used BF16, batch size 1, greedy decoding, repository chat templates, no
system prompt, a 1,024-token output cap, and ten cycles.

Install the experiment requirements from an EvaLoop checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-assignment4-cloud.txt
python -m pip install -e .
python -m pytest -q
```

The cloud run recorded 84 passing tests.

The model revisions are fixed rather than resolved from changing repository heads:

| Model repository | Revision |
|---|---|
| Qwen/Qwen2.5-Coder-7B-Instruct | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| Qwen/CodeQwen1.5-7B-Chat | `7b0cc3380fe815e6f08fe2f80c03e05a8b1883d8` |
| infly/OpenCoder-8B-Instruct | `01badbbf10c2dfd7e2a0b5f570065ef44548576c` |

The source files in this ZIP overlay an upstream checkout. For a clean installation, check out
the base commit above, apply `assignment4.patch`, and then install the requirements. The
environment needs Python 3.10+, a CUDA-compatible PyTorch installation, Transformers 4.57.6,
EvalPlus 0.3.1, Lingua 2.0.2, and the other dependencies in the requirements file. Full analysis
requires SciPy; the summary-only rebuild needs only Python's standard library.

To reproduce an inference configuration on a dedicated CUDA VM, choose one model ID from
`assignment4_config.json` and run both arms. For example:

```bash
python scripts/run_assignment4.py --model-id qwen2.5-coder-7b-instruct \
  --arm english-control --n-tasks 100 --max-cycles 10 --max-new-tokens 1024
python scripts/run_assignment4.py --model-id qwen2.5-coder-7b-instruct \
  --arm spanish --n-tasks 100 --max-cycles 10 --max-new-tokens 1024
```

Repeat with `codeqwen1.5-7b-chat` and `opencoder-8b-instruct` for all six configurations.
These commands perform inference; they are unnecessary for rebuilding archived numbers.

Generated programs were tested on the dedicated Google Cloud evaluation VM through the
upstream EvalPlus tester, with process isolation and execution limits. They were not executed
on the author's personal machine. EvalPlus's reliability guard is not a complete security sandbox;
the cloud VM provided the separation from local personal files.
The saved experiment records do not establish the VM service-account permissions, metadata
access, or credentials available to generated programs. No claim of credential-free execution
is made. Reproduction should use a disposable VM without privileged service accounts or personal
credentials, with resource permissions and network access restricted before testing.

## Rebuild the analysis

Rebuild the six-run `summary.csv` using only the six archived result JSONs, configuration,
task order, and Python's standard library. No course package, GPU, SciPy, or model access is needed:

```bash
python scripts/analyze_assignment4.py --summary-only
python scripts/analyze_assignment4.py --summary-only --output-dir .
python scripts/export_submission_metadata.py
python tests/test_summary_rebuild.py
```

The isolated test disables site packages and confirms a byte-identical six-row summary.
It also checks that an incomplete run is rejected. Baseline rebuilding additionally needs
the private course package. Place `assignment2-data` beside the checkout, then run:

```bash
python scripts/analyze_assignment4.py \
  --data-dir ../assignment2-data \
  --results-dir results \
  --output-dir analysis_assignment4
python scripts/code_failure_analysis.py
python scripts/analyze_assignment4.py --summary-only --output-dir .
```

The first command rebuilds the six-row `summary.csv`, the three-row `baseline_summary.csv`,
baseline references, `comparisons.json`, and the seeded failure sample. Baseline records are
read in memory from the course package. The second command reapplies the author-reviewed Spanish codes
and their review metadata.

`comparisons.json` also reports shorter/equal/longer task counts, cycle-1 disagreement with
the supplied records, and capped/total generations per arm. These checks use saved results
and require no new inference.

## Post-run audits

`POST_RUN_AUDIT.md` documents the retrospective sampling, coding rubric, AI assistance,
planning history, and cost evidence. The additional files are:

- `baseline_failure_audit.csv`: ten English-control and ten standard-loop failures, with references and rationales;
- `failure_mode_comparison.csv`: the three descriptive label distributions;
- `language_audit.csv`: 30 summaries from distinct task IDs within each model;
- `truncation_sensitivity.json`: effects after excluding task pairs with capped generations.

These files are under `analysis_assignment4/`. Codex (GPT-6) drafted qualitative labels.
The author manually reviewed the 20 Spanish failure samples on 20 October 2026 and retained
their labels. The ten English-control failures, ten standard failures, and 30 language samples
remain AI-assisted supplementary checks without recorded human confirmation. The audit document
contains the prompts, decision rules, review status, and ambiguous cases. Incremental API cost
was $0; subscription cost per example and exact decoding settings are unavailable.
Quantitative scores never use these labels.

Rebuild the recorded audits and sensitivity check without new inference:

```bash
python scripts/posthoc_audit.py
python scripts/posthoc_audit.py --export-audits --data-dir ../assignment2-data
```

The first command uses only performed-run JSONs. The second verifies selection against supplied
standard records, then exports the fixed audit decisions. Standard per-cycle text is never copied.

CodeQwen's effect is +0.411 on 97 uncapped pairs, compared with +0.399 on all 100. Exclusion
depends on a post-treatment variable and does not establish the direction of truncation bias.
The language audit found 29 samples with Spanish prose and one with a minor English word;
all 30 were predominantly Spanish. This does not establish exclusive Spanish in every summary.

## Cost

The billing screenshot in `evidence/cloud_billing_20261020.png` shows US$20.75 for
14–20 October 2026, including Compute Engine US$17.63 and Persistent Disk US$3.01.
This supersedes the earlier approximate $6 estimate. I forgot to turn off the VM after the
experiment, so it kept running and added charges beyond the evaluation itself. The displayed
total exceeds the planned $10 budget. The screenshot does not show project filters or separate
active experiment costs from idle charges, so the exact experiment-only cost is unavailable.
Accepted generation time is 11.22 hours and excludes loading, tests, idle time, and interrupted
work. The screenshot does not establish the current VM or disk status.

## Rebuild the report

Install `python-docx==1.2.0`, then run `python scripts/build_report.py --output report.docx`.
Export the document to PDF in Word or LibreOffice and confirm that it remains three pages.
The report source contains the values verified by the analysis above.

## Changes to the framework

The existing generation and testing modules are unchanged. The extension adds prompts,
extraction, wrapper logic, and metrics under `evaloop/extensions/`, plus runners, analysis,
and reporting scripts under `scripts/`. The README explains the new workflow. The patch
contains all changes relative to upstream `dev`.

## Main result

The pooled Spanish-minus-English ASL estimate is -0.074 with a task-bootstrap 95% confidence
interval of [-0.432, 0.294]. Effects differ by model, and no model-level confidence interval
excludes zero. Lingua classifies 100% of Spanish summaries as Spanish, and cycle-1 disagreement between the new
arms is zero for all three models.
