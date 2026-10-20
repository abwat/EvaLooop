"""Select reproducible qualitative audits and compute post-run sensitivity checks.

No model inference is performed. Standard-loop input text stays in the private data package.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Qualitative decisions made after reading each selected trace, not automatic classifiers.
BASELINE_CODES = {
    ('english-control', 'opencoder-8b-instruct', 'Mbpp/726'): ('requirement dropped', 'Summary invents a squared singleton result instead of an empty adjacent-product tuple.'),
    ('english-control', 'qwen2.5-coder-7b-instruct', 'Mbpp/93'): ('requirement dropped', 'Summary narrows exponentiation to positive integers; recursive code does not handle negative powers.'),
    ('english-control', 'codeqwen1.5-7b-chat', 'Mbpp/736'): ('requirement dropped', 'Summary replaces the left insertion boundary with a strict less-than comparison.'),
    ('english-control', 'opencoder-8b-instruct', 'Mbpp/616'): ('wrong implementation of retained description', 'Elementwise modulo remains the task; regeneration reverses the original operand order.'),
    ('english-control', 'qwen2.5-coder-7b-instruct', 'Mbpp/245'): ('requirement dropped', 'Summary changes the shared bitonic peak into non-overlapping subsequences excluding the current item.'),
    ('english-control', 'opencoder-8b-instruct', 'Mbpp/397'): ('wrong implementation of retained description', 'Summary retains the median; regenerated branches return c for several permutations.'),
    ('english-control', 'qwen2.5-coder-7b-instruct', 'Mbpp/57'): ('wrong implementation of retained description', 'Summary retains concatenation sorting; code calls functools without importing it.'),
    ('english-control', 'opencoder-8b-instruct', 'Mbpp/475'): ('requirement dropped', 'Summary invents a key-based tie rule; the prior passing code preserves insertion order on ties.'),
    ('english-control', 'opencoder-8b-instruct', 'Mbpp/555'): ('wrong implementation of retained description', 'Summary retains cubes minus integers; generated subtraction reverses the operands.'),
    ('english-control', 'opencoder-8b-instruct', 'Mbpp/123'): ('wrong implementation of retained description', 'Summary retains amicable numbers; helper mutates n and miscomputes the proper-divisor sum.'),
    ('standard', 'opencoder-8b-instruct', 'Mbpp/123'): ('wrong implementation of retained description', 'Operative summary still requests amicable-number sum despite repeated fences; the divisor helper is incorrect. Formatting is a secondary symptom.'),
    ('standard', 'qwen2.5-coder-7b-instruct', 'Mbpp/227'): ('wrong implementation of retained description', 'Minimum-of-three requirement remains; strict comparisons mishandle tied minima.'),
    ('standard', 'qwen2.5-coder-7b-instruct', 'Mbpp/750'): ('wrong implementation of retained description', 'Concatenation is retained; code prepends the tuple despite the retained list-first assertion. Summary order is ambiguous.'),
    ('standard', 'codeqwen1.5-7b-chat', 'Mbpp/736'): ('requirement dropped', 'Summary reverses the insertion comparison and loses the left-boundary rule.'),
    ('standard', 'codeqwen1.5-7b-chat', 'Mbpp/14'): ('requirement dropped', 'Triangular prism changes to trapezoidal prism; the resulting signature adds a fourth argument.'),
    ('standard', 'opencoder-8b-instruct', 'Mbpp/162'): ('requirement dropped', 'Summary loses the step-two series formula; code computes the ordinary arithmetic series.'),
    ('standard', 'qwen2.5-coder-7b-instruct', 'Mbpp/130'): ('wrong implementation of retained description', 'Maximum-frequency task remains; online counting changes first-occurrence tie behavior of the prior passing code.'),
    ('standard', 'qwen2.5-coder-7b-instruct', 'Mbpp/251'): ('requirement dropped', 'Insertion before each item becomes insertion between pairs; code removes the first prefix.'),
    ('standard', 'opencoder-8b-instruct', 'Mbpp/11'): ('wrong implementation of retained description', 'Summary retains first/last character removal; code deletes the entire interval and ignores singleton occurrences.'),
    ('standard', 'codeqwen1.5-7b-chat', 'Mbpp/3'): ('requirement dropped', 'Summary reduces non-primality to divisor search and omits the below-two case.'),
}


def load(path):
    return json.loads(path.read_text())


def baseline_sample(config, task_ids, data_dir, arm):
    candidates = []
    for model in config['models']:
        path = (data_dir / 'loops' / model['baseline_file'] if arm == 'standard'
                else ROOT / 'results' / f"{model['model_id']}__english-control.json")
        for task in load(path)['tasks']:
            if task['task_id'] in task_ids and 1 <= task['successful_cycles'] < 10:
                candidates.append((model['model_id'], task))
    rng = random.Random(config['seed'])
    rng.shuffle(candidates)
    selected, covered, used = [], set(), set()
    for model, task in candidates:
        stratum = (model, task['successful_cycles'] + 1)
        if stratum not in covered:
            selected.append((model, task))
            covered.add(stratum)
            used.add((model, task['task_id']))
            if len(selected) == 10:
                return selected
    for model, task in candidates:
        if (model, task['task_id']) not in used:
            selected.append((model, task))
            if len(selected) == 10:
                return selected
    raise ValueError('Fewer than ten eligible failures')


def language_sample(config):
    selected = []
    for model in config['models']:
        result = load(ROOT / 'results' / f"{model['model_id']}__spanish.json")
        candidates = [(model['model_id'], task['task_id'], c['cycle'], c['summary_raw'])
                      for task in result['tasks'] for c in task['cycles'] if 'summary_raw' in c]
        # Two short, four identifier-heavy, and four random summaries per model.
        used = set()
        def take_unique(rows, n):
            chosen = []
            for row in rows:
                if row[1] not in used:
                    chosen.append(row)
                    used.add(row[1])
                    if len(chosen) == n:
                        break
            return chosen
        short = take_unique(sorted(candidates, key=lambda x: (len(x[3].split()), x[1], x[2])), 2)
        rest = [x for x in candidates if x[1] not in used]
        def code_ratio(row):
            code = sum(len(x) for x in re.findall(r'`[^`]+`', row[3]))
            return code / max(1, len(row[3]))
        heavy = take_unique(sorted(rest, key=lambda x: (-code_ratio(x), x[1], x[2])), 4)
        rest = [x for x in rest if x[1] not in used]
        rng = random.Random(config['seed'])
        rng.shuffle(rest)
        random_rows = take_unique(rest, 4)
        selected.extend((stratum, x) for stratum, rows in
                        [('short', short), ('identifier-heavy', heavy), ('random', random_rows)]
                        for x in rows)
    return selected


def sensitivity(config):
    outputs = {}
    for model in config['models']:
        arms = {arm: {t['task_id']: t for t in load(ROOT / 'results' /
                f"{model['model_id']}__{arm}.json")['tasks']}
                for arm in ['english-control', 'spanish']}
        def capped(task):
            return any(g.get('hit_token_cap', False) for cycle in task['cycles']
                       for g in [cycle.get('code_generation'), cycle.get('summary_generation')] if g)
        # Remove a pair if either arm hit the cap, including the unused cycle-10 summary.
        excluded = [i for i in arms['spanish'] if any(capped(a[i]) for a in arms.values())]
        kept = [i for i in arms['spanish'] if i not in excluded]
        def asl(arm):
            return sum(arms[arm][i]['successful_cycles'] ** 2 / 10 for i in kept) / len(kept)
        outputs[model['model_id']] = {
            'n_pairs': len(kept), 'excluded_task_ids': excluded,
            'english_asl': asl('english-control'), 'spanish_asl': asl('spanish'),
            'spanish_minus_english_asl': asl('spanish') - asl('english-control'),
            'interpretation': 'Post hoc exclusion sensitivity; cap status is post-treatment. Not a causal correction.',
        }
    return outputs


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_audits(config, task_ids, data_dir):
    output = ROOT / 'analysis_assignment4'
    baseline_rows = []
    for arm in ['english-control', 'standard']:
        for model, task in baseline_sample(config, task_ids, data_dir, arm):
            label, note = BASELINE_CODES[(arm, model, task['task_id'])]
            baseline_rows.append({'configuration': arm, 'model': model, 'task_id': task['task_id'],
                                  'summary_cycle': task['successful_cycles'],
                                  'failed_cycle': task['successful_cycles'] + 1,
                                  'failure_mode': label, 'notes': note,
                                  'reviewer': 'Codex GPT-6', 'human_confirmed': 'not recorded'})
    write_csv(output / 'baseline_failure_audit.csv', baseline_rows)
    language_rows = []
    for stratum, (model, task, cycle, text) in language_sample(config):
        mixed = (model, task, cycle) == ('qwen2.5-coder-7b-instruct', 'Mbpp/478', 2)
        language_rows.append({'model': model, 'task_id': task, 'cycle': cycle, 'stratum': stratum,
                              'review': 'Spanish with minor English prose' if mixed else 'Spanish prose',
                              'notes': 'English word effectively occurs in Spanish prose.' if mixed
                                       else 'Spanish explanatory prose; copied identifiers and literals are excluded.',
                              'reviewer': 'Codex GPT-6', 'human_confirmed': 'not recorded'})
    write_csv(output / 'language_audit.csv', language_rows)
    with (output / 'failure_analysis_sample.csv').open(newline='') as handle:
        spanish = list(csv.DictReader(handle))
    distributions = []
    for arm, rows in [('spanish', spanish), ('english-control', baseline_rows[:10]), ('standard', baseline_rows[10:])]:
        distributions.append({'configuration': arm, 'n': len(rows),
                              'requirement_dropped': sum(r['failure_mode'] == 'requirement dropped' for r in rows),
                              'wrong_implementation': sum(r['failure_mode'] == 'wrong implementation of retained description' for r in rows),
                              'other_primary': sum(r['failure_mode'] not in ['requirement dropped', 'wrong implementation of retained description'] for r in rows)})
    write_csv(output / 'failure_mode_comparison.csv', distributions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--show', choices=['english-control', 'standard', 'language'])
    parser.add_argument('--export-audits', action='store_true')
    parser.add_argument('--cycle-tests', action='store_true',
                        help='Also rebuild the secondary successful-cycle Wilcoxon tests; requires SciPy')
    args = parser.parse_args()
    config = load(ROOT / 'assignment4_config.json')
    ids = set(load(ROOT / 'protocol/task_order.json')['principal_ids'])
    if args.cycle_tests:
        from scipy.stats import wilcoxon
        tests = {}
        for model in config['models']:
            arms = [{t['task_id']: t['successful_cycles'] for t in
                     load(ROOT / 'results' / f"{model['model_id']}__{arm}.json")['tasks']}
                    for arm in ['spanish', 'english-control']]
            differences = [arms[0][i] - arms[1][i] for i in sorted(ids)]
            result = wilcoxon(differences, zero_method='pratt', alternative='two-sided')
            tests[model['model_id']] = {'statistic': float(result.statistic),
                                       'pvalue': float(result.pvalue)}
        (ROOT / 'analysis_assignment4/cycle_count_tests.json').write_text(json.dumps(tests, indent=2) + '\n')
    if args.export_audits:
        if args.data_dir is None:
            parser.error('--data-dir is needed to verify the standard audit sample')
        export_audits(config, ids, args.data_dir)
    if args.show == 'language':
        for stratum, (model, task, cycle, text) in language_sample(config):
            print(json.dumps({'model': model, 'task': task, 'cycle': cycle,
                              'stratum': stratum, 'summary': text}, ensure_ascii=False))
    elif args.show:
        if args.show == 'standard' and args.data_dir is None:
            parser.error('--data-dir is needed to inspect standard-loop records')
        for model, task in baseline_sample(config, ids, args.data_dir, args.show):
            cycles = task['cycles']
            print(json.dumps({'model': model, 'task': task['task_id'],
                              'failed_cycle': task['successful_cycles'] + 1,
                              'original': task['initial_prompt'],
                              'last_summary': cycles[-2].get('summary_raw', cycles[-2].get('summary')),
                              'failed_code': cycles[-1]['code']}, ensure_ascii=False))
    else:
        output = ROOT / 'analysis_assignment4' / 'truncation_sensitivity.json'
        output.write_text(json.dumps(sensitivity(config), indent=2) + '\n')
        print(output)


if __name__ == '__main__':
    main()
