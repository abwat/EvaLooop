"""Export selected task IDs and exact prompt templates without model access."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaloop.extensions.spanish_loop import (
    ARMS, GENERATION_PROMPT, STANDARD_SUMMARY_PROMPT, CONTROL_SUMMARY_PROMPT,
    verify_standard_prompt_hashes, wrap_summary,
)


def main():
    verify_standard_prompt_hashes()
    order = json.loads((ROOT / 'protocol/task_order.json').read_text())
    ids = order['principal_ids']
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError('Expected 100 unique principal task IDs')
    (ROOT / 'task_ids.json').write_text(json.dumps(ids, indent=2) + '\n')
    prompts = {
        'generation': GENERATION_PROMPT,
        'standard_summary': STANDARD_SUMMARY_PROMPT,
        'control_summary_template': CONTROL_SUMMARY_PROMPT,
        'english_control_summary': ARMS['english-control'].prompt('{code}'),
        'spanish_summary': ARMS['spanish'].prompt('{code}'),
        'next_cycle_wrapper': wrap_summary('{summary}', '{original_assert_line}'),
        'assertion_rule': 'Retain the last assert line from the original task prompt.',
    }
    (ROOT / 'protocol/prompts.json').write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + '\n')
    print('Exported task_ids.json and protocol/prompts.json')


if __name__ == '__main__':
    main()
