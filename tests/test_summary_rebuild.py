"""The performed-run summary needs only the archived JSON files and Python stdlib."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SummaryRebuildTest(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'results').is_dir(), 'requires the completed result bundle')
    def test_isolated_rebuild_and_incomplete_run_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for relative in ['scripts/analyze_assignment4.py',
                             'evaloop/extensions/assignment4_metrics.py',
                             'evaloop/extensions/__init__.py',
                             'assignment4_config.json', 'protocol/task_order.json']:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            shutil.copytree(ROOT / 'results', target / 'results')
            command = [sys.executable, '-S', 'scripts/analyze_assignment4.py', '--summary-only']
            completed = subprocess.run(command, cwd=target, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            actual = (target / 'analysis_assignment4/summary.csv').read_bytes()
            self.assertEqual(actual, (ROOT / 'analysis_assignment4/summary.csv').read_bytes())
            self.assertFalse((target / 'analysis_assignment4/baseline_summary.csv').exists())
            result_path = next((target / 'results').glob('*.json'))
            result = json.loads(result_path.read_text())
            result['tasks'].pop()
            result_path.write_text(json.dumps(result))
            failed = subprocess.run(command, cwd=target, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Incomplete', failed.stderr)


if __name__ == '__main__':
    unittest.main()
