import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import colab_20gb_compare as compare


class ColabCompareTests(unittest.TestCase):
    def test_pool_stays_below_observed_cap(self):
        card = {'total_mib': 40960, 'used_mib': 700}
        self.assertEqual(compare.pool_size(card, 19000, 16000), 15800)
        env = compare.run_environment(card, 15800)
        fraction = f'{15800 / 40960:.8f}'
        self.assertEqual(env['XLA_CLIENT_MEM_FRACTION'], fraction)
        self.assertNotIn('XLA_PYTHON_CLIENT_MEM_FRACTION', env)
        self.assertEqual(env['L2D_BATCH'], '1')

    def test_reuse_requires_successful_guard_and_process(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            expected = {'archive_sha256': 'abc', 'topology_seed': 1,
                        'optimizer_seed': 2, 'minutes': 60, 'cap_mib': 19000}
            row = {'status': 'complete', 'sha256': 'abc', 'topology_seed': 1,
                   'optimizer_seed': 2, 'frequencies': 50, 'minutes': 60,
                   'microbatch_objective': True, 'best_feasible_loss': .1}
            guard = {'status': 'within_cap', 'process_exit_code': 0,
                     'peak_used_mib': 17000}
            (folder / 'meta.json').write_text(json.dumps(expected))
            (folder / 'run.json').write_text(json.dumps(row))
            guard_file = folder / 'memory_guard.json'
            guard_file.write_text(json.dumps(guard))
            self.assertIsNotNone(compare.reusable(folder, expected))
            guard['process_exit_code'] = 1
            guard_file.write_text(json.dumps(guard))
            self.assertIsNone(compare.reusable(folder, expected))
            guard['process_exit_code'] = 0
            guard['peak_used_mib'] = 19000
            guard_file.write_text(json.dumps(guard))
            self.assertIsNone(compare.reusable(folder, expected))

    def test_failed_guard_cannot_win_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = SimpleNamespace(out=out, seed_index=0, seed_file=out / 'seeds.csv',
                                   minutes=60, cap_mib=19000)
            methods = [{'key': 'champion', 'label': 'Champion',
                        'archive': {'sha256': 'abc'}}]
            row = {'status': 'complete', 'best_feasible_loss': .1}
            guard = {'status': 'memory_cap_exceeded', 'process_exit_code': -15,
                     'peak_used_mib': 19100}
            report = compare.write_report(args, methods, (1, 2),
                                          {'name': 'GPU'}, 15000,
                                          {'champion': (row, guard)})
            self.assertIsNone(report['best_on_this_single_topology'])
            self.assertIsNone(report['results'][0]['best_feasible_loss'])


if __name__ == '__main__':
    unittest.main()
