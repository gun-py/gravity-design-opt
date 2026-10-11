import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_champion_v2_core_test', ROOT / 'entries/champion_v2/core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

spec = importlib.util.spec_from_file_location(
    '_compare_final_test', ROOT / 'tools/compare_final.py')
import sys
sys.path.insert(0, str(ROOT / 'tools'))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ChampionV2Tests(unittest.TestCase):
    def test_independent_adam_resets_only_replaced_trajectory(self):
        step, optimizer = core.make_independent_adam_step(.1, 1., .25)
        points = jnp.zeros((2, 3))
        state = optimizer.init(points)
        points, state = step(points, state, jnp.ones_like(points), jnp.zeros_like(points))
        retained_moment = np.asarray(state[0].mu[0]).copy()
        state = core.reset_adam_rows(state, [1])
        np.testing.assert_array_equal(np.asarray(state[0].count), [1, 0])
        np.testing.assert_allclose(np.asarray(state[0].mu[0]), retained_moment)
        self.assertTrue(np.all(retained_moment > 0))
        np.testing.assert_allclose(np.asarray(state[0].mu[1]), [0., 0., 0.])
        _, state = step(points, state, jnp.ones_like(points), jnp.zeros_like(points))
        np.testing.assert_array_equal(np.asarray(state[0].count), [2, 1])

    def test_basins_use_personal_feasible_bests_and_separation(self):
        past = np.array([[3., 0.], [3., .01], [-3., 0.], [0., 3.]])
        losses = np.array([.1, .11, .3, .2])
        current = np.zeros_like(past)
        selected = core.select_basins(past, losses, current, np.ones(4), count=3)
        np.testing.assert_allclose(selected, past[[0, 3, 2]])

    def test_full_budget_paired_gate_and_incomplete_fallback(self):
        with tempfile.TemporaryDirectory() as dirname:
            out = Path(dirname)
            seeds = out / 'seeds.csv'
            seeds.write_text('topology_seed,optimizer_seed\n1,2\n3,4\n5,6\n')
            baseline = out / 'baseline.zip'
            candidate = out / 'candidate.zip'
            baseline.write_bytes(b'baseline')
            candidate.write_bytes(b'candidate')
            archives = {'baseline': baseline, 'candidate': candidate}
            hashes = {name: comparison.digest(path) for name, path in archives.items()}
            args = SimpleNamespace(indices=[0, 1, 2], minutes=240, seed_file=seeds,
                                   adapt_baseline=False, baseline_verified=False,
                                   candidate_features=None, out=out)
            pairs = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
            outcomes = {
                ('baseline', 0): {'status': 'complete', 'best_feasible_loss': .4},
                ('baseline', 1): {'status': 'complete', 'best_feasible_loss': .5},
                ('baseline', 2): {'status': 'complete', 'best_feasible_loss': .6},
                ('candidate', 0): {'status': 'complete', 'best_feasible_loss': .3},
                ('candidate', 1): {'status': 'complete', 'best_feasible_loss': .4},
                ('candidate', 2): {'status': 'complete', 'best_feasible_loss': .59},
            }
            result = comparison.decision(args, archives, hashes, pairs, outcomes, 'H100')
            self.assertTrue(result['promotion_gate_passed'])
            self.assertEqual((out / 'selected.zip').read_bytes(), b'candidate')
            args.minutes = 30
            self.assertFalse(comparison.decision(args, archives, hashes, pairs, outcomes, 'H100')['promotion_gate_passed'])
            self.assertFalse((out / 'selected.zip').exists())
            args.minutes = 240
            args.candidate_features = 'screen'
            self.assertFalse(comparison.decision(args, archives, hashes, pairs, outcomes, 'H100')['promotion_gate_passed'])
            args.candidate_features = None
            outcomes[('candidate', 1)] = {'status': 'error', 'best_feasible_loss': None}
            self.assertFalse(comparison.decision(args, archives, hashes, pairs, outcomes, 'H100')['complete'])


if __name__ == '__main__':
    unittest.main()
