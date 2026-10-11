import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXTRA_PATHS = [str(ROOT / 'entries/champion_v3'), str(ROOT / 'l2d'), str(ROOT / 'tools')]
sys.path[:0] = EXTRA_PATHS
from core import late_lr_scale, make_independent_adam_step, reset_adam_rows
from rescue import BasinRescue
from champion_experiment import discover, gain, select
from pack_champion_experiments import build, digest
for path in EXTRA_PATHS:
    sys.path.remove(path)


class ChampionV3Tests(unittest.TestCase):
    def test_corrected_control_keeps_v2_adam_steps_and_losses(self):
        spec = importlib.util.spec_from_file_location('v2_core_control_check',
                    ROOT / 'entries/champion_v2/core.py')
        v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2)
        old_step, old_opt = v2.make_independent_adam_step(.1, 1., .25)
        new_step, new_opt = make_independent_adam_step(.1, 1., .25)
        x_old = x_new = jnp.ones((3, 4)) * .3
        so, sn = old_opt.init(x_old), new_opt.init(x_new)
        for i in range(4):
            grad = jnp.ones_like(x_old) * (i + 1)
            noise = jnp.ones_like(x_old) * .05
            x_old, so = old_step(x_old, so, grad, noise)
            x_new, sn = new_step(x_new, sn, grad, noise, 1.0)
            np.testing.assert_array_equal(np.asarray(x_old), np.asarray(x_new))
            self.assertEqual(float(jnp.sum(x_old ** 2)), float(jnp.sum(x_new ** 2)))

    def test_rescue_replaces_only_stalled_nonanchors_and_resets_own_state(self):
        rescue = BasinRescue(2, 1000, 0, 7)
        points = np.array([[i / 2, i / 3] for i in range(18)], dtype=float)
        losses = np.arange(18, dtype=float) + 1
        feasible = np.ones(18, dtype=bool)
        feasible[1] = False
        aux = {'is_feasible': feasible}
        rescue.record(points, losses, aux, 10)
        unchanged, replaced = rescue.maybe_rescue(points, .39, 390)
        self.assertEqual(replaced, [])
        np.testing.assert_array_equal(unchanged, points)
        changed, replaced = rescue.maybe_rescue(points, .40, 400)
        self.assertEqual(replaced, [1, 11, 17])
        for anchor in (0, 6, 12):
            np.testing.assert_array_equal(changed[anchor], points[anchor])
        self.assertEqual(rescue.events[0]['violation_source'], 'penalized_loss')
        self.assertEqual(rescue.generation[1], 1)
        self.assertTrue(np.isinf(rescue.best_loss[11]))
        self.assertTrue(np.isfinite(rescue.basin_loss[1]))
        self.assertLessEqual(len(rescue.archive), 32)
        _, again = rescue.maybe_rescue(changed, .40, 410)
        self.assertEqual(again, [])
        step, opt = make_independent_adam_step(.1, 1., .25)
        x = jnp.zeros((18, 2))
        _, state = step(x, opt.init(x), jnp.ones_like(x), jnp.zeros_like(x))
        state = reset_adam_rows(state, replaced)
        counts = np.asarray(state[0].count)
        self.assertEqual([int(counts[i]) for i in replaced], [0, 0, 0])
        self.assertEqual([int(counts[i]) for i in (0, 6, 12)], [1, 1, 1])

    def test_power_violation_fallback_and_separate_rescue_rng(self):
        points = np.zeros((18, 2))
        points[:, 0] = np.arange(18)
        loss = np.arange(18, dtype=float) + 10
        feasible = np.zeros(18, dtype=bool)
        power = {name: np.ones((18, 1)) * 2 for name in ('hard', 'soft', 'detector')}
        power['hard'][3, 0] = .1
        for name in ('soft', 'detector'):
            power[name][3, 0] = .1
        r = BasinRescue(2, 1000, 0, 42, {'hard': 1, 'soft': 1, 'detector': 1})
        r.record(points, loss, {'is_feasible': feasible, 'power_values': power}, 10)
        r.maybe_rescue(points, .4, 400)
        self.assertEqual(r.events[0]['source'], 'least_violating')
        self.assertEqual(r.events[0]['violation_source'], 'power')
        np.testing.assert_allclose(r.basin_violation_x[0], points[3])

    def test_late_lr_boundaries_and_scaled_noise_cap(self):
        self.assertEqual(late_lr_scale(.60), 1)
        self.assertAlmostEqual(late_lr_scale(.775), .55)
        self.assertAlmostEqual(late_lr_scale(.95), .1)
        self.assertAlmostEqual(late_lr_scale(1), .1)
        step, opt = make_independent_adam_step(.1, 1., .25)
        x = jnp.zeros((1, 3))
        state = opt.init(x)
        g, noise = jnp.ones_like(x), jnp.ones_like(x) * 100
        full, sf = step(x, state, g, noise, 1.0)
        slow, ss = step(x, state, g, noise, .1)
        self.assertAlmostEqual(float(jnp.linalg.norm(slow)),
                               .1 * float(jnp.linalg.norm(full)), places=5)
        np.testing.assert_array_equal(np.asarray(sf[0].count), np.asarray(ss[0].count))

    def test_polish_logs_matching_unbounded_point_and_preserves_rejected_incumbent(self):
        class Objective:
            max_time = 1000.
            def __init__(self, admit=True):
                self.time_elapsed = 850.
                self.eval_count = 0
                self.admit = admit
                self.logged = []
                self.confirmations = 0
            @property
            def budget_exceeded(self):
                return self.time_elapsed >= self.max_time
            @property
            def budget_progress_fraction(self):
                return self.time_elapsed / self.max_time
            def value_function_aux(self, unbounded=True):
                assert unbounded
                def f(x):
                    loss = jnp.sum((x - 1.) ** 2)
                    return loss, {'is_feasible': jnp.asarray(True), 'penalty': loss,
                                  'sensitivity_loss': loss,
                                  'power_values': {'hard': jnp.ones((1,)),
                                                   'soft': jnp.ones((1,)),
                                                   'detector': jnp.ones((1,))},
                                  'violations': {'hard': jnp.zeros((1,))}}
                return f
            def log_evaluation(self, params, loss, grad, aux):
                p = np.asarray(params).copy()
                self.logged.append((p, float(loss), np.asarray(grad).copy(), bool(aux['is_feasible'])))
                self.time_elapsed += 20
                if self.admit:
                    self.eval_count += 1
            def value_aux(self, x):
                self.confirmations += 1
                self.time_elapsed += 1
                if self.admit:
                    self.eval_count += 1
                return self.value_function_aux()(x)
        from core import Tracker
        old_modules = {name: sys.modules.get(name) for name in ('warm', 'name_gen', 'submission')}
        sys.path[:0] = EXTRA_PATHS
        try:
            from submission import ChampionV3
            algo = ChampionV3()
        finally:
            for path in EXTRA_PATHS:
                sys.path.remove(path)
            for name, previous in old_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        algo.diagnostics = {'polish': {}}
        obj, tr = Objective(), Tracker()
        tr.best_feasible_loss = 9.
        tr.best_feasible_x = np.array([5., 5.])
        algo._polish_efficient(obj, np.array([3., 3.]), tr)
        self.assertGreaterEqual(len(obj.logged), 1)
        np.testing.assert_allclose(obj.logged[0][0], [3., 3.])
        for point, loss, grad, feasible in obj.logged:
            self.assertAlmostEqual(loss, float(np.sum((point - 1.) ** 2)), places=5)
            np.testing.assert_allclose(grad, 2 * (point - 1.), rtol=1e-5)
            self.assertTrue(feasible)
        self.assertLess(tr.best_feasible_loss, 9.)
        self.assertLessEqual(obj.confirmations, 1)
        control, control_tracker = Objective(), Tracker()
        control_tracker.best_feasible_loss = 9.
        algo._polish(control, np.array([3., 3.]), control_tracker)
        self.assertGreaterEqual(len(control.logged), 1)
        np.testing.assert_allclose(control.logged[0][0], [3., 3.])
        rejected, retained = Objective(admit=False), Tracker()
        retained.best_feasible_loss = 3.
        retained.best_feasible_x = np.array([5., 5.])
        algo._polish_efficient(rejected, np.array([3., 3.]), retained)
        self.assertEqual(retained.best_feasible_loss, 3.)
        np.testing.assert_array_equal(retained.best_feasible_x, [5., 5.])
        expired, unchanged = Objective(), Tracker()
        expired.time_elapsed = 980.
        unchanged.best_feasible_loss = 3.
        algo._polish_efficient(expired, np.array([3., 3.]), unchanged)
        self.assertEqual(expired.logged, [])
        self.assertEqual(expired.confirmations, 0)
        self.assertEqual(unchanged.best_feasible_loss, 3.)
        class Failed(Objective):
            def value_function_aux(self, unbounded=True):
                def fail(x):
                    raise RuntimeError('failed compilation')
                return fail
        failed, preserved = Failed(), Tracker()
        preserved.best_feasible_loss = 2.
        algo._polish_efficient(failed, np.array([3., 3.]), preserved)
        self.assertEqual(preserved.best_feasible_loss, 2.)
        self.assertEqual(failed.confirmations, 0)

    def test_stage_gates(self):
        def result(loss):
            return ({'status': 'complete', 'best_feasible_loss': loss},
                    {'status': 'within_cap', 'process_exit_code': 0})
        outcomes = {('discovery', 'control'): result(.50),
                    ('discovery', 'rescue'): result(.46),
                    ('discovery', 'late_lr'): result(.49),
                    ('discovery', 'polish'): result(.52)}
        self.assertEqual(discover(outcomes), ['rescue', 'late_lr'])
        outcomes.update({('cross_check', 'control'): result(.60),
                         ('cross_check', 'rescue'): result(.59),
                         ('cross_check', 'late_lr'): result(.58)})
        self.assertEqual(select(outcomes, ['rescue', 'late_lr']), 'rescue')
        outcomes[('cross_check', 'rescue')] = result(.61)
        self.assertIsNone(select(outcomes, ['rescue', 'late_lr']))
        self.assertIsNone(gain(outcomes, 'validation', 'rescue'))

    def test_archives_validate_without_touching_originals(self):
        originals = [ROOT / 'artifacts/baselines/local_champion.zip',
                     ROOT / 'dist/champion_v2.zip']
        old = [digest(path.read_bytes()) for path in originals]
        with tempfile.TemporaryDirectory() as directory:
            manifest = build(Path(directory))
            self.assertEqual(set(manifest['variants']), {'control', 'rescue', 'late_lr', 'polish'})
            self.assertEqual(len({row['sha256'] for row in manifest['variants'].values()}), 4)
        self.assertEqual(old, [digest(path.read_bytes()) for path in originals])


if __name__ == '__main__':
    unittest.main()
