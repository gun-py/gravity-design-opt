import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'l2d'))

from relay import Point, Relay, Track, cross, groups, logit, step


class Small:
    def __init__(self):
        self.calls = []
        self.time_elapsed = 0.0
        self.max_time = 1000.0
        self.budget_exceeded = False

    @property
    def time_left(self):
        return self.max_time - self.time_elapsed

    @property
    def budget_progress_fraction(self):
        return .5 if len(self.calls) >= 3 else 0.0

    def vmap_value_and_grad_aux(self, x):
        self.calls.append(len(x))
        self.time_elapsed += .01
        return jnp.ones(len(x)), jnp.ones_like(x), {'is_feasible':jnp.zeros(len(x), bool)}

    def value_and_grad_aux(self, x):
        self.time_elapsed += .01
        loss = (x[0] - 1) ** 2 + (x[1] - .35) ** 2
        grad = jnp.array([2 * (x[0] - 1), 2 * (x[1] - .35)])
        return loss, grad, {'is_feasible':jnp.asarray(True)}

    def value_aux(self, x):
        self.time_elapsed += .01
        return (x[0] - 1) ** 2 + (x[1] - .35) ** 2, {'is_feasible':jnp.asarray(True)}


def relay(obj=None):
    r = Relay.__new__(Relay)
    r.obj = obj or Small()
    r.seed = 7
    r.lo = np.zeros(2)
    r.hi = np.ones(2)
    r.span = np.ones(2)
    r.safe_span = np.ones(2)
    r.live = np.ones(2, bool)
    r.lo_dev = jnp.zeros(2)
    r.span_dev = jnp.ones(2)
    r.live_dev = jnp.ones(2, bool)
    r.total = 1000.0
    r.batch = 2
    r.drops = 0
    r.next_id = 0
    r.steps = 0
    r.archive = []
    r.inc = None
    r.confirmed = None
    r.tracks = []
    r.latencies = []
    r.failures = 0
    return r


class RelayTests(unittest.TestCase):
    def test_coordinates_and_chain(self):
        u = np.array([1e-8, .2, .9])
        z = logit(u)
        np.testing.assert_allclose(np.asarray(jax.nn.sigmoid(z)), u, rtol=1e-6, atol=1e-9)
        span = jnp.array([2., 4., 3.])
        grad = jnp.array([1., -2., 3.])
        direct = jax.grad(lambda w: jnp.sum((span * jax.nn.sigmoid(w)) * grad))(jnp.asarray(z))
        np.testing.assert_allclose(np.asarray(direct), np.asarray(span * grad * u * (1-u)), rtol=1e-6, atol=1e-8)

    def test_component_blocks(self):
        blocks = groups([('ml11','reflectivity'),('ml11','tuning'),('ml11sus','mass'),
                         ('mr11_ml12','length'),('mr11_ml12','length')])
        self.assertEqual(blocks, [[0,1,2],[3,4]])
        a = np.zeros(5)
        b = np.ones(5)
        out = cross(a,b,blocks,np.random.default_rng(2))
        self.assertEqual(out[0],out[1])
        self.assertEqual(out[1],out[2])
        self.assertEqual(out[3],out[4])

    def test_device_step_and_reset(self):
        p = jnp.array([[0., 0.],[1., .5]])
        zeros = jnp.zeros_like(p)
        count = jnp.zeros(2,jnp.int32)
        keys = jnp.stack([jax.random.PRNGKey(1),jax.random.PRNGKey(2)])
        out = step(p,zeros,zeros,count,keys,jnp.array([0,1]),jnp.array([True,True]),
                   jnp.array([[1.,1.],[-1.,1.]]),jnp.ones(2),jnp.ones(2,bool),0.)
        self.assertEqual(np.asarray(out[3]).tolist(),[1,1])
        self.assertTrue(np.all(np.asarray(out[0][1]) >= 0))
        t = Track(out[0][0],0,0,7,0)
        t.m, t.v, t.count = out[1][0],out[2][0],out[3][0]
        r = relay()
        r.replace(t,np.array([.5,.5]),kind=1)
        self.assertEqual(int(t.count),0)
        self.assertEqual(t.kind,1)
        self.assertTrue(np.all(np.asarray(t.m) == 0))

    def test_partial_batches_and_budget(self):
        r = relay()
        r.tracks = [r.new(np.array([.5,.5])) for _ in range(5)]
        r.descend(.1)
        self.assertEqual(r.obj.calls,[2,2,1])
        self.assertEqual(r.steps,1)

    def test_last_batch_keeps_feasible_point(self):
        class Last(Small):
            def vmap_value_and_grad_aux(self, x):
                self.calls.append(len(x))
                if len(self.calls) == 3:
                    self.budget_exceeded = True
                    return jnp.array([-1.]), jnp.ones_like(x), {'is_feasible':jnp.array([True])}
                return jnp.ones(len(x)), jnp.ones_like(x), {'is_feasible':jnp.zeros(len(x), bool)}
        r = relay(Last())
        r.tracks = [r.new(np.array([.5,.5])) for _ in range(5)]
        r.descend(.1)
        self.assertEqual(r.obj.calls,[2,2,1])
        self.assertEqual(r.inc.loss,-1.)

    def test_repeated_bad_gradient_restarts_track(self):
        class Bad(Small):
            def vmap_value_and_grad_aux(self, x):
                self.calls.append(len(x))
                return jnp.ones(len(x)), jnp.full_like(x, jnp.nan), {'is_feasible':jnp.zeros(len(x), bool)}
        r = relay(Bad())
        r.starts = np.array([[.5,.5]])
        r.tracks = [r.new(np.array([.5,.5]))]
        r.descend(.1)
        self.assertEqual(r.obj.calls,[1,1,1])
        self.assertEqual(r.failures,1)
        self.assertEqual(int(r.tracks[0].count),0)

    def test_incumbent_survives_replacement(self):
        r = relay()
        t = r.new(np.array([.5,.5]))
        r.tracks = [t]
        r.keep(np.array([.5,.5]),.2,True,t)
        r.replace(t,np.array([.1,.1]))
        self.assertAlmostEqual(r.inc.loss,.2)
        np.testing.assert_allclose(r.inc.x,[.5,.5])

    def test_mixed_feasibility_and_memory(self):
        r = relay()
        class Power:
            power_thresholds = {'hard':1.0}
        r.obj = Power()
        aux = {'power_values':{'hard':jnp.array([[.5,.9],[1.2,.7]])}}
        np.testing.assert_allclose(r.violation(aux,2),[0,.2])
        r.batch = 4
        self.assertTrue(r.recover(RuntimeError('RESOURCE_EXHAUSTED')))
        self.assertEqual(r.batch,2)
        self.assertTrue(r.recover(RuntimeError('RESOURCE_EXHAUSTED')))
        self.assertFalse(r.recover(RuntimeError('RESOURCE_EXHAUSTED')))

    def test_polish_reinjects_bounded_anchor(self):
        r = relay()
        x = np.array([.9,.3])
        old = .0125
        r.archive = [Point(x,old)]
        r.inc = Point(x,old)
        r.tracks = [r.new(x,anchor=True)]
        r.polish(.35)
        self.assertEqual(r.tracks[0].kind,1)
        self.assertTrue(r.tracks[0].anchor)
        self.assertLess(r.inc.loss,old)
        np.testing.assert_allclose(r.tracks[0].best_x,r.inc.x,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
