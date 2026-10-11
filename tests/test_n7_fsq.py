import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'l2d'))

from fsq import Basin, FSQ, Oracle, bfgs, clip_b, memory_error, power, solve_qp
from relay import Track


class Circle:
    bounds = np.array([[0.,0.,2.],[1.,1.,2.]])
    n_params = 3
    power_thresholds = {'hard':1.,'soft':1.,'detector':1.}

    def __init__(self, max_time=100):
        self.max_time = max_time
        self.time_elapsed = 0.
        self.loss_history = []
        self.is_feasible_history = []
        self.budget_exceeded = False

    def value_function_aux(self, unbounded=False):
        return self.raw

    def raw(self, x):
        a, b = x[:2]
        vals = {'hard':jnp.array([a*a+b*b]),
                'soft':jnp.array([a/1.2]),
                'detector':jnp.array([b/1.2])}
        c = jnp.array([vals['hard'][0]-1, vals['soft'][0]-1, vals['detector'][0]-1])
        f = -a + .1*b
        score = f + jnp.sum(jnp.maximum(c,0)/(1+jnp.maximum(c,0)))
        aux = {'sensitivity_loss':f,'power_values':vals,'is_feasible':jnp.all(c<=0)}
        return score, aux

    def log_evaluation(self, x, score, aux=None):
        self.time_elapsed += .01
        if self.time_elapsed <= self.max_time:
            self.loss_history.append(float(score))
            self.is_feasible_history.append(bool(aux['is_feasible']))
        else:
            self.budget_exceeded = True

    def value_aux(self, x):
        score, aux = self.raw(x)
        self.log_evaluation(x, score, aux)
        return score, aux


def fsq(obj):
    r = FSQ.__new__(FSQ)
    r.obj = obj
    r.oracle = Oracle(obj)
    r.lo = obj.bounds[0]
    r.hi = obj.bounds[1]
    r.span = r.hi-r.lo
    r.safe_span = np.where(r.span>0,r.span,1)
    r.live = r.span>0
    r.total = obj.max_time
    r.archive = []
    r.inc = None
    r.tracks = []
    r.accepted = []
    r.restore_seconds = 0.
    r.latencies = []
    return r


class N7Tests(unittest.TestCase):
    def test_coordinates_and_constraint_order(self):
        obj = Circle()
        o = Oracle(obj)
        u = np.array([.6,.8])
        np.testing.assert_allclose(o.to_x(u),[.6,.8,2.])
        np.testing.assert_allclose(o.from_x(o.to_x(u)),u)
        e = o.trial(u)
        self.assertEqual(len(e.c),3)
        np.testing.assert_allclose(e.c,[0,-.5,-1/3],atol=1e-12)
        self.assertTrue(e.admitted)

    def test_vjp_derivatives(self):
        o = Oracle(Circle())
        u = np.array([.6,.75])
        e, g, j = o.model(u,np.array([0,2],np.int32))
        self.assertTrue(e.admitted)
        self.assertAlmostEqual(o.obj.loss_history[-1],e.score)
        np.testing.assert_allclose(g,[-1,.1],atol=1e-10)
        np.testing.assert_allclose(j,[[1.2,1.5],[0,1/1.2]],atol=1e-10)
        h = 1e-5
        for k in range(2):
            v = np.eye(2)[k]*h
            plus, minus = o.trial(u+v),o.trial(u-v)
            self.assertAlmostEqual(g[k],(plus.f-minus.f)/(2*h),places=7)
            np.testing.assert_allclose(j[:,k],(plus.c[[0,2]]-minus.c[[0,2]])/(2*h),atol=1e-7)

    def test_derivative_chunks_keep_all_rows(self):
        class Many(Circle):
            def raw(self, x):
                score, aux = super().raw(x)
                vals = dict(aux['power_values'])
                vals['hard'] = x[0]*jnp.arange(1,13,dtype=jnp.float64)/20
                aux['power_values'] = vals
                return score, aux
        o = Oracle(Many())
        obs, g, jac = o.model(np.array([.4,.5]),np.arange(14))
        self.assertEqual(jac.shape,(14,2))
        self.assertTrue(np.all(np.isfinite(jac)))
        self.assertAlmostEqual(jac[11,0],12/20)

    def test_bfgs_and_qp(self):
        b = bfgs(np.eye(2),np.array([1.,0.]),np.array([.01,1.]))
        self.assertGreater(np.linalg.eigvalsh(b).min(),0)
        self.assertGreater(np.linalg.eigvalsh(clip_b(np.diag([-1.,2.]))).min(),0)
        solved = solve_qp(np.eye(2),np.array([-1.,0.]),np.eye(2),
                          np.array([-.5,-.5]),np.array([.5,.5]))
        self.assertIsNotNone(solved)
        np.testing.assert_allclose(solved[0],[.5,0],atol=1e-5)
        self.assertIsNone(solve_qp(np.eye(1),np.zeros(1),np.eye(1),
                                   np.array([1.]),np.array([0.])))

    def test_restore_and_budget(self):
        obj = Circle()
        r = fsq(obj)
        before = len(obj.loss_history)
        raw, h, _ = r.oracle.restore_model(np.array([.9,.9]))
        self.assertEqual(len(obj.loss_history),before+1)
        self.assertAlmostEqual(obj.loss_history[-1],raw.score)
        self.assertNotAlmostEqual(obj.loss_history[-1],h)
        e = r.restore(np.array([.9,.9]),30)
        self.assertIsNotNone(e)
        self.assertTrue(e.feasible)
        self.assertIsNotNone(r.inc)
        short = Circle(max_time=.005)
        r2 = fsq(short)
        e2 = r2.oracle.trial(np.array([.6,.7]))
        self.assertFalse(e2.admitted)
        self.assertIsNone(r2.restore(np.array([.9,.9]),30))

    def test_curved_boundary_and_incumbent(self):
        obj = Circle()
        r = fsq(obj)
        u = np.array([.6,.8-1e-6])
        first = r.oracle.trial(u)
        self.assertTrue(first.feasible)
        r.keep(r.oracle.to_x(u),first.score,True)
        basin = Basin(u,2,3)
        basin.score = first.score
        for _ in range(8):
            if not r.refine_once(basin,30):
                break
        self.assertTrue(r.accepted)
        self.assertTrue(all(feasible for _,feasible in r.accepted))
        losses = [first.score]+[loss for loss,_ in r.accepted]
        self.assertTrue(all(b<a for a,b in zip(losses,losses[1:])))
        self.assertLess(r.inc.loss,first.score)
        old = r.inc.loss
        basin.radius = 1e-6
        r.refine_once(basin,30)
        self.assertLessEqual(r.inc.loss,old)

    def test_second_order_correction(self):
        r = fsq(Circle())
        u = np.array([.6,.8-1e-6])
        old = r.oracle.trial(u)
        d = np.array([.05,-.0375])
        trial = r.oracle.trial(u+d)
        self.assertFalse(trial.feasible)
        fixed = r.correct(Basin(u,2,3),old,trial,d,np.array([-1.,.1]))
        self.assertIsNotNone(fixed)
        self.assertTrue(fixed[0].feasible)
        self.assertLess(fixed[0].score,old.score)

    def test_failure_and_memory(self):
        obj = Circle()
        r = fsq(obj)
        u = np.array([.5,.5])
        obs = r.oracle.trial(u)
        basin = Basin(u,2,3)
        old = r.sqp
        r.sqp = lambda *args: None
        try:
            r.refine_once(basin,30)
            self.assertTrue(np.all(np.isfinite(basin.u)))
        finally:
            r.sqp = old
        self.assertTrue(memory_error(RuntimeError('RESOURCE_EXHAUSTED')))
        self.assertFalse(memory_error(ValueError('bad shape')))

    def test_reinjection_resets_state_and_keeps_track_best(self):
        r = fsq(Circle())
        r.seed = 5
        r.next_id = 1
        t = Track(jnp.array([.2,.2,0.]),1,0,5,0,anchor=True)
        t.best_x = np.array([.9,.1,2.])
        t.best_loss = -.89
        t.m = jnp.ones(3)
        t.count = jnp.asarray(10)
        r.tracks = [t]
        obs = r.oracle.trial(np.array([.7,.2]))
        self.assertTrue(obs.feasible)
        r.anchor(obs)
        self.assertEqual(int(t.count),0)
        np.testing.assert_allclose(t.best_x,[.9,.1,2.])
        self.assertEqual(t.best_loss,-.89)


if __name__ == '__main__':
    unittest.main()
