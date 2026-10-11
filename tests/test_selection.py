import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'l2d'),str(ROOT/'tools')]

from warm import _bounded_to_unbounded
from portfolio import Portfolio, Track
from bench import MicroObjective, score
from choose_submission import choose_screen, compare


class History:
    loss_history = [np.array([.1,-2,.5]),np.array([-.3,5.0])]
    is_feasible_history = [np.array([False,True,True]),np.array([True,False])]
    time_elapsed = 1.0
    eval_count = 5
    budget_exceeded = False


class Tiny:
    def __init__(self):
        self.calls = []
        self.time_elapsed = 0.0
        self.budget_exceeded = False
        self.n_params = 2
        self.max_time = 100
        self.time_left = 100

    @property
    def budget_progress_fraction(self):
        return .2 if len(self.calls) >= 3 else 0

    def vmap_value_and_grad_aux(self, z):
        self.calls.append(len(z))
        return np.arange(len(z),dtype=float), np.ones_like(z), {'is_feasible':np.ones(len(z),bool)}


class Broken:
    def __init__(self):
        self.calls = []

    def vmap_value(self, z):
        self.calls.append(len(z))
        return np.arange(len(z))

    def warmup_vmap_value(self, batch_size):
        self.calls.append(batch_size)

    def vmap_value_aux(self,z):
        self.calls.append(len(z))
        n = len(z)
        return np.arange(n), {'is_feasible':np.ones(n,bool)}


class SelectionTests(unittest.TestCase):
    def test_coordinates(self):
        lo = np.array([0.,2.,-1.])
        hi = np.array([1.,2.,3.])
        x = np.array([.2,2.,2.])
        z = _bounded_to_unbounded(x,lo,hi)
        got = lo + (hi-lo)/(1+np.exp(-z))
        np.testing.assert_allclose(got,x,rtol=0,atol=1e-12)

    def test_reset_and_incumbent(self):
        obj = Tiny()
        p = Portfolio.__new__(Portfolio)
        p.obj = obj
        p.bounds = np.array([[0.,0.],[1.,1.]])
        p.span = np.ones(2)
        p.live = np.ones(2,bool)
        p.inc_z = None
        p.inc_loss = np.inf
        p.archive = []
        t = Track(np.zeros(2),0,1,0)
        p.keep(t.z,-.2,True,t)
        self.assertEqual(p.inc_loss,-.2)
        t.m[:] = 2
        t.v[:] = 3
        t.count = 5
        t.reset(np.ones(2),1,1,5)
        self.assertEqual(t.count,0)
        self.assertTrue(np.all(t.m == 0) and np.all(t.v == 0))
        self.assertEqual(p.inc_loss,-.2)
        self.assertEqual(len(p.archive),1)

    def test_partial_batch_and_budget(self):
        obj = Tiny()
        p = Portfolio.__new__(Portfolio)
        p.obj = obj
        p.batch = 2
        p.drops = 0
        p.live = np.ones(2,bool)
        p.bounds = np.array([[0.,0.],[1.,1.]])
        p.span = np.ones(2)
        p.inc_z = None
        p.inc_loss = np.inf
        p.archive = []
        p.tracks = [Track(np.zeros(2),i,1,0) for i in range(5)]
        p.steps = 0
        p.mode = 'rescue'
        p.total = 100
        p.descend(.1)
        self.assertEqual(obj.calls,[2,2,1])
        self.assertEqual(p.steps,1)

    def test_memory_recovery(self):
        class Oom:
            def vmap_value_aux(self,z):
                raise RuntimeError('RESOURCE_EXHAUSTED: Out of memory')
        p = Portfolio.__new__(Portfolio)
        p.obj = Oom()
        p.batch = 2
        p.drops = 0
        with self.assertRaisesRegex(RuntimeError,'retry smaller chunk'):
            p.call(np.zeros((2,2)))
        self.assertEqual(p.batch,1)
        self.assertFalse(p.recover(RuntimeError('RESOURCE_EXHAUSTED')))

    def test_mixed_feasibility(self):
        data = score(History())
        self.assertEqual(data['best_feasible_loss'],-2.)
        self.assertEqual(data['feasible_candidate_count'],3)

    def test_microbatch(self):
        obj = Broken()
        proxy = MicroObjective(obj,2)
        proxy.warmup_vmap_value(batch_size=19)
        got = proxy.vmap_value(np.zeros((5,2)))
        self.assertEqual(obj.calls,[2,2,2,1])
        self.assertEqual(len(got),5)
        val, aux = proxy.vmap_value_aux(np.zeros((5,2)))
        self.assertEqual(obj.calls[-3:],[2,2,1])
        self.assertEqual(len(val),5)
        self.assertEqual(len(aux['is_feasible']),5)

    def test_microbatch_nested_aux(self):
        class Nested(Broken):
            def vmap_value_aux(self, z):
                n = len(z)
                return np.arange(n), {'is_feasible':np.ones(n,bool),
                                       'power_values': {'laser':np.ones((n,2))}}
        obj = MicroObjective(Nested(),2)
        loss, aux = obj.vmap_value_aux(np.zeros((5,2)))
        self.assertEqual(len(loss),5)
        self.assertEqual(aux['power_values']['laser'].shape,(5,2))

    def test_screen_once(self):
        class Screen(Tiny):
            def vmap_value_aux(self,z):
                self.calls.append(len(z))
                n = len(z)
                return np.arange(n,dtype=float), {'is_feasible':np.ones(n,bool)}
        obj = Screen()
        p = Portfolio.__new__(Portfolio)
        p.obj = obj
        p.batch = 2
        p.drops = 0
        p.starts = np.zeros((5,2))
        p.bounds = np.array([[0.,0.],[1.,1.]])
        p.span = np.ones(2)
        p.live = np.ones(2,bool)
        p.inc_z = None
        p.inc_loss = np.inf
        p.archive = []
        p.next_id = 0
        p.seed = 1
        p.screen()
        self.assertEqual(obj.calls,[2,2,1])
        self.assertEqual(len(p.tracks),5)

    def test_selection_gates(self):
        rows = {(n,i):{'status':'complete','best_feasible_loss':v}
                for n,vals in {'baseline':[.4,.5,.4], 'n5_rescue':[.3,.4,.39]}.items()
                for i,v in zip((2,7,9),vals)}
        self.assertTrue(compare(rows,'n5_rescue')['passed'])
        screen = {(n,i):{'status':'complete','best_feasible_loss':v}
                  for n,vals in {'champion_repair':[.2,.2],'n5_rescue':[.197,.198]}.items()
                  for i,v in zip((0,5),vals)}
        self.assertEqual(choose_screen(screen),'champion_repair')

    def test_polish_needs_scalar_confirmation(self):
        class Reject:
            time_elapsed = 0.0
            budget_exceeded = False
            time_left = 100.0
            max_time = 100.0

            def value_and_grad_aux(self,z):
                return float(-1), np.zeros(len(z)), {'is_feasible':True}

            def value_aux(self,z):
                return float(-1), {'is_feasible':False}

        p = Portfolio.__new__(Portfolio)
        p.obj = Reject()
        p.mode = 'rescue'
        p.total = 100.0
        p.bounds = np.array([[0.,0.],[1.,1.]])
        p.span = np.ones(2)
        p.live = np.ones(2,bool)
        p.archive = [(np.zeros(2),0.,0)]
        p.inc_z = np.zeros(2)
        p.inc_loss = 0.
        p.tracks = [Track(np.zeros(2),0,1,0.)]
        p.polish('test',1.)
        self.assertEqual(p.inc_loss,0.)
        self.assertEqual(p.tracks[0].ident,0)


if __name__ == '__main__':
    unittest.main()
