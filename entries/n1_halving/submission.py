import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from box import bounds_np, unb_to_u, make_proj_step
from core import Tracker, anneal, insure_incumbent, nan_guard
from fit import pick_batch

# successive halving: explore many starts cheaply, spend the tail of the budget on the few
# survivors. uniform allocation (what every baseline does) wastes ~90% of compute on
# trajectories that were already losing at 10% of budget.
SCREEN = 64
GBATCH = 16
RUNGS = [(0.06, 32), (0.15, 16), (0.32, 8), (0.58, 4), (1.01, 2)]


class Halving(OptimizationAlgorithm):
    algorithm_str = "halving"

    def __init__(self):
        pass

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.02, s0=0.06, anneal_frac=0.55, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)
        lo, hi, span = bounds_np(obj)
        n = obj.n_params
        rng = np.random.default_rng(int(seed))

        prior = Prior(find_prior_path())
        base = build_starts(obj, k=min(SCREEN, 24), prior=prior, rng_seed=int(seed))
        u_all = unb_to_u(base)
        # pad out to SCREEN with jittered copies of the good ones
        while u_all.shape[0] < SCREEN:
            src = u_all[rng.integers(0, min(8, u_all.shape[0]))]
            u_all = np.vstack([u_all, np.clip(src + rng.normal(size=n) * 0.08, 0, 1)])
        u_all = u_all[:SCREEN]

        step, opt = make_proj_step(lr)
        GBATCH_ = pick_batch(obj, 'grad_aux', cap=GBATCH)
        obj.warmup_vmap_value(batch_size=GBATCH_)
        obj.start_logging()

        tr = Tracker()
        lo_j, span_j = jnp.asarray(lo), jnp.asarray(span)
        to_x = lambda uu: lo + np.asarray(uu) * span

        def eval_chunks(u_np, grad=True):
            """run the whole population through vmap in <=GBATCH chunks"""
            vs, gs, fes = [], [], []
            for i in range(0, u_np.shape[0], GBATCH_):
                if obj.budget_exceeded: break
                ch = jnp.asarray(u_np[i:i + GBATCH_])
                x = lo_j + ch * span_j
                if grad:
                    v, gx, aux = obj.vmap_value_and_grad_aux(x)
                    v, gx = nan_guard(v, gx)
                    gs.append(np.asarray(gx * span_j))
                else:
                    v = obj.vmap_value(x); aux = None
                    v = nan_guard(v)
                vs.append(np.asarray(v))
                fes.append(np.asarray(aux["is_feasible"]) if aux is not None
                           else np.zeros(ch.shape[0], bool))
            if not vs: return None, None, None
            return (np.concatenate(vs),
                    np.concatenate(gs) if grad and gs else None,
                    np.concatenate(fes))

        # cheap screen -> keep best 32
        v, _, fe = eval_chunks(u_all, grad=False)
        if v is not None:
            tr.update(u_all[:len(v)], v, fe)
            u_all = u_all[:len(v)][np.argsort(v)[:RUNGS[0][1]]]

        best_per_traj = np.full(u_all.shape[0], np.inf)
        st = opt.init(jnp.asarray(u_all))

        for frac_end, keep in RUNGS:
            while not obj.budget_exceeded and float(obj.budget_progress_fraction) < frac_end:
                v, gu, fe = eval_chunks(u_all, grad=True)
                if v is None: break
                m = len(v)
                tr.update(u_all[:m], v, fe)
                best_per_traj[:m] = np.minimum(best_per_traj[:m], np.where(fe, v, v + 5.0))

                sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
                key, sk = jax.random.split(key)
                nz = np.asarray(jax.random.normal(sk, (m, n))) * sigma
                nz[0] = 0.0
                uj, st = step(jnp.asarray(u_all[:m]), st, jnp.asarray(gu[:m]), jnp.asarray(nz))
                u_all = np.asarray(uj)

                if tr.iter % insure_every == 0 and tr.iter > 0:
                    insure_incumbent(obj, tr, to_x)

            if obj.budget_exceeded: break
            # prune to the survivors and restart the optimiser state for the smaller batch
            order = np.argsort(best_per_traj[:u_all.shape[0]])[:keep]
            u_all = u_all[order]
            best_per_traj = best_per_traj[order]
            st = opt.init(jnp.asarray(u_all))

        # spend anything left on the finalists, no noise
        while not obj.budget_exceeded:
            try:
                v, gu, fe = eval_chunks(u_all, grad=True)
                if v is None: break
                m = len(v)
                tr.update(u_all[:m], v, fe)
                uj, st = step(jnp.asarray(u_all[:m]), st, jnp.asarray(gu[:m]),
                              jnp.zeros((m, n)))
                u_all = np.asarray(uj)
            except Exception:
                pass
            if tr.iter % 50 == 0: insure_incumbent(obj, tr, to_x)
        insure_incumbent(obj, tr, to_x)
