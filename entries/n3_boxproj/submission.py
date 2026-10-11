import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from box import bounds_np, u_to_x, unb_to_u, make_proj_step
from core import Tracker, anneal, insure_incumbent, nan_guard, run_to_wire
from fit import pick_batch

K = 16


class BoxProj(OptimizationAlgorithm):
    algorithm_str = "boxproj"

    def __init__(self, k=K):
        self.k = k

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.02, s0=0.06, anneal_frac=0.55, stall_reseed=250,
                 reseed_jitter=0.08, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)
        lo, hi, span = bounds_np(obj)
        n = obj.n_params

        prior = Prior(find_prior_path())
        if init_params is None:
            u = unb_to_u(build_starts(obj, k=self.k, prior=prior, rng_seed=int(seed)))
        else:
            rng = np.random.default_rng(int(seed))
            u0 = np.clip((np.asarray(init_params, np.float64) - lo) / span, 0, 1)
            u = np.clip(u0 + rng.normal(size=(self.k, n)) * 0.05, 0, 1)
            u[0] = u0
        u = jnp.asarray(u)

        step, opt = make_proj_step(lr)
        st = opt.init(u)

        self.k = pick_batch(obj, 'grad_aux', cap=self.k)
        u = u[:self.k]; st = opt.init(u)
        obj.start_logging()

        tr = Tracker()
        span_j, lo_j = jnp.asarray(span), jnp.asarray(lo)

        def loop(_):
            nonlocal u, st, key
            x = lo_j + u * span_j
            v, gx, aux = obj.vmap_value_and_grad_aux(x)
            v, gx = nan_guard(v, gx)
            gu = gx * span_j                       # chain rule; no sigmoid factor to kill it
            vn = np.asarray(v)
            tr.update(np.asarray(u), vn, np.asarray(aux["is_feasible"]))

            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, u.shape) * sigma
            nz = nz.at[0].set(0.0)
            u, st = step(u, st, gu, nz)

            if self.k > 1 and tr.stall >= stall_reseed:
                a = tr.anchor()
                if a is not None:
                    worst = (np.argsort(vn[1:]) + 1)[-(self.k // 2):]
                    key, sk = jax.random.split(key)
                    j = np.asarray(jax.random.normal(sk, (len(worst), n))) * reseed_jitter
                    un = np.asarray(u).copy()
                    un[worst] = np.clip(a[None, :] + j, 0.0, 1.0)
                    u = jnp.asarray(un)
                tr.stall = 0

        run_to_wire(obj, loop, tr, insure_every=insure_every,
                    to_x=lambda uu: lo + np.asarray(uu) * span)
