import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from core import Tracker, make_adam_step, anneal, nan_guard, run_to_wire


class WarmPack(OptimizationAlgorithm):
    algorithm_str = "warmpack"

    def __init__(self, k=12):
        self.k = k

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, s0=0.3, anneal_frac=0.5, cap_noise=0.25,
                 stall_reseed=250, reseed_jitter=0.30, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        prior = Prior(find_prior_path())
        if init_params is None:
            x = build_starts(obj, k=self.k, prior=prior, rng_seed=int(seed))
        else:
            b = np.asarray(init_params, dtype=np.float64)
            rng = np.random.default_rng(int(seed))
            x = np.broadcast_to(b, (self.k, obj.n_params)).copy() + rng.normal(size=(self.k, obj.n_params)) * 0.1
            x[0] = b
        x = jnp.asarray(x)

        step, opt = make_adam_step(lr, clip_grad=1.0, cap_noise_frac=cap_noise)
        st = opt.init(x)

        obj.warmup_vmap_value_and_grad_aux(batch_size=self.k)
        obj.start_logging()

        tr = Tracker()

        def loop(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x)
            v, g = nan_guard(v, g)
            fe = np.asarray(aux["is_feasible"]); vn = np.asarray(v)
            tr.update(np.asarray(x), vn, fe)

            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            nz = nz.at[0].set(0.0)  # row 0 protected
            x, st = step(x, st, g, nz)

            if self.k > 1 and tr.stall >= stall_reseed:
                a = tr.anchor()
                if a is not None:
                    worst = (np.argsort(vn[1:]) + 1)[-(self.k // 2):]
                    key, sk = jax.random.split(key)
                    j = np.asarray(jax.random.normal(sk, (len(worst), obj.n_params))) * reseed_jitter
                    xn = np.asarray(x).copy(); xn[worst] = a[None, :] + j
                    x = jnp.asarray(xn)
                tr.stall = 0

        run_to_wire(obj, loop, tr, insure_every=insure_every)
