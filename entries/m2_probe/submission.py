import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from core import Tracker, make_adam_step, anneal, insure_incumbent, nan_guard, run_to_wire

SCREEN, GRAD, BASINS, REPS = 36, 19, 3, 6
END_B, END_C, NOISE_END = 0.25, 0.875, 0.5


class ProbePortfolio(OptimizationAlgorithm):
    def __init__(self):
        pass

    algorithm_str = "probeportfolio"

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, s0=0.3, cap_noise=0.25, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        n = obj.n_params

        prior = Prior(find_prior_path())
        cand = np.asarray(build_starts(obj, k=SCREEN, prior=prior, rng_seed=int(seed)))

        step, opt = make_adam_step(lr, clip_grad=1.0, cap_noise_frac=cap_noise)

        obj.warmup_vmap_value(batch_size=SCREEN)
        obj.warmup_vmap_value_and_grad_aux(batch_size=GRAD)
        obj.warmup_vmap_value_and_grad_aux(batch_size=BASINS * REPS)
        obj.start_logging()

        tr = Tracker()

        # A: cheap 36-wide screen -> top 19
        v = np.where(np.isfinite(np.asarray(obj.vmap_value(jnp.asarray(cand)))),
                     np.asarray(obj.vmap_value(jnp.asarray(cand))), np.inf)
        x = jnp.asarray(cand[np.argsort(v)[:GRAD]])
        st = opt.init(x)

        def iter_b(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, NOISE_END)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            nz = nz.at[0].set(0.0)
            x, st = step(x, st, g, nz)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < END_B:
            try: iter_b(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)

        # C: pick 3 basins, replicate into 18
        if obj.budget_exceeded: return
        v_last, _, aux_last = obj.vmap_value_and_grad_aux(x)
        vln = np.asarray(v_last); fen = np.asarray(aux_last["is_feasible"])
        score = np.where(fen, vln, vln + 10.0)
        basins = np.asarray(x)[np.argsort(score)[:BASINS]]

        xc = np.empty((BASINS * REPS, n), dtype=np.float64)
        for b in range(BASINS):
            key, sk = jax.random.split(key)
            noise = np.asarray(jax.random.normal(sk, (REPS, n))) * 0.2
            noise[0] = 0.0
            xc[b * REPS:(b + 1) * REPS] = basins[b] + noise
        x = jnp.asarray(xc); st = opt.init(x)

        def iter_c(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, NOISE_END)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            for b in range(BASINS):
                nz = nz.at[b * REPS].set(0.0)
            x, st = step(x, st, g, nz)

            if tr.iter % 300 == 0 and tr.iter > 0 and tr.anchor() is not None:
                a = tr.anchor(); xn = np.asarray(x).copy(); vn = np.asarray(v)
                for b in range(BASINS):
                    sl = slice(b * REPS, (b + 1) * REPS)
                    j = int(np.argmax(vn[sl][1:]) + 1) if REPS > 1 else 0
                    key, sk = jax.random.split(key)
                    xn[sl.start + j] = a + np.asarray(jax.random.normal(sk, (n,))) * 0.2
                x = jnp.asarray(xn)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < END_C:
            try: iter_c(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)

        # D: refine, noise=0
        def iter_d(_):
            nonlocal x, st
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            x, st = step(x, st, g, jnp.zeros_like(x))

        run_to_wire(obj, iter_d, tr, insure_every=50)
