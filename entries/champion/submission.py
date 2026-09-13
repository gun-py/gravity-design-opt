import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from core import Tracker, make_adam_step, anneal, insure_incumbent, nan_guard, run_to_wire

SCREEN, GRAD, BASINS, REPS = 36, 19, 3, 6
END_B, END_C = 0.25, 0.875


class Champion(OptimizationAlgorithm):
    def __init__(self):
        pass

    algorithm_str = "champion"

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, s0=0.3, cap_noise=0.25, anneal_frac=0.5, insure_every=200, **kw):
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

        # A: value-only screen
        try:
            v = np.where(np.isfinite(np.asarray(obj.vmap_value(jnp.asarray(cand)))),
                          np.asarray(obj.vmap_value(jnp.asarray(cand))), np.inf)
            x = jnp.asarray(cand[np.argsort(v)[:GRAD]])
        except Exception:
            x = jnp.asarray(cand[:GRAD])
        st = opt.init(x)

        # B: 19-wide capped noise
        def iter_b(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma; nz = nz.at[0].set(0.0)
            x, st = step(x, st, g, nz)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < END_B:
            try: iter_b(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)

        if obj.budget_exceeded:
            insure_incumbent(obj, tr); return

        # C: replicate 3 best basins
        try:
            v_l, _, aux_l = obj.vmap_value_and_grad_aux(x)
            vl, fl = np.asarray(v_l), np.asarray(aux_l["is_feasible"])
            score = np.where(fl, vl, vl + 10.0)
            basins = np.asarray(x)[np.argsort(score)[:BASINS]]

            xc = np.empty((BASINS * REPS, n), np.float64)
            for b in range(BASINS):
                key, sk = jax.random.split(key)
                noise = np.asarray(jax.random.normal(sk, (REPS, n))) * 0.2
                noise[0] = 0.0
                xc[b * REPS:(b + 1) * REPS] = basins[b] + noise
            x = jnp.asarray(xc); st = opt.init(x)
        except Exception:
            pass

        def iter_c(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            for b in range(BASINS):
                if b * REPS < x.shape[0]: nz = nz.at[b * REPS].set(0.0)
            x, st = step(x, st, g, nz)

            if tr.iter % 300 == 0 and tr.iter > 0 and tr.anchor() is not None:
                a = tr.anchor(); xn = np.asarray(x).copy(); vn = np.asarray(v)
                for b in range(BASINS):
                    sl = slice(b * REPS, min((b + 1) * REPS, xn.shape[0]))
                    if sl.stop - sl.start <= 1: continue
                    j = int(np.argmax(vn[sl][1:]) + 1)
                    key, sk = jax.random.split(key)
                    xn[sl.start + j] = a + np.asarray(jax.random.normal(sk, (n,))) * 0.2
                x = jnp.asarray(xn)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < END_C:
            try: iter_c(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)

        # D: L-BFGS polish on incumbent, then noise=0 descent to the wire
        try:
            a = tr.anchor()
            if a is not None and not obj.budget_exceeded:
                self._polish(obj, a, tr)
        except Exception:
            pass

        def iter_d(_):
            nonlocal x, st
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            x, st = step(x, st, g, jnp.zeros_like(x))

        run_to_wire(obj, iter_d, tr, insure_every=50)

    def _polish(self, obj, x0, tr):
        f_aux = obj.value_function_aux(unbounded=True)
        if f_aux is None: return

        def loss_only(xi):
            l, _ = f_aux(xi); return l
        vg = jax.value_and_grad(loss_only)
        opt = optax.lbfgs(); x = jnp.asarray(x0); st = opt.init(x)

        @jax.jit
        def step(x, st):
            l, g = vg(x)
            u, st2 = opt.update(g, st, x, value=l, grad=g, value_fn=loss_only)
            return optax.apply_updates(x, u), st2, l, g

        step(x, st)
        lb, ub = jnp.asarray(obj.bounds)[0], jnp.asarray(obj.bounds)[1]
        aux_keys = ("sensitivity_loss", "penalty", "is_feasible", "power_values", "violations")
        stop = min(0.985, float(obj.budget_progress_fraction) + 0.06)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < stop:
            xp = x; x, st, loss, g = step(x, st)
            try:
                _, aux = f_aux(xp)
                bx = lb + (ub - lb) * jax.nn.sigmoid(xp)
                obj.log_evaluation(params=bx, loss=loss, grad=g, aux={kk: aux[kk] for kk in aux_keys})
                if bool(aux["is_feasible"]) and float(loss) < tr.best_feasible_loss:
                    tr.best_feasible_loss = float(loss); tr.best_feasible_x = np.asarray(xp)
            except Exception:
                pass
            tr.iter += 1
