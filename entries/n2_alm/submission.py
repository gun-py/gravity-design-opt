import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from box import bounds_np, unb_to_u
from core import Tracker, insure_incumbent
from fit import pick_batch

# Augmented Lagrangian (PHR). score = sensitivity_loss subject to power<=threshold, so
# optimise exactly that instead of a fixed penalised surrogate. multipliers let us sit ON
# the constraint boundary, which is where the good designs are -- a fixed penalty either
# pushes you off it or saturates (the default squashed_relu is gradient-dead once violated).
K = 12
TH = {"hard": 3.5e6, "soft": 2e3, "detector": 1e-2}


class ALM(OptimizationAlgorithm):
    algorithm_str = "alm"

    def __init__(self, k=K):
        self.k = k

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.02, rho0=1.0, rho_max=1e4, dual_every=40, s0=0.05,
                 anneal_frac=0.5, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)
        lo, hi, span = bounds_np(obj)
        n = obj.n_params

        prior = Prior(find_prior_path())
        u = unb_to_u(build_starts(obj, k=self.k, prior=prior, rng_seed=int(seed)))
        u = jnp.asarray(u)

        th = getattr(obj, "power_thresholds", None) or TH
        t_h, t_s, t_d = float(th["hard"]), float(th["soft"]), float(th["detector"])

        self.k = pick_batch(obj, 'grad_aux', cap=self.k)
        u = u[:self.k]
        obj.start_logging()
        f_aux = obj.value_function_aux(unbounded=False)
        if f_aux is None:
            self._plain(obj, u, key, lo, span, lr, insure_every)
            return

        lo_j, span_j = jnp.asarray(lo), jnp.asarray(span)

        def g_of(aux):
            pv = aux["power_values"]
            return jnp.concatenate([pv["hard"].reshape(-1) / t_h - 1.0,
                                    pv["soft"].reshape(-1) / t_s - 1.0,
                                    pv["detector"].reshape(-1) / t_d - 1.0])

        def L(u_i, lam, rho):
            x = lo_j + u_i * span_j
            loss, aux = f_aux(x)
            g = g_of(aux)
            s = jnp.maximum(0.0, lam + rho * g)
            pen = jnp.sum(s ** 2 - lam ** 2) / (2.0 * rho)
            return aux["sensitivity_loss"] + pen, (loss, aux, g)

        vg = jax.jit(jax.vmap(jax.value_and_grad(L, has_aux=True), in_axes=(0, 0, None)))

        # probe once to size the multiplier vector
        _, aux0 = f_aux(lo_j + u[0] * span_j)
        m = int(g_of(aux0).shape[0])
        lam = jnp.zeros((self.k, m))
        rho = float(rho0)
        _ = vg(u, lam, rho)          # timed compile

        opt = optax.adam(lr); st = opt.init(u)

        @jax.jit
        def step(u, st, gu, nz):
            gu = jnp.where(jnp.isfinite(gu), gu, 0.0)
            gn = jnp.linalg.norm(gu, axis=-1, keepdims=True)
            gu = gu * jnp.minimum(1.0, 1.0 / (gn + 1e-12))
            upd, st2 = opt.update(gu, st, u)
            return jnp.clip(u + upd + nz, 0.0, 1.0), st2

        tr = Tracker()
        akeys = ("sensitivity_loss", "penalty", "is_feasible", "power_values", "violations")
        to_x = lambda uu: lo + np.asarray(uu) * span
        prev_viol = np.inf

        while not obj.budget_exceeded:
            try:
                (_, (loss, aux, g)), gu = vg(u, lam, rho)
                x = lo_j + u * span_j
                try:
                    obj.log_evaluation(params=x, loss=loss, aux={k: aux[k] for k in akeys})
                except Exception:
                    j = int(np.argmin(np.asarray(loss)))
                    sa = {k: ({k2: v2[j] for k2, v2 in aux[k].items()}
                              if isinstance(aux[k], dict) else aux[k][j]) for k in akeys}
                    obj.log_evaluation(params=x[j], loss=loss[j], aux=sa)

                tr.update(np.asarray(u), np.asarray(loss), np.asarray(aux["is_feasible"]))

                p = float(obj.budget_progress_fraction)
                sigma = max(0.0, s0 * (1.0 - min(1.0, p / anneal_frac)))
                key, sk = jax.random.split(key)
                nz = jax.random.normal(sk, u.shape) * sigma
                nz = nz.at[0].set(0.0)
                u, st = step(u, st, gu, nz)

                if tr.iter % dual_every == 0 and tr.iter > 0:
                    lam = jnp.maximum(0.0, lam + rho * g)          # dual ascent
                    viol = float(jnp.max(jnp.maximum(0.0, g)))
                    if viol > 0.75 * prev_viol and rho < rho_max:
                        rho *= 2.0                                  # not converging -> tighten
                    prev_viol = viol
            except Exception:
                pass
            if tr.iter % insure_every == 0 and tr.iter > 0:
                insure_incumbent(obj, tr, to_x)
        insure_incumbent(obj, tr, to_x)

    def _plain(self, obj, u, key, lo, span, lr, insure_every):
        from box import make_proj_step
        from core import nan_guard, run_to_wire
        step, opt = make_proj_step(lr); st = opt.init(u)
        obj.warmup_vmap_value_and_grad(batch_size=u.shape[0])
        tr = Tracker()
        lo_j, span_j = jnp.asarray(lo), jnp.asarray(span)

        def loop(_):
            nonlocal u, st
            v, gx = obj.vmap_value_and_grad(lo_j + u * span_j)
            v, gx = nan_guard(v, gx)
            tr.update(np.asarray(u), np.asarray(v), np.zeros(u.shape[0], bool))
            u, st = step(u, st, gx * span_j, jnp.zeros_like(u))

        run_to_wire(obj, loop, tr, insure_every=insure_every,
                    to_x=lambda uu: lo + np.asarray(uu) * span)
