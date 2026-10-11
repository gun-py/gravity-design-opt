import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from box import bounds_np, unb_to_u, make_proj_step
from core import Tracker, anneal, insure_incumbent, nan_guard
from fit import pick_batch

# n3 parameterisation (box-projected, params can reach bounds) + n1 budget schedule
# (successive halving) + n2 tail (augmented Lagrangian on the finalists).
SCREEN = 48
GB = 16
RUNGS = [(0.08, 24), (0.20, 12), (0.40, 6), (0.65, 3)]
ALM_FROM = 0.80
TH = {"hard": 3.5e6, "soft": 2e3, "detector": 1e-2}


class Fusion(OptimizationAlgorithm):
    algorithm_str = "fusion"

    def __init__(self):
        pass

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.02, s0=0.06, anneal_frac=0.55, rho0=1.0, rho_max=1e4,
                 dual_every=40, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)
        lo, hi, span = bounds_np(obj)
        n = obj.n_params
        rng = np.random.default_rng(int(seed))
        lo_j, span_j = jnp.asarray(lo), jnp.asarray(span)
        to_x = lambda uu: lo + np.asarray(uu) * span

        prior = Prior(find_prior_path())
        u_all = unb_to_u(build_starts(obj, k=min(SCREEN, 24), prior=prior, rng_seed=int(seed)))
        while u_all.shape[0] < SCREEN:
            src = u_all[rng.integers(0, min(8, u_all.shape[0]))]
            u_all = np.vstack([u_all, np.clip(src + rng.normal(size=n) * 0.08, 0, 1)])
        u_all = u_all[:SCREEN]

        step, opt = make_proj_step(lr)
        GB_ = pick_batch(obj, 'grad_aux', cap=GB)
        obj.warmup_vmap_value(batch_size=GB_)
        obj.start_logging()

        tr = Tracker()

        def sweep(u_np, grad=True):
            vs, gs, fes = [], [], []
            for i in range(0, u_np.shape[0], GB_):
                if obj.budget_exceeded: break
                ch = jnp.asarray(u_np[i:i + GB_]); x = lo_j + ch * span_j
                if grad:
                    v, gx, aux = obj.vmap_value_and_grad_aux(x)
                    v, gx = nan_guard(v, gx); gs.append(np.asarray(gx * span_j))
                else:
                    v = nan_guard(obj.vmap_value(x)); aux = None
                vs.append(np.asarray(v))
                fes.append(np.asarray(aux["is_feasible"]) if aux is not None
                           else np.zeros(ch.shape[0], bool))
            if not vs: return None, None, None
            return np.concatenate(vs), (np.concatenate(gs) if grad and gs else None), np.concatenate(fes)

        v, _, fe = sweep(u_all, grad=False)
        if v is not None:
            tr.update(u_all[:len(v)], v, fe)
            u_all = u_all[:len(v)][np.argsort(v)[:RUNGS[0][1]]]

        bpt = np.full(u_all.shape[0], np.inf)
        st = opt.init(jnp.asarray(u_all))

        for frac_end, keep in RUNGS:
            while not obj.budget_exceeded and float(obj.budget_progress_fraction) < frac_end:
                v, gu, fe = sweep(u_all)
                if v is None: break
                m = len(v)
                tr.update(u_all[:m], v, fe)
                bpt[:m] = np.minimum(bpt[:m], np.where(fe, v, v + 5.0))
                sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
                key, sk = jax.random.split(key)
                nz = np.asarray(jax.random.normal(sk, (m, n))) * sigma; nz[0] = 0.0
                uj, st = step(jnp.asarray(u_all[:m]), st, jnp.asarray(gu[:m]), jnp.asarray(nz))
                u_all = np.asarray(uj)
                if tr.iter % insure_every == 0 and tr.iter > 0:
                    insure_incumbent(obj, tr, to_x)
            if obj.budget_exceeded: break
            order = np.argsort(bpt[:u_all.shape[0]])[:keep]
            u_all, bpt = u_all[order], bpt[order]
            st = opt.init(jnp.asarray(u_all))

        # ALM tail on the finalists; falls back to plain projected descent if anything trips
        try:
            if not obj.budget_exceeded:
                self._alm(obj, u_all, key, lo_j, span_j, tr, to_x,
                          lr, rho0, rho_max, dual_every, insure_every)
        except Exception:
            pass

        while not obj.budget_exceeded:
            try:
                v, gu, fe = sweep(u_all)
                if v is None: break
                m = len(v)
                tr.update(u_all[:m], v, fe)
                uj, st = step(jnp.asarray(u_all[:m]), st, jnp.asarray(gu[:m]), jnp.zeros((m, n)))
                u_all = np.asarray(uj)
            except Exception:
                pass
            if tr.iter % 50 == 0: insure_incumbent(obj, tr, to_x)
        insure_incumbent(obj, tr, to_x)

    def _alm(self, obj, u_np, key, lo_j, span_j, tr, to_x,
             lr, rho0, rho_max, dual_every, insure_every):
        f_aux = obj.value_function_aux(unbounded=False)
        if f_aux is None: return
        th = getattr(obj, "power_thresholds", None) or TH
        t_h, t_s, t_d = float(th["hard"]), float(th["soft"]), float(th["detector"])

        def g_of(aux):
            pv = aux["power_values"]
            return jnp.concatenate([pv["hard"].reshape(-1) / t_h - 1.0,
                                    pv["soft"].reshape(-1) / t_s - 1.0,
                                    pv["detector"].reshape(-1) / t_d - 1.0])

        def L(u_i, lam, rho):
            loss, aux = f_aux(lo_j + u_i * span_j)
            g = g_of(aux)
            s = jnp.maximum(0.0, lam + rho * g)
            return aux["sensitivity_loss"] + jnp.sum(s ** 2 - lam ** 2) / (2.0 * rho), (loss, aux, g)

        vg = jax.jit(jax.vmap(jax.value_and_grad(L, has_aux=True), in_axes=(0, 0, None)))
        u = jnp.asarray(u_np)
        _, aux0 = f_aux(lo_j + u[0] * span_j)
        lam = jnp.zeros((u.shape[0], int(g_of(aux0).shape[0])))
        rho = float(rho0)
        _ = vg(u, lam, rho)

        opt = optax.adam(lr); st = opt.init(u)

        @jax.jit
        def step(u, st, gu):
            gu = jnp.where(jnp.isfinite(gu), gu, 0.0)
            gn = jnp.linalg.norm(gu, axis=-1, keepdims=True)
            gu = gu * jnp.minimum(1.0, 1.0 / (gn + 1e-12))
            upd, st2 = opt.update(gu, st, u)
            return jnp.clip(u + upd, 0.0, 1.0), st2

        akeys = ("sensitivity_loss", "penalty", "is_feasible", "power_values", "violations")
        prev = np.inf
        while not obj.budget_exceeded:
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
            u, st = step(u, st, gu)
            if tr.iter % dual_every == 0 and tr.iter > 0:
                lam = jnp.maximum(0.0, lam + rho * g)
                viol = float(jnp.max(jnp.maximum(0.0, g)))
                if viol > 0.75 * prev and rho < rho_max: rho *= 2.0
                prev = viol
            if tr.iter % insure_every == 0: insure_incumbent(obj, tr, to_x)
