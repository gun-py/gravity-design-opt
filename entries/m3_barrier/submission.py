import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from core import Tracker, cnorm, anneal, insure_incumbent, nan_guard, run_to_wire

_DEFAULT_TH = {"hard": 3.5e6, "soft": 2e3, "detector": 1e-2}


class BarrierGD(OptimizationAlgorithm):
    algorithm_str = "barriergd"

    def __init__(self, k=12):
        self.k = k

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.05, s0=0.15, anneal_frac=0.5, beta=1.0, barrier_eps=0.02,
                 clip_grad=1.0, polish_start=0.88, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        n, k = obj.n_params, self.k

        prior = Prior(find_prior_path())
        x = jnp.asarray(build_starts(obj, k=k, prior=prior, rng_seed=int(seed))
                        if init_params is None
                        else np.broadcast_to(np.asarray(init_params, np.float64), (k, n)).copy())

        lb, ub = jnp.asarray(obj.bounds)[0], jnp.asarray(obj.bounds)[1]
        th = getattr(obj, "power_thresholds", None) or _DEFAULT_TH
        th_h, th_s, th_d = float(th["hard"]), float(th["soft"]), float(th["detector"])

        obj.start_logging()

        f_aux = obj.value_function_aux(unbounded=True)
        if f_aux is None:
            self._fallback(obj, x, key, lr, s0, anneal_frac, clip_grad, insure_every)
            return

        def barrier(v, t):
            return jnp.mean(jax.nn.softplus((v / t - 1.0) / barrier_eps))

        def phi(xi):
            loss, aux = f_aux(xi)
            hard = aux["power_values"]["hard"].reshape(-1)
            soft = aux["power_values"]["soft"].reshape(-1)
            detn = aux["power_values"]["detector"].reshape(-1)
            surr = aux["sensitivity_loss"] + beta * (barrier(hard, th_h) + barrier(soft, th_s) + barrier(detn, th_d))
            return surr, (loss, aux)

        vg = jax.jit(jax.vmap(jax.value_and_grad(phi, has_aux=True)))
        _ = vg(x)  # timed warmup

        opt = optax.adam(lr); st = opt.init(x)

        @jax.jit
        def step(x, st, g, nz):
            g = cnorm(g, clip_grad)
            u, st2 = opt.update(g, st, x)
            un = jnp.linalg.norm(u, axis=-1, keepdims=True)
            nn = jnp.linalg.norm(nz, axis=-1, keepdims=True)
            scale = jnp.minimum(1.0, 0.25 * un / (nn + 1e-12))
            return optax.apply_updates(x, u + nz * scale), st2

        tr = Tracker()
        aux_keys = ("sensitivity_loss", "penalty", "is_feasible", "power_values", "violations")

        def loop(_):
            nonlocal x, st, key
            (_, (loss, aux)), grad = vg(x)
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            loss = jnp.where(jnp.isfinite(loss), loss, 1e10)
            xb = jax.vmap(lambda xi: lb + (ub - lb) * jax.nn.sigmoid(xi))(x)
            per_row = {kk: aux[kk] for kk in aux_keys}
            try:
                obj.log_evaluation(params=xb, loss=loss, aux=per_row)
            except Exception:
                j = int(np.argmin(np.asarray(loss)))
                sa = {}
                for kk, vv in per_row.items():
                    sa[kk] = ({k2: v2[j] for k2, v2 in vv.items()} if isinstance(vv, dict) else vv[j])
                obj.log_evaluation(params=xb[j], loss=loss[j], aux=sa)

            tr.update(np.asarray(x), np.asarray(loss), np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            nz = nz.at[0].set(0.0)
            x, st = step(x, st, grad, nz)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < polish_start:
            try: loop(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)

        try:
            a = tr.anchor()
            if a is not None and not obj.budget_exceeded:
                self._lbfgs(obj, a, f_aux, lb, ub, tr, aux_keys)
        except Exception:
            pass

        run_to_wire(obj, loop, tr, insure_every=50)

    def _lbfgs(self, obj, x0, f_aux, lb, ub, tr, aux_keys):
        opt = optax.lbfgs(); x = jnp.asarray(x0); st = opt.init(x)

        def loss_only(xi):
            l, _ = f_aux(xi); return l
        vg = jax.value_and_grad(loss_only)

        @jax.jit
        def step(x, st):
            l, g = vg(x)
            u, st2 = opt.update(g, st, x, value=l, grad=g, value_fn=loss_only)
            return optax.apply_updates(x, u), st2, l, g

        step(x, st)
        while not obj.budget_exceeded:
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

    def _fallback(self, obj, x, key, lr, s0, anneal_frac, clip_grad, insure_every):
        # no aux support (e.g. VoyagerProblem): plain multi-start Adam
        opt = optax.adam(lr); st = opt.init(x)
        obj.warmup_vmap_value_and_grad(batch_size=x.shape[0])
        tr = Tracker()

        @jax.jit
        def step(x, st, g, nz):
            g = cnorm(g, clip_grad)
            u, st2 = opt.update(g, st, x)
            un = jnp.linalg.norm(u, axis=-1, keepdims=True)
            nn = jnp.linalg.norm(nz, axis=-1, keepdims=True)
            scale = jnp.minimum(1.0, 0.25 * un / (nn + 1e-12))
            return optax.apply_updates(x, u + nz * scale), st2

        def loop(_):
            nonlocal x, st, key
            v, g = obj.vmap_value_and_grad(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.zeros(x.shape[0], bool))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            nz = nz.at[0].set(0.0)
            x, st = step(x, st, g, nz)

        run_to_wire(obj, loop, tr, insure_every=insure_every)
