import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from core import Tracker, nan_guard, run_to_wire

SCREEN, REFINE, KNN_K = 36, 12, 10


@partial(jax.jit, static_argnames=("k",))
def _knn(Xt, yt, Xq, k=10):
    d2 = jnp.sum((Xq[:, None, :] - Xt[None, :, :]) ** 2, axis=-1)
    _, top = jax.lax.top_k(-d2, k)
    td = jnp.take_along_axis(d2, top, axis=1)
    w = 1.0 / (jnp.sqrt(td) + 1e-8); w = w / jnp.sum(w, axis=1, keepdims=True)
    return jnp.sum(w * yt[top], axis=1)


class SurroSearch(OptimizationAlgorithm):
    algorithm_str = "surrosearch"

    def __init__(self, ref_size=400, cands=10000, refine_batch=REFINE):
        self.ref_size = ref_size; self.cands = cands; self.refine_batch = refine_batch

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, temperature=0.5, refine_steps=12, perturb=0.15,
                 elite_frac=0.5, insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)
        n = obj.n_params
        lb, ub = np.asarray(obj.bounds[0]), np.asarray(obj.bounds[1])
        span = ub - lb

        prior = Prior(find_prior_path())
        starts_u = build_starts(obj, k=SCREEN, prior=prior, rng_seed=int(seed))
        starts_b = np.clip(lb + span * (1.0 / (1.0 + np.exp(-starts_u))), lb, ub)

        obj.warmup_vmap_value(batch_size=SCREEN)
        obj.warmup_vmap_value_and_grad(batch_size=self.refine_batch)
        obj.start_logging()

        tr = Tracker()
        rx = starts_b.astype(np.float64)
        ry = np.where(np.isfinite(np.asarray(obj.vmap_value(jnp.asarray(rx)))),
                       np.asarray(obj.vmap_value(jnp.asarray(rx))), 1e10)
        tr.update(rx, ry, np.zeros(SCREEN, bool))

        rng = np.random.default_rng(int(seed))
        if rx.shape[0] < self.ref_size:
            extra = self.ref_size - rx.shape[0]
            unif = lb + span * rng.uniform(size=(extra, n))
            v = np.where(np.isfinite(np.asarray(obj.vmap_value(jnp.asarray(unif)))),
                          np.asarray(obj.vmap_value(jnp.asarray(unif))), 1e10)
            rx = np.concatenate([rx, unif]); ry = np.concatenate([ry, v])
            tr.update(unif, v, np.zeros(extra, bool))
        idx = np.argsort(ry)[:self.ref_size]; rx, ry = rx[idx], ry[idx]

        opt = optax.adam(lr)

        def loop(_):
            nonlocal rx, ry, key
            m = rx.mean(0, keepdims=True); s = rx.std(0, keepdims=True) + 1e-8

            n_e = int(self.cands * elite_frac); n_u = self.cands - n_e
            elite = np.clip(rx[rng.integers(0, rx.shape[0], size=n_e)]
                             + rng.normal(size=(n_e, n)) * (perturb * span), lb, ub)
            unif = lb + span * rng.uniform(size=(n_u, n))
            cand = np.concatenate([elite, unif], axis=0)

            Xt = (rx - m) / s; Xq = (cand - m) / s
            pred = np.asarray(_knn(jnp.asarray(Xt), jnp.asarray(ry), jnp.asarray(Xq), KNN_K))

            logits = -pred / max(temperature, 1e-6); logits -= logits.max()
            w = np.exp(logits); w /= (w.sum() + 1e-12)
            pick = rng.choice(cand.shape[0], size=self.refine_batch, replace=False, p=w)
            xb = jnp.asarray(cand[pick]); st = opt.init(xb)
            for _ in range(refine_steps):
                if obj.budget_exceeded: break
                v, g = obj.vmap_value_and_grad(xb); v, g = nan_guard(v, g)
                u, st = opt.update(g, st, xb)
                xb = jnp.clip(optax.apply_updates(xb, u), jnp.asarray(lb), jnp.asarray(ub))

            xn = np.asarray(xb)
            v_final = np.where(np.isfinite(np.asarray(obj.vmap_value(xb))),
                                np.asarray(obj.vmap_value(xb)), 1e10)
            tr.update(xn, v_final, np.zeros(v_final.shape, bool))

            rx = np.concatenate([rx, xn]); ry = np.concatenate([ry, v_final])
            k = np.argsort(ry)[:self.ref_size]; rx, ry = rx[k], ry[k]

        run_to_wire(obj, loop, tr, insure_every=insure_every)
