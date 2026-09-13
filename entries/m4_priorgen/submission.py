import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path, _canon_pairs, _prop_bounds_vector, _bounded_to_unbounded
from name_gen import signature
from core import Tracker, make_adam_step, anneal, nan_guard, run_to_wire

SCREEN, GRAD = 36, 12
_BT = ["L", "S", "D", "H"]


def _feat(topo, size):
    if size != 3: return np.zeros(9 + 12 * 4, np.float32)
    sig = signature(topo, size)
    ints = np.array(sig[:9], np.float32)
    bnd = np.zeros((12, 4), np.float32)
    for i, c in enumerate(sig[9:]): bnd[i, _BT.index(c)] = 1.0
    return np.concatenate([ints, bnd.reshape(-1)])


def _q_to_v(q, qt):
    return np.interp(np.clip(q, 0.0, 1.0), np.array([0., .05, .25, .5, .75, .95, 1.]), qt)


def _weights_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here, "weights.npz"),
              os.path.join(here, "..", "..", "entries", "m4_priorgen", "weights.npz")]:
        if os.path.isfile(c): return os.path.abspath(c)
    raise FileNotFoundError("weights.npz")


def _sample(weights, prior, nk, lo, hi, feat, n, tau, jitter, rng):
    dq, df, dl = weights["donor_q"], weights["donor_feat"], weights["donor_loss"]
    mu, sd = weights["global_mean_q"], weights["global_std_q"]
    if feat.shape[0] == df.shape[1]:
        dist = np.sum((df - feat[None, :]) ** 2, axis=1)
    else:
        dist = np.zeros(dq.shape[0])
    logits = -dist / max(tau, 1e-6) - dl / 0.05
    logits -= logits.max()
    w = np.exp(logits); w /= (w.sum() + 1e-12)

    out = np.empty((n, len(nk)), np.float64)
    for i in range(n):
        for j, (c, p) in enumerate(nk):
            kid = prior.key_to_id.get(f"{c}/{p}")
            if kid is None:
                out[i, j] = float(rng.uniform(lo[j], hi[j])); continue
            col = dq[:, kid]; m = np.isfinite(col)
            if m.sum() < 3:
                q = mu[kid] + rng.normal() * sd[kid]
            else:
                wm = w[m]; wm /= (wm.sum() + 1e-12)
                q = float(col[rng.choice(np.where(m)[0], p=wm)] + rng.normal() * jitter)
            out[i, j] = float(_q_to_v(np.array([np.clip(q, 0, 1)]), prior.per_key_quant[kid])[0])
    return np.clip(out, lo, hi)


class PriorGen(OptimizationAlgorithm):
    algorithm_str = "priorgen"

    def __init__(self, k=GRAD):
        self.k = k

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, s0=0.3, anneal_frac=0.5, cap_noise=0.25,
                 stall_reseed=250, reseed_jitter=0.30, tau=1.0, q_jitter=0.1,
                 insure_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        rng = np.random.default_rng(int(seed))
        n = obj.n_params

        prior = Prior(find_prior_path())
        nk = _canon_pairs(obj); lo, hi = _prop_bounds_vector(obj)
        donor_starts = build_starts(obj, k=SCREEN // 2, prior=prior, rng_seed=int(seed))

        # M4's own kernel-mixture proposals
        try:
            w = np.load(_weights_path(), allow_pickle=False)
            params = obj.problem_spec.get("params", obj.problem_spec)
            feat = _feat(params["topology"], int(params["size"])) if "topology" in params else np.zeros(9 + 12 * 4, np.float32)
            b_gen = _sample(w, prior, nk, lo, hi, feat, SCREEN // 2, tau, q_jitter, rng)
            gen = np.stack([_bounded_to_unbounded(b_gen[i], lo, hi) for i in range(b_gen.shape[0])])
        except Exception:
            gen = donor_starts.copy()[:SCREEN // 2]

        cand = np.concatenate([donor_starts, gen], axis=0)

        obj.warmup_vmap_value(batch_size=SCREEN)
        step, opt = make_adam_step(lr, clip_grad=1.0, cap_noise_frac=cap_noise)
        obj.warmup_vmap_value_and_grad_aux(batch_size=self.k)
        obj.start_logging()

        v = np.where(np.isfinite(np.asarray(obj.vmap_value(jnp.asarray(cand)))),
                     np.asarray(obj.vmap_value(jnp.asarray(cand))), np.inf)
        x = jnp.asarray(cand[np.argsort(v)[:self.k]])
        st = opt.init(x)

        tr = Tracker()

        def loop(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            vn = np.asarray(v); tr.update(np.asarray(x), vn, np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            nz = nz.at[0].set(0.0)
            x, st = step(x, st, g, nz)

            if self.k > 1 and tr.stall >= stall_reseed:
                a = tr.anchor()
                if a is not None:
                    worst = (np.argsort(vn[1:]) + 1)[-(self.k // 2):]
                    key, sk = jax.random.split(key)
                    j = np.asarray(jax.random.normal(sk, (len(worst), n))) * reseed_jitter
                    xn = np.asarray(x).copy(); xn[worst] = a[None, :] + j
                    x = jnp.asarray(xn)
                tr.stall = 0

        run_to_wire(obj, loop, tr, insure_every=insure_every)
