import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm


def cnorm(a, m):
    n = jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a * jnp.minimum(1.0, m / (n + 1e-12))


class Pack(OptimizationAlgorithm):
    algorithm_str = "pack"

    def __init__(self, k=8):
        self.k = k

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, s0=0.3, frac=0.5, cap=0.25, pat=300, jog=0.3, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        k, n = self.k, obj.n_params

        if init_params is None:
            x = obj.random_params_unbounded(n_samples=k)
        else:
            b = jnp.asarray(init_params)
            key, sk = jax.random.split(key)
            x = b + jax.random.normal(sk, (k, n)) * 0.1
            x = x.at[0].set(b)

        opt = optax.adam(lr)
        st = opt.init(x)

        @jax.jit
        def step(x, st, g, nz):
            u, st = opt.update(g, st, x)
            un = jnp.linalg.norm(u, axis=-1, keepdims=True)
            nn = jnp.linalg.norm(nz, axis=-1, keepdims=True)
            nz = nz * jnp.minimum(1.0, cap * un / (nn + 1e-12))
            return optax.apply_updates(x, u + nz), st

        obj.warmup_vmap_value_and_grad_aux(batch_size=k)
        obj.start_logging()

        bl = jnp.inf
        bx = x[0]
        bfl = jnp.inf
        bfx = None
        stall = 0
        i = 0

        while not obj.budget_exceeded:
            v, g, aux = obj.vmap_value_and_grad_aux(x)
            g = cnorm(g, 1.0)
            fe = aux["is_feasible"]

            j = int(jnp.argmin(v))
            if float(v[j]) < float(bl):
                bl = v[j]
                bx = x[j]
                stall = 0
            else:
                stall += 1

            vf = jnp.where(fe, v, jnp.inf)
            jf = int(jnp.argmin(vf))
            if float(vf[jf]) < float(bfl):
                bfl = vf[jf]
                bfx = x[jf]

            p = obj.budget_progress_fraction
            s = s0 * max(0.0, 1.0 - p / frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, (k, n)) * s
            nz = nz.at[0].set(0.0)
            x, st = step(x, st, g, nz)

            if pat and k > 1 and stall >= pat:
                sp = bfx if bfx is not None else bx
                key, sk = jax.random.split(key)
                w = (jnp.argsort(v[1:]) + 1)[-(k // 2):]
                x = x.at[w].set(sp + jax.random.normal(sk, (w.shape[0], n)) * jog)
                stall = 0

            i += 1
