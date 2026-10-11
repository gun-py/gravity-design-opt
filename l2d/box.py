# normalized-box helpers: work in u in [0,1]^n, x = lb + u*(ub-lb).
# projected Adam in u-space lets params sit EXACTLY at bounds, where ~72% of elite
# design params live. sigmoid parameterisation kills those gradients (sigmoid' ~1e-9
# at |x|=20, and 26% of elite params need |x|>20), so the baselines can never get there.
import numpy as np
import jax
import jax.numpy as jnp
import optax


def bounds_np(obj):
    b = np.asarray(obj.bounds, dtype=np.float64)
    lo, hi = b[0], b[1]
    return lo, hi, np.where(hi > lo, hi - lo, 1.0)


def u_to_x(u, lo, span):
    return lo + u * span


def x_to_u(x, lo, span):
    return np.clip((x - lo) / span, 0.0, 1.0)


def unb_to_u(xu):
    # sigmoid-space vector -> normalised box coords
    return 1.0 / (1.0 + np.exp(-np.asarray(xu)))


def make_proj_step(lr):
    opt = optax.adam(lr)

    @jax.jit
    def step(u, st, gu, nz):
        gu = jnp.where(jnp.isfinite(gu), gu, 0.0)
        gn = jnp.linalg.norm(gu, axis=-1, keepdims=True)
        gu = gu * jnp.minimum(1.0, 1.0 / (gn + 1e-12))
        upd, st2 = opt.update(gu, st, u)
        return jnp.clip(u + upd + nz, 0.0, 1.0), st2

    return step, opt
