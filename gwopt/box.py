import numpy as np
import jax
import jax.numpy as jnp
import optax


def get_bounds(obj):
    b = np.asarray(obj.bounds, dtype=np.float64)
    lo, hi = b[0], b[1]
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo, hi, span


def unit_to_physical(u, lo, span):
    return lo + u * span


def physical_to_unit(x, lo, span):
    return np.clip((x - lo) / span, 0.0, 1.0)


def logit_to_unit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z)))


def make_projected_step(lr):
    opt = optax.adam(lr)

    @jax.jit
    def step(u, state, grad_u, noise):
        grad_u = jnp.where(jnp.isfinite(grad_u), grad_u, 0.0)
        grad_norm = jnp.linalg.norm(grad_u, axis=-1, keepdims=True)
        grad_u = grad_u * jnp.minimum(1.0, 1.0 / (grad_norm + 1e-12))
        updates, new_state = opt.update(grad_u, state, u)
        return jnp.clip(u + updates + noise, 0.0, 1.0), new_state

    return step, opt
