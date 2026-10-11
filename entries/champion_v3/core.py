import numpy as np
import jax
import jax.numpy as jnp
import optax


def late_lr_scale(progress):
    """Keep the search rate until 60%, then decay it tenfold by 95%."""
    p = min(1.0, max(0.0, float(progress)))
    if p <= 0.60:
        return 1.0
    if p >= 0.95:
        return 0.1
    t = (p - 0.60) / 0.35
    return 0.1 + 0.45 * (1.0 + np.cos(np.pi * t))


def cnorm(a, m):
    n = jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a * jnp.minimum(1.0, m / (n + 1e-12))


def anneal(p, s0, s1, frac):
    if frac <= 0: return float(s1)
    t = min(1.0, max(0.0, p / frac))
    return float(s0 + (s1 - s0) * t)


def make_adam_step(lr, clip_grad, cap_noise_frac):
    opt = optax.adam(lr)

    @jax.jit
    def step(x, st, g, nz, lr_scale=1.0):
        g = cnorm(g, clip_grad)
        u, st2 = opt.update(g, st, x)
        u = u * lr_scale
        un = jnp.linalg.norm(u, axis=-1, keepdims=True)
        nn = jnp.linalg.norm(nz, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, cap_noise_frac * un / (nn + 1e-12))
        return optax.apply_updates(x, u + nz * scale), st2

    return step, opt


def make_independent_adam_step(lr, clip_grad, cap_noise_frac):
    single = optax.adam(lr)

    class BatchAdam:
        def init(self, x):
            return jax.vmap(single.init)(x)

    @jax.jit
    def step(x, st, g, nz, lr_scale=1.0):
        g = cnorm(g, clip_grad)
        updates, st2 = jax.vmap(single.update)(g, st, x)
        updates = updates * lr_scale
        update_norm = jnp.linalg.norm(updates, axis=-1, keepdims=True)
        noise_norm = jnp.linalg.norm(nz, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, cap_noise_frac * update_norm / (noise_norm + 1e-12))
        return optax.apply_updates(x, updates + nz * scale), st2

    return step, BatchAdam()


def reset_adam_rows(state, indices):
    indices = jnp.asarray(indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda leaf: leaf.at[indices].set(0), state)


def select_basins(feasible_x, feasible_loss, fallback_x, fallback_loss, count=3, separation=0.03):
    pool = []
    for points, losses in ((feasible_x, feasible_loss), (fallback_x, fallback_loss)):
        for index in np.argsort(losses, kind='stable'):
            if np.isfinite(losses[index]):
                pool.append((np.asarray(points[index]), float(losses[index])))
    if not pool:
        return np.asarray(fallback_x)[:count].copy()
    selected = []
    for point, _ in pool:
        u = np.asarray(jax.nn.sigmoid(point))
        if all(np.sqrt(np.mean((u - np.asarray(jax.nn.sigmoid(other))) ** 2)) >= separation
               for other in selected):
            selected.append(point)
            if len(selected) == count:
                break
    if len(selected) < count:
        for point, _ in pool:
            if not any(np.array_equal(point, other) for other in selected):
                selected.append(point)
                if len(selected) == count:
                    break
    while len(selected) < count:
        selected.append(pool[len(selected) % len(pool)][0])
    return np.asarray(selected)


class Tracker:
    def __init__(self):
        self.best_loss = float("inf"); self.best_x = None
        self.best_feasible_loss = float("inf"); self.best_feasible_x = None
        self.iter = 0; self.stall = 0; self.last_improve_iter = -1

    def update(self, xb, losses, fez):
        losses = np.asarray(losses); fez = np.asarray(fez)
        j = int(np.argmin(losses))
        improved = float(losses[j]) < self.best_loss
        if improved:
            self.best_loss = float(losses[j]); self.best_x = np.array(xb[j])
            self.last_improve_iter = self.iter; self.stall = 0
        else:
            self.stall += 1
        vf = np.where(fez.astype(bool), losses, np.inf)
        jf = int(np.argmin(vf))
        if float(vf[jf]) < self.best_feasible_loss:
            self.best_feasible_loss = float(vf[jf]); self.best_feasible_x = np.array(xb[jf])
        self.iter += 1
        return improved

    def anchor(self):
        return self.best_feasible_x if self.best_feasible_x is not None else self.best_x


def insure_incumbent(obj, tr, to_x=None):
    # single-point re-eval so the incumbent's feasibility gets its own history entry.
    # to_x maps the tracker's coords into whatever space the Objective expects.
    x = tr.anchor()
    if x is None or obj.budget_exceeded: return
    if to_x is not None:
        try: x = to_x(x)
        except Exception: return
    try:
        f = getattr(obj, "value_aux", None)
        if f is not None:
            loss, aux = f(jnp.asarray(x))
            if bool(aux["is_feasible"]) and float(loss) < tr.best_feasible_loss:
                tr.best_feasible_loss = float(loss); tr.best_feasible_x = np.array(x)
        else:
            obj.value(jnp.asarray(x))
    except Exception:
        pass


def shrink_batch(obj, k):
    left = getattr(obj, "evals_left", None)
    return max(1, min(k, int(left))) if left is not None else k


def nan_guard(v, g=None):
    v2 = jnp.where(jnp.isfinite(v), v, 1e10)
    return v2 if g is None else (v2, jnp.where(jnp.isfinite(g), g, 0.0))


def set_penalty(obj, name):
    if name not in {"squashed_relu_penalty", "relu_penalty", "zero_penalty"}: return False
    try:
        import dfbench.problems as dp
        obj.set_penalty_fn({"squashed_relu_penalty": dp.squashed_relu_penalty,
                            "relu_penalty": dp.relu_penalty,
                            "zero_penalty": dp.zero_penalty}[name])
        return True
    except Exception:
        return False


def run_to_wire(obj, iter_fn, tr, insure_every=200, to_x=None):
    while not obj.budget_exceeded:
        try: iter_fn(tr.iter)
        except Exception:
            try: insure_incumbent(obj, tr, to_x)
            except Exception: pass
            continue
        if tr.iter % insure_every == 0 and tr.iter > 0:
            insure_incumbent(obj, tr, to_x)
    insure_incumbent(obj, tr, to_x)
