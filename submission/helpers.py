import re

import numpy as np
import jax
import jax.numpy as jnp
import optax


def clip_norm(arr, max_norm):
    norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
    return arr * jnp.minimum(1.0, max_norm / (norm + 1e-12))


def cosine_anneal(progress, start_val, end_val, ramp_frac):
    if ramp_frac <= 0:
        return float(end_val)
    t = min(1.0, max(0.0, progress / ramp_frac))
    return float(end_val + (start_val - end_val) * 0.5 * (1 + np.cos(np.pi * t)))


def make_adam_step(lr, grad_clip=1.0, noise_cap=0.25):
    single = optax.adam(lr)

    class BatchAdam:
        def init(self, x):
            return jax.vmap(single.init)(x)

    @jax.jit
    def step(params, state, grads, noise):
        grads = clip_norm(grads, grad_clip)
        updates, new_state = jax.vmap(single.update)(grads, state, params)
        update_norm = jnp.linalg.norm(updates, axis=-1, keepdims=True)
        noise_norm = jnp.linalg.norm(noise, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, noise_cap * update_norm / (noise_norm + 1e-12))
        return optax.apply_updates(params, updates + noise * scale), new_state

    return step, BatchAdam()


def reset_adam_rows(state, indices):
    indices = jnp.asarray(indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda leaf: leaf.at[indices].set(0), state)


def param_groups(pairs):
    out = {}
    for i, (name, _) in enumerate(pairs):
        key = re.sub(r'sus$', '', name)
        out.setdefault(key, []).append(i)
    return list(out.values())


def crossover(a, b, blocks, rng):
    x = a.copy()
    take = rng.choice(len(blocks), max(1, len(blocks) // 4), replace=False)
    for j in take:
        x[blocks[j]] = b[blocks[j]]
    return x


def select_basins(feasible_x, feasible_loss, fallback_x, fallback_loss,
                  count=3, min_separation=0.03):
    pool = []
    for points, losses in ((feasible_x, feasible_loss),
                           (fallback_x, fallback_loss)):
        for idx in np.argsort(losses, kind='stable'):
            if np.isfinite(losses[idx]):
                pool.append((np.asarray(points[idx]), float(losses[idx])))
    if not pool:
        return np.asarray(fallback_x)[:count].copy()
    selected = []
    for point, _ in pool:
        u = np.asarray(jax.nn.sigmoid(point))
        if all(np.sqrt(np.mean((u - np.asarray(jax.nn.sigmoid(other))) ** 2))
               >= min_separation for other in selected):
            selected.append(point)
            if len(selected) == count:
                break
    if len(selected) < count:
        for point, _ in pool:
            if not any(np.array_equal(point, s) for s in selected):
                selected.append(point)
                if len(selected) == count:
                    break
    while len(selected) < count:
        selected.append(pool[len(selected) % len(pool)][0])
    return np.asarray(selected)


class Tracker:
    def __init__(self):
        self.best_loss = float("inf")
        self.best_x = None
        self.best_feasible_loss = float("inf")
        self.best_feasible_x = None
        self.iter = 0
        self.stall = 0
        self.archive = []

    def update(self, points, losses, feasible):
        losses = np.asarray(losses)
        feasible = np.asarray(feasible)
        best_idx = int(np.argmin(losses))
        if float(losses[best_idx]) < self.best_loss:
            self.best_loss = float(losses[best_idx])
            self.best_x = np.array(points[best_idx])
            self.stall = 0
        else:
            self.stall += 1
        feasible_losses = np.where(feasible.astype(bool), losses, np.inf)
        best_feas = int(np.argmin(feasible_losses))
        if float(feasible_losses[best_feas]) < self.best_feasible_loss:
            self.best_feasible_loss = float(feasible_losses[best_feas])
            self.best_feasible_x = np.array(points[best_feas])
        for i in range(len(losses)):
            if feasible[i] and np.isfinite(losses[i]):
                self._archive_update(np.array(points[i]), float(losses[i]))
        self.iter += 1

    def _archive_update(self, x, loss):
        u = np.asarray(jax.nn.sigmoid(x))
        for j, (ax, al) in enumerate(self.archive):
            au = np.asarray(jax.nn.sigmoid(ax))
            if np.sqrt(np.mean((u - au) ** 2)) < 1e-3:
                if loss < al:
                    self.archive[j] = (x.copy(), loss)
                return
        self.archive.append((x.copy(), loss))
        self.archive.sort(key=lambda a: a[1])
        del self.archive[24:]

    def anchor(self):
        if self.best_feasible_x is not None:
            return self.best_feasible_x
        return self.best_x


def confirm_best(obj, tracker):
    x = tracker.anchor()
    if x is None or obj.budget_exceeded:
        return
    try:
        fn = getattr(obj, "value_aux", None)
        if fn is not None:
            loss, aux = fn(jnp.asarray(x))
            if bool(aux["is_feasible"]) and float(loss) < tracker.best_feasible_loss:
                tracker.best_feasible_loss = float(loss)
                tracker.best_feasible_x = np.array(x)
        else:
            obj.value(jnp.asarray(x))
    except Exception:
        pass


def clean_nans(values, grads=None):
    safe_vals = jnp.where(jnp.isfinite(values), values, 1e10)
    if grads is None:
        return safe_vals
    return safe_vals, jnp.where(jnp.isfinite(grads), grads, 0.0)
