import numpy as np
import jax
import jax.numpy as jnp
import optax


def clip_norm(arr, max_norm):
    norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
    return arr * jnp.minimum(1.0, max_norm / (norm + 1e-12))


def linear_anneal(progress, start_val, end_val, ramp_frac):
    if ramp_frac <= 0:
        return float(end_val)
    t = min(1.0, max(0.0, progress / ramp_frac))
    return float(start_val + (end_val - start_val) * t)


def cosine_anneal(progress, start_val, end_val, ramp_frac):
    if ramp_frac <= 0:
        return float(end_val)
    t = min(1.0, max(0.0, progress / ramp_frac))
    return float(end_val + (start_val - end_val) * 0.5 * (1 + np.cos(np.pi * t)))


def make_adam_step(lr, grad_clip=1.0, noise_cap=0.25):
    opt = optax.adam(lr)

    @jax.jit
    def step(params, state, grads, noise):
        grads = clip_norm(grads, grad_clip)
        updates, new_state = opt.update(grads, state, params)
        update_norm = jnp.linalg.norm(updates, axis=-1, keepdims=True)
        noise_norm = jnp.linalg.norm(noise, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, noise_cap * update_norm / (noise_norm + 1e-12))
        return optax.apply_updates(params, updates + noise * scale), new_state

    return step, opt


class Tracker:
    def __init__(self):
        self.best_loss = float("inf")
        self.best_x = None
        self.best_feasible_loss = float("inf")
        self.best_feasible_x = None
        self.iter = 0
        self.stall = 0
        self.last_improve_iter = -1

    def update(self, points, losses, feasible):
        losses = np.asarray(losses)
        feasible = np.asarray(feasible)
        best_idx = int(np.argmin(losses))
        improved = float(losses[best_idx]) < self.best_loss
        if improved:
            self.best_loss = float(losses[best_idx])
            self.best_x = np.array(points[best_idx])
            self.last_improve_iter = self.iter
            self.stall = 0
        else:
            self.stall += 1
        feasible_losses = np.where(feasible.astype(bool), losses, np.inf)
        best_feas_idx = int(np.argmin(feasible_losses))
        if float(feasible_losses[best_feas_idx]) < self.best_feasible_loss:
            self.best_feasible_loss = float(feasible_losses[best_feas_idx])
            self.best_feasible_x = np.array(points[best_feas_idx])
        self.iter += 1
        return improved

    def anchor(self):
        if self.best_feasible_x is not None:
            return self.best_feasible_x
        return self.best_x


def confirm_best(obj, tracker, to_x=None):
    x = tracker.anchor()
    if x is None or obj.budget_exceeded:
        return
    if to_x is not None:
        try:
            x = to_x(x)
        except Exception:
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


def shrink_batch(obj, k):
    left = getattr(obj, "evals_left", None)
    return max(1, min(k, int(left))) if left is not None else k


def clean_nans(values, grads=None):
    safe_vals = jnp.where(jnp.isfinite(values), values, 1e10)
    if grads is None:
        return safe_vals
    safe_grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
    return safe_vals, safe_grads


def set_penalty(obj, name):
    if name not in {"squashed_relu_penalty", "relu_penalty", "zero_penalty"}:
        return False
    try:
        import dfbench.problems as dp
        penalties = {
            "squashed_relu_penalty": dp.squashed_relu_penalty,
            "relu_penalty": dp.relu_penalty,
            "zero_penalty": dp.zero_penalty,
        }
        obj.set_penalty_fn(penalties[name])
        return True
    except Exception:
        return False


def run_until_done(obj, step_fn, tracker, confirm_every=200, to_x=None):
    while not obj.budget_exceeded:
        try:
            step_fn(tracker.iter)
        except Exception:
            try:
                confirm_best(obj, tracker, to_x)
            except Exception:
                pass
            continue
        if tracker.iter % confirm_every == 0 and tracker.iter > 0:
            confirm_best(obj, tracker, to_x)
    confirm_best(obj, tracker, to_x)
