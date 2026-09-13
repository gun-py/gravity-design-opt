import numpy as np
import jax
import jax.numpy as jnp
import optax


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
    def step(x, st, g, nz):
        g = cnorm(g, clip_grad)
        u, st2 = opt.update(g, st, x)
        un = jnp.linalg.norm(u, axis=-1, keepdims=True)
        nn = jnp.linalg.norm(nz, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, cap_noise_frac * un / (nn + 1e-12))
        return optax.apply_updates(x, u + nz * scale), st2

    return step, opt


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


def insure_incumbent(obj, tr):
    # single-point re-eval so the incumbent's feasibility gets its own history entry
    x = tr.anchor()
    if x is None or obj.budget_exceeded: return
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


def run_to_wire(obj, iter_fn, tr, insure_every=200):
    while not obj.budget_exceeded:
        try: iter_fn(tr.iter)
        except Exception:
            try: insure_incumbent(obj, tr)
            except Exception: pass
            continue
        if tr.iter % insure_every == 0 and tr.iter > 0:
            insure_incumbent(obj, tr)
    insure_incumbent(obj, tr)
