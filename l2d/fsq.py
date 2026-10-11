import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

try:
    import osqp
except ImportError:
    osqp = None

from relay import Relay


def power(aux, limits):
    return jnp.concatenate([
        jnp.ravel(aux['power_values'][name]) / limits[name] - 1
        for name in ('hard', 'soft', 'detector')
    ])


def clip_b(b, lower=1e-6, upper=1e6):
    b = (b + b.T) / 2
    if not np.all(np.isfinite(b)):
        return np.eye(len(b))
    vals, vecs = np.linalg.eigh(b)
    return (vecs * np.clip(vals, lower, upper)) @ vecs.T


def bfgs(b, s, y):
    bs = b @ s
    sb = float(s @ bs)
    sy = float(s @ y)
    if not np.isfinite(sb + sy) or sb <= 1e-14:
        return b
    theta = 1.0 if sy >= .2 * sb else .8 * sb / (sb - sy)
    r = theta * y + (1 - theta) * bs
    sr = float(s @ r)
    if not np.isfinite(sr) or sr <= 1e-14:
        return b
    return clip_b(b - np.outer(bs, bs) / sb + np.outer(r, r) / sr)


def memory_error(exc):
    msg = str(exc).lower()
    return any(s in msg for s in ('resource_exhausted', 'out of memory',
                                  'cuda_error_out_of_memory', 'allocator-allocation-error'))


def solve_qp(p, q, a, lower, upper, warm=None):
    n = len(q)
    p = clip_b(np.asarray(p, np.float64), 1e-10, 1e8)
    q = np.asarray(q, np.float64)
    a = np.asarray(a, np.float64).reshape(-1, n)
    lower = np.asarray(lower, np.float64)
    upper = np.asarray(upper, np.float64)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.all(np.isfinite(a))):
        return None
    if np.any(lower > upper):
        return None
    if osqp is not None:
        try:
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(p), q=q, A=sparse.csc_matrix(a),
                         l=lower, u=upper, eps_abs=1e-8, eps_rel=1e-8,
                         max_iter=4000, polishing=True, time_limit=.5,
                         warm_starting=True, verbose=False)
            if warm is not None and len(warm) == n:
                solver.warm_start(x=warm)
            result = solver.solve(raise_error=False)
        except Exception as exc:
            if memory_error(exc):
                raise
            if type(exc).__name__ not in ('OSQPException', 'ValueError', 'RuntimeError'):
                raise
            return None
        if result.info.status != 'solved':
            return None
        x = np.asarray(result.x, np.float64)
        dual = np.asarray(result.y, np.float64)
    else:
        constraints = []
        for row, lo, hi in zip(a, lower, upper):
            if np.isfinite(hi):
                constraints.append({'type': 'ineq', 'fun': lambda x, r=row, h=hi: h-r@x,
                                    'jac': lambda x, r=row: -r})
            if np.isfinite(lo):
                constraints.append({'type': 'ineq', 'fun': lambda x, r=row, h=lo: r@x-h,
                                    'jac': lambda x, r=row: r})
        result = minimize(lambda x: (float(q@x+.5*x@p@x), q+p@x),
                          np.zeros(n) if warm is None else warm, jac=True,
                          constraints=constraints, method='SLSQP',
                          options={'ftol':1e-10, 'maxiter':200})
        if not result.success:
            return None
        x = np.asarray(result.x, np.float64)
        dual = np.zeros(len(a))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(dual)):
        return None
    ax = a @ x
    tol = 1e-6
    if np.any(ax < lower - tol) or np.any(ax > upper + tol):
        return None
    if osqp is not None and np.linalg.norm(p@x + q + a.T@dual, np.inf) > 1e-5*max(1,np.linalg.norm(q,np.inf)):
        return None
    return x, dual


class Eval:
    def __init__(self, u, score, aux, limits, admitted):
        self.u = np.asarray(u, np.float64).copy()
        self.score = float(score)
        self.f = float(aux['sensitivity_loss'])
        self.c = np.asarray(power(aux, limits), np.float64)
        self.feasible = bool(aux['is_feasible']) and bool(np.all(self.c <= 1e-10))
        self.admitted = bool(admitted)


class Oracle:
    def __init__(self, obj):
        self.obj = obj
        bounds = np.asarray(obj.bounds, np.float64)
        self.lo, self.hi = bounds
        self.span = self.hi - self.lo
        self.ids = np.flatnonzero(self.span > 0)
        self.lo_j = jnp.asarray(self.lo)
        self.span_j = jnp.asarray(self.span[self.ids])
        limits = obj.power_thresholds
        if not isinstance(limits, dict) or any(limits.get(k, 0) <= 0 for k in ('hard','soft','detector')):
            raise ValueError('positive public power thresholds are required')
        self.limits = {k:float(v) for k,v in limits.items()}
        self.raw = obj.value_function_aux(unbounded=False)
        self.latencies = []
        if self.raw is None:
            raise ValueError('the objective must expose value_function_aux')

    def to_x(self, u):
        x = self.lo.copy()
        x[self.ids] += self.span[self.ids] * np.asarray(u)
        return x

    def to_jax(self, u):
        return self.lo_j.at[self.ids].add(self.span_j * u)

    def from_x(self, x):
        return np.clip((np.asarray(x, np.float64)[self.ids] - self.lo[self.ids]) /
                       self.span[self.ids], 0, 1)

    def logged(self, u, score, aux):
        score, aux = jax.block_until_ready((score, aux))
        before = len(self.obj.loss_history)
        self.obj.log_evaluation(jnp.asarray(self.to_x(u)), score, aux=aux)
        return Eval(u, score, aux, self.limits, len(self.obj.loss_history) > before)

    def trial(self, u):
        began = time.perf_counter()
        u = np.clip(np.asarray(u, np.float64), 0, 1)
        before = len(self.obj.loss_history)
        score, aux = self.obj.value_aux(jnp.asarray(self.to_x(u)))
        score, aux = jax.block_until_ready((score, aux))
        obs = Eval(u, score, aux, self.limits, len(self.obj.loss_history) > before)
        self.latencies.append(time.perf_counter()-began)
        self.latencies = self.latencies[-100:]
        return obs

    def model(self, u, rows=None):
        began = time.perf_counter()
        u_j = jnp.asarray(u, jnp.float64)

        def fun(v):
            score, aux = self.raw(self.to_jax(v))
            return (aux['sensitivity_loss'], power(aux, self.limits)), (score, aux)

        (f, c), pull, (score, aux) = jax.vjp(fun, u_j, has_aux=True)
        obs = self.logged(u, score, aux)
        if not obs.admitted:
            return obs, None, None
        m = len(obs.c)
        if rows is None:
            rows = np.arange(m)
        rows = np.asarray(rows, np.int32)
        g = np.asarray(pull((jnp.ones_like(f), jnp.zeros_like(c)))[0], np.float64)
        parts = []
        for start in range(0, len(rows), 8):
            take = rows[start:start+8]
            padded = np.pad(take, (0, 8-len(take)), constant_values=-1)

            def one(i):
                return pull((jnp.zeros_like(f), jax.nn.one_hot(i, m, dtype=c.dtype)))[0]

            block = jax.lax.map(one, jnp.asarray(padded, jnp.int32))
            parts.append(np.asarray(block[:len(take)], np.float64))
        jac = np.concatenate(parts) if parts else np.empty((0, len(u)))
        del pull
        self.latencies.append(time.perf_counter()-began)
        self.latencies = self.latencies[-100:]
        return obs, g, jac

    def restore_model(self, u):
        began = time.perf_counter()
        u_j = jnp.asarray(u, jnp.float64)

        def fun(v):
            score, aux = self.raw(self.to_jax(v))
            c = power(aux, self.limits)
            return .5 * jnp.sum(jnp.maximum(c, 0) ** 2), (score, aux)

        (h, (score, aux)), grad = jax.value_and_grad(fun, has_aux=True)(u_j)
        obs = self.logged(u, score, aux)
        self.latencies.append(time.perf_counter()-began)
        self.latencies = self.latencies[-100:]
        return obs, float(h), np.asarray(grad, np.float64)


class Basin:
    def __init__(self, u, n, m):
        self.u = np.asarray(u, np.float64).copy()
        self.b = np.eye(n)
        self.lam = np.zeros(m)
        self.prev = np.zeros(m, bool)
        self.radius = .05
        self.fail = 0
        self.bad = 0
        self.warm = None
        self.score = np.inf


class FSQ(Relay):
    def __init__(self, obj, seed, init=None, batch=None):
        super().__init__(obj, seed, init, batch)
        self.oracle = None
        self.locals = []
        self.local_pos = 0
        self.restore_seconds = 0.0
        self.accepted = []
        self.local_oom = 0
        self.local_disabled = False

    def basins(self):
        if not self.archive and self.tracks:
            self.starts[0] = self.oracle.to_x(self.oracle.from_x(self.starts[0]))
            first = self.tracks[0]
            self.starts[0] = np.asarray(self.physical(first.p[None], jnp.asarray([first.kind]))[0])
        super().basins()

    def good(self, old, new, g, move):
        descent = float(g @ move)
        return (new.admitted and new.feasible and np.isfinite(new.score) and
                np.isfinite(new.f) and descent < 0 and
                new.f <= old.f + 1e-4 * descent + 1e-12 and
                new.score < old.score - 1e-10 * max(1, abs(old.score)))

    def anchor(self, obs):
        x = self.oracle.to_x(obs.u)
        self.keep(x, obs.score, True)
        self.accepted.append((obs.score, obs.feasible))
        if not self.tracks:
            return
        anchors = [t for t in self.tracks if t.anchor]
        pool = anchors or self.tracks
        target = min(pool, key=lambda t: self.distance(x, t.best_x if t.best_x is not None else
                     np.asarray(self.physical(t.p[None], jnp.asarray([t.kind]))[0])))
        old_best = None if target.best_x is None else target.best_x.copy()
        old_loss = target.best_loss
        self.replace(target, x, kind=1, anchor=True)
        target.best_x = old_best if old_best is not None and old_loss < obs.score else x.copy()
        target.best_loss = min(old_loss, obs.score)
        target.mark_loss = target.best_loss
        target.mark_time = float(self.obj.time_elapsed)

    def restore(self, u, stop):
        began = time.perf_counter()
        u = np.asarray(u, np.float64).copy()
        for _ in range(25):
            if self.obj.budget_exceeded or float(self.obj.time_elapsed) >= stop:
                break
            if time.perf_counter() - began >= 60 or self.restore_seconds + time.perf_counter() - began >= .05 * self.total:
                break
            old, h, g = self.oracle.restore_model(u)
            if not old.admitted:
                break
            if old.feasible:
                self.keep(self.oracle.to_x(u), old.score, True)
                self.restore_seconds += time.perf_counter() - began
                return old
            if not np.all(np.isfinite(g)) or np.linalg.norm(g, np.inf) == 0:
                break
            changed = False
            for k in range(12):
                s = np.clip(u - (2.0 ** -k) * g / max(1, np.linalg.norm(g, np.inf)), 0, 1) - u
                if not np.any(s):
                    continue
                trial = self.oracle.trial(u+s)
                if not trial.admitted:
                    break
                if trial.feasible:
                    self.keep(self.oracle.to_x(trial.u), trial.score, True)
                    self.restore_seconds += time.perf_counter() - began
                    return trial
                next_h = .5 * float(np.sum(np.maximum(trial.c, 0) ** 2))
                if next_h <= h + 1e-4 * float(g @ s):
                    u = trial.u
                    changed = True
                    break
            if not changed:
                break
        self.restore_seconds += time.perf_counter() - began
        return None

    def pick(self, count=None, spacing=1e-3):
        chosen = []
        for q in self.archive:
            if all(self.distance(q.x, old.x) > spacing for old in chosen):
                chosen.append(q)
            if count is not None and len(chosen) >= count:
                break
        return chosen

    def seed_basins(self, stop):
        for q in self.pick(6, .03):
            u = self.oracle.from_x(q.x)
            trial = self.oracle.trial(u)
            if trial.admitted and trial.feasible:
                basin = Basin(u, len(u), len(trial.c))
                basin.score = trial.score
                self.locals.append(basin)
        if self.locals:
            return
        pool = [t for t in self.tracks if t.best_x is not None]
        if not pool:
            pool = self.tracks[:3]
        for t in pool[:3]:
            x = t.best_x if t.best_x is not None else np.asarray(
                self.physical(t.p[None], jnp.asarray([t.kind]))[0])
            trial = self.restore(self.oracle.from_x(x), stop)
            if trial is not None:
                basin = Basin(trial.u, len(trial.u), len(trial.c))
                basin.score = trial.score
                self.locals.append(basin)

    def refresh_basins(self):
        for q in self.pick(6, .03):
            u = self.oracle.from_x(q.x)
            if self.locals:
                near = min(self.locals, key=lambda b:np.sqrt(np.mean((b.u-u)**2)))
                gap = np.sqrt(np.mean((near.u-u)**2))
                if gap <= .03:
                    if q.loss < near.score-1e-4:
                        near.u = u
                        near.score = q.loss
                        near.b = np.eye(len(u))
                        near.lam[:] = 0
                        near.prev[:] = False
                        near.warm = None
                        near.radius = .05
                        near.fail = 0
                        near.bad = 0
                    continue
            if len(self.locals) >= 6:
                worst = max(self.locals,key=lambda b:b.score)
                if q.loss >= worst.score-1e-4:
                    continue
                self.locals.remove(worst)
            trial = self.oracle.trial(u)
            if trial.admitted and trial.feasible:
                new = Basin(u,len(u),len(trial.c))
                new.score = trial.score
                self.locals.append(new)

    def active(self, basin, obs):
        margin = max(1e-4, min(.02, basin.radius ** 2))
        return np.flatnonzero((obs.c >= -margin) | (basin.lam > 1e-8) | basin.prev)

    def bounds_step(self, basin):
        return np.maximum(-basin.u, -basin.radius), np.minimum(1-basin.u, basin.radius)

    def sqp(self, basin, obs, g, jac, rows):
        n = len(g)
        low, high = self.bounds_step(basin)
        a = np.vstack((jac, np.eye(n)))
        lower = np.r_[np.full(len(rows), -np.inf), low]
        upper = np.r_[-obs.c[rows], high]
        solved = solve_qp(basin.b, g, a, lower, upper, basin.warm)
        if solved is None:
            return None
        d, dual = solved
        pred = float(g@d + .5*d@basin.b@d)
        if not np.isfinite(pred) or pred >= -1e-12 or g@d >= -1e-12:
            return None
        basin.warm = d
        return d, np.maximum(0, dual[:len(rows)]), pred

    def correct(self, basin, old, trial, d, g):
        limit = min(basin.radius / 2, 2 * np.linalg.norm(d, np.inf) ** 2)
        if limit <= 1e-12:
            return None
        for _ in range(2):
            rows = np.flatnonzero(trial.c > -1e-7)
            if not len(rows):
                return None
            measured, _, jac = self.oracle.model(trial.u, rows)
            if not measured.admitted or jac is None or not np.all(np.isfinite(jac)):
                return None
            used = trial.u - old.u - d
            lower = np.maximum.reduce((-trial.u, -limit-used,
                                       -np.full(len(d), basin.radius)-d-used))
            upper = np.minimum.reduce((1-trial.u, limit-used,
                                       np.full(len(d), basin.radius)-d-used))
            a = np.vstack((jac, np.eye(len(d))))
            solved = solve_qp(np.eye(len(d)), np.zeros(len(d)), a,
                              np.r_[np.full(len(rows), -np.inf), lower],
                              np.r_[-measured.c[rows]-1e-10, upper])
            if solved is None:
                return None
            r = solved[0]
            s = trial.u - old.u + r
            if g@s >= 0:
                return None
            trial = self.oracle.trial(old.u+s)
            if not trial.admitted:
                return None
            if self.good(old, trial, g, s):
                return trial, s
        return None

    def safeguard(self, basin, obs, g, jac, rows):
        n = len(g)
        scale = max(1, np.linalg.norm(g, np.inf))
        b = clip_b(basin.b / scale, 1e-4, 1e4)
        p = np.zeros((n+1, n+1))
        p[:n, :n] = b
        p[-1, -1] = 1e-12
        q = np.r_[np.zeros(n), 1.0]
        low, high = self.bounds_step(basin)
        a = np.vstack((np.r_[g/scale, -1.0],
                       np.column_stack((jac, -np.ones(len(rows)))),
                       np.r_[np.zeros(n), 1.0],
                       np.column_stack((np.eye(n), np.zeros(n)))))
        lower = np.r_[-np.inf, np.full(len(rows), -np.inf), -np.inf, low]
        upper = np.r_[0, -obs.c[rows], 0, high]
        solved = solve_qp(p, q, a, lower, upper)
        if solved is None:
            return None
        d, t = solved[0][:n], solved[0][-1]
        if t + .5*d@b@d >= -1e-12 or g@d >= -1e-12:
            return None
        for k in range(12):
            s = (2.0 ** -k) * d
            trial = self.oracle.trial(obs.u+s)
            if not trial.admitted:
                break
            if self.good(obs, trial, g, s):
                return trial, s
        return None

    def refine_once(self, basin, stop):
        if self.obj.budget_exceeded or float(self.obj.time_elapsed) >= stop:
            return False
        trial = self.oracle.trial(basin.u)
        if not trial.admitted:
            return False
        if not trial.feasible:
            restored = self.restore(basin.u, stop)
            if restored is None:
                return False
            basin.u = restored.u
            basin.score = restored.score
            trial = restored
        rows = self.active(basin, trial)
        old, g, jac = self.oracle.model(basin.u, rows)
        if not old.admitted or g is None:
            return False
        if not (old.feasible and np.isfinite(old.score) and np.isfinite(old.f) and
                np.all(np.isfinite(old.c)) and np.all(np.isfinite(g)) and np.all(np.isfinite(jac))):
            basin.bad += 1
            basin.b = np.eye(len(basin.u))
            if basin.bad >= 3:
                return False
            return True
        basin.bad = 0
        outcome = None
        proposal = self.sqp(basin, old, g, jac, rows)
        if proposal is not None:
            d, dual, pred = proposal
            for k, alpha in enumerate((1.0, .5, .25)):
                if self.obj.budget_exceeded or float(self.obj.time_elapsed) >= stop:
                    break
                s = alpha * d
                tested = self.oracle.trial(old.u+s)
                if not tested.admitted:
                    break
                basin.prev |= tested.c > 0
                if self.good(old, tested, g, s):
                    outcome = (tested, s, dual, pred, True)
                    break
                if k == 0 and not tested.feasible and pred < 0:
                    fixed = self.correct(basin, old, tested, d, g)
                    if fixed is not None:
                        outcome = (fixed[0], fixed[1], dual, pred, True)
                        break
        if outcome is None and not self.obj.budget_exceeded and float(self.obj.time_elapsed) < stop:
            safe = self.safeguard(basin, old, g, jac, rows)
            if safe is not None:
                outcome = (safe[0], safe[1], None, None, False)
        if outcome is None:
            basin.radius = max(1e-6, basin.radius * .25)
            basin.fail = basin.fail+1 if basin.radius <= 1e-6 else 0
            return basin.fail < 3
        new, s, dual, pred, was_sqp = outcome
        basin.u = new.u
        basin.score = new.score
        basin.fail = 0
        basin.prev[:] = False
        self.anchor(new)
        if was_sqp:
            basin.lam[:] = 0
            basin.lam[rows] = dual
            predicted = -(float(g@s + .5*s@basin.b@s))
            ratio = (old.f-new.f) / predicted if predicted > 0 else 0
            if ratio > .75 and np.linalg.norm(s, np.inf) >= .8 * basin.radius:
                basin.radius = min(.25, 2*basin.radius)
            elif ratio < .1:
                basin.radius = max(1e-6, basin.radius / 2)
            if not self.obj.budget_exceeded and float(self.obj.time_elapsed) < stop:
                fresh, g2, j2 = self.oracle.model(new.u, rows)
                if fresh.admitted and g2 is not None and np.all(np.isfinite(g2)) and np.all(np.isfinite(j2)):
                    lag1 = g + jac.T @ dual
                    lag2 = g2 + j2.T @ dual
                    basin.b = bfgs(basin.b, s, lag2-lag1)
        return True

    def local_slice(self, stop, final=False):
        if self.local_disabled:
            return False
        if not self.locals:
            self.seed_basins(stop)
        if self.locals:
            self.refresh_basins()
        if final:
            chosen = self.pick(3)
            selected = []
            for q in chosen:
                u = self.oracle.from_x(q.x)
                nearest = min(self.locals, key=lambda b: np.linalg.norm(b.u-u)) if self.locals else None
                if nearest is None or np.linalg.norm(nearest.u-u) > 1e-3:
                    nearest = Basin(u, len(u), len(self.oracle.trial(u).c))
                    self.locals.append(nearest)
                selected.append(nearest)
        else:
            selected = self.locals
        if not selected:
            return False
        active = list(selected)
        done = 0
        while active and not self.obj.budget_exceeded and float(self.obj.time_elapsed) < stop:
            j = self.local_pos % len(active)
            basin = active[j]
            before = float(self.obj.time_elapsed)
            try:
                more = self.refine_once(basin, stop)
            except Exception as exc:
                if memory_error(exc):
                    basin.b = np.eye(len(basin.u))
                    more = False
                    self.local_oom += 1
                    self.recover(exc)
                    if self.local_oom >= 2:
                        self.local_disabled = True
                    print(f'[n7] local memory: {type(exc).__name__}: {str(exc)[:120]}', flush=True)
                else:
                    raise
            if not more:
                active.pop(j)
            else:
                self.local_pos += 1
            done += 1
            if float(self.obj.time_elapsed) <= before or done >= 10000:
                break
            if self.local_disabled:
                break
        return done > 0

    def run(self):
        self.ready()
        self.obj.start_logging()
        self.oracle = Oracle(self.obj)
        self.screen()
        if not np.any(self.live):
            self.confirm()
            return
        self.descend(.25)
        if self.obj.budget_exceeded:
            return
        self.basins()
        self.seed_basins(min(.30*self.total, self.total-10))
        rescue_at = [.4, .6, .8]
        which = 0
        while not self.obj.budget_exceeded and float(self.obj.budget_progress_fraction) < .85:
            now = float(self.obj.time_elapsed)
            next_event = rescue_at[which] if which < len(rescue_at) else .85
            pop_stop = min(.85, next_event, (now+120)/self.total)
            self.descend(pop_stop)
            if self.obj.budget_exceeded:
                return
            if which < len(rescue_at) and float(self.obj.budget_progress_fraction) >= rescue_at[which]:
                self.rescue()
                which += 1
                continue
            now = float(self.obj.time_elapsed)
            local_stop = min(.85, next_event, (now+120)/self.total)
            self.local_slice(local_stop*self.total)
            if float(self.obj.budget_progress_fraction) < local_stop:
                self.descend(local_stop)
            if which < len(rescue_at) and float(self.obj.budget_progress_fraction) >= rescue_at[which]:
                self.rescue()
                which += 1
        waits = self.latencies + self.oracle.latencies
        reserve = max(10.0, 2*np.percentile(waits,95)) if waits else 10.0
        end = max(0.0, self.total-reserve)
        if float(self.obj.time_elapsed) < end and not self.obj.budget_exceeded:
            self.local_slice(end, final=True)
        if float(self.obj.time_elapsed) < end and not self.obj.budget_exceeded:
            self.descend(end/self.total, quiet=True)
        self.confirm()
