import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .warmstart import Prior, build_starts, find_prior, donor_starts


class StopPolish(Exception):
    pass


class Track:
    def __init__(self, z, ident, seed, now):
        self.z = np.asarray(z, np.float64).copy()
        self.ident = ident
        self.rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), ident]))
        self.m = np.zeros_like(self.z)
        self.v = np.zeros_like(self.z)
        self.count = 0
        self.best_z = None
        self.best_loss = np.inf
        self.mark_loss = np.inf
        self.mark_time = now
        self.born = now
        self.last_loss = np.inf
        self.last_feasible = False

    def reset(self, z, ident, seed, now):
        self.__init__(z, ident, seed, now)


class Portfolio:
    def __init__(self, obj, seed, mode, init_params=None, batch=None):
        self.obj = obj
        self.seed = int(seed)
        self.mode = mode
        self.bounds = np.asarray(obj.bounds, np.float64)
        self.span = self.bounds[1] - self.bounds[0]
        self.live = self.span > 0
        self.total = float(obj.max_time or 14400.0)
        self.batch = min(4, max(1, int(batch or os.environ.get('L2D_BATCH', '2'))))
        self.drops = 0
        self.inc_z = None
        self.inc_loss = np.inf
        self.archive = []
        self.steps = 0
        self.next_id = 0
        self.rescue_count = 0
        self.donor_count = 0
        self.rng = np.random.default_rng(self.seed)
        self.prior = Prior(find_prior())
        self.starts = build_starts(obj, 36, self.prior, self.seed)
        if init_params is not None:
            z = np.asarray(init_params, np.float64)
            if z.shape != (obj.n_params,):
                raise ValueError('init_params shape mismatch')
            self.starts[0] = z
        self.tracks = []

    def ready(self):
        while True:
            try:
                for b in sorted(set([1, self.batch])):
                    self.obj.warmup_vmap_value_aux(batch_size=b)
                    self.obj.warmup_vmap_value_and_grad_aux(batch_size=b)
                self.obj.warmup_value_aux()
                self.obj.warmup_value_and_grad_aux()
                jnp.ones(1, dtype=jnp.float64).block_until_ready()
                if hasattr(jax, 'effects_barrier'):
                    jax.effects_barrier()
                return
            except Exception as exc:
                if not self.recover(exc):
                    raise

    def recover(self, exc):
        msg = str(exc).lower()
        is_oom = any(s in msg for s in (
            'resource_exhausted', 'out of memory',
            'cuda_error_out_of_memory', 'allocator-allocation-error'))
        if not is_oom or self.batch == 1 or self.drops >= 2:
            return False
        self.batch = max(1, self.batch // 2)
        self.drops += 1
        print(f'[gwopt] oom recovery batch={self.batch}', flush=True)
        return True

    def time_left(self):
        return max(0.0, float(self.obj.time_left or 0.0)) if self.obj.max_time else np.inf

    def progress(self):
        return float(self.obj.budget_progress_fraction)

    def to_physical(self, z):
        return self.bounds[0] + self.span * expit(np.asarray(z, np.float64))

    def distance(self, a, b):
        if not self.live.any():
            return 0.0
        xa, xb = self.to_physical(a), self.to_physical(b)
        d = (xa[self.live] - xb[self.live]) / self.span[self.live]
        return float(np.sqrt(np.mean(d * d)))

    def keep(self, z, loss, feasible, track=None):
        loss = float(loss)
        if not np.isfinite(loss):
            return
        now = float(self.obj.time_elapsed)
        if track is not None:
            track.last_loss = loss
            track.last_feasible = bool(feasible)
        if not feasible:
            return
        old_best = track.best_loss if track is not None else np.inf
        if loss >= old_best and loss >= self.inc_loss:
            return
        z = np.asarray(z, np.float64).copy()
        if track is not None and loss < track.best_loss:
            track.best_z = z
            track.best_loss = loss
            if not np.isfinite(track.mark_loss) or track.mark_loss - loss > 1e-4:
                track.mark_loss = loss
                track.mark_time = now
        if loss < self.inc_loss:
            self.inc_z = z.copy()
            self.inc_loss = loss
            print(f'[gwopt] t={now:.1f}s best_feasible={loss:+.8f}', flush=True)
        for i, (old_z, old_loss, _) in enumerate(self.archive):
            if self.distance(z, old_z) <= 1e-3:
                if loss < old_loss:
                    self.archive[i] = (z, loss, track.ident if track else None)
                break
        else:
            self.archive.append((z, loss, track.ident if track else None))
        self.archive.sort(key=lambda a: a[1])
        self.archive = self.archive[:32]

    def evaluate(self, points, grad=False):
        z = jnp.asarray(points, jnp.float64)
        while True:
            try:
                if grad:
                    v, g, aux = self.obj.vmap_value_and_grad_aux(z)
                    return (np.asarray(v), np.asarray(g),
                            np.asarray(aux['is_feasible'], bool))
                v, aux = self.obj.vmap_value_aux(z)
                return (np.asarray(v), None,
                        np.asarray(aux['is_feasible'], bool))
            except Exception as exc:
                if not self.recover(exc):
                    raise
                raise RuntimeError('retry smaller chunk') from exc

    def apply_gradient(self, track, grad, sigma=0.0, protect=False):
        g = np.where(np.isfinite(grad), grad, 0.0)
        g[~self.live] = 0.0
        norm = np.linalg.norm(g)
        if norm > 1:
            g = g / norm
        track.count += 1
        track.m = .9 * track.m + .1 * g
        track.v = .999 * track.v + .001 * g * g
        m_hat = track.m / (1 - .9 ** track.count)
        v_hat = track.v / (1 - .999 ** track.count)
        step = -.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        if sigma > 0 and not protect:
            noise = track.rng.normal(size=track.z.size) * sigma
            noise_size = np.linalg.norm(noise)
            if noise_size:
                noise *= min(1.0, .25 * np.linalg.norm(step) / noise_size)
                step += noise
        step[~self.live] = 0.0
        track.z += step

    def screen(self):
        vals, feas = [], []
        idx = 0
        while idx < len(self.starts) and not self.obj.budget_exceeded:
            b = min(self.batch, len(self.starts) - idx)
            try:
                v, _, f = self.evaluate(self.starts[idx:idx + b], False)
            except RuntimeError as exc:
                if str(exc) == 'retry smaller chunk':
                    continue
                raise
            if self.obj.budget_exceeded:
                break
            for j in range(b):
                self.keep(self.starts[idx + j], v[j], f[j])
            vals.extend(v.tolist())
            feas.extend(f.tolist())
            idx += b
        if not vals:
            raise RuntimeError('screen did not evaluate a candidate')
        vals = np.asarray(vals)
        feas = np.asarray(feas)
        rank = np.lexsort((np.where(np.isfinite(vals), vals, np.inf), ~feas))
        now = float(self.obj.time_elapsed)
        self.tracks = [self._new_track(self.starts[i], now) for i in rank[:19]]
        for t, i in zip(self.tracks, rank[:19]):
            self.keep(t.z, vals[i], feas[i], t)

    def _new_track(self, z, now):
        t = Track(z, self.next_id, self.seed, now)
        self.next_id += 1
        return t

    def noise_level(self):
        t = min(1.0, self.progress() / .5)
        return .3 * 0.5 * (1 + np.cos(np.pi * t))

    def descend(self, stop_frac, quiet=False):
        while not self.obj.budget_exceeded and self.progress() < stop_frac:
            done = 0
            while done < len(self.tracks) and not self.obj.budget_exceeded:
                b = min(self.batch, len(self.tracks) - done)
                group = self.tracks[done:done + b]
                points = np.stack([t.z for t in group])
                try:
                    v, g, fe = self.evaluate(points, True)
                except RuntimeError as exc:
                    if str(exc) == 'retry smaller chunk':
                        continue
                    raise
                if self.obj.budget_exceeded:
                    break
                for k, t in enumerate(group):
                    self.keep(points[k], v[k], fe[k], t)
                    is_anchor = ((done + k == 0) if len(self.tracks) == 19
                                 else ((done + k) % 6 == 0))
                    self.apply_gradient(
                        t, g[k], 0.0 if quiet else self.noise_level(),
                        is_anchor)
                done += b
            if done == 0:
                break
            self.steps += 1
            if (self.mode == 'repair' and len(self.tracks) == 18
                    and self.steps % 300 == 0):
                self._replace_worst()
            if self.steps % 200 == 0:
                self.confirm()

    def form_basins(self):
        rank = sorted(self.tracks, key=lambda t: (
            not np.isfinite(t.best_loss), t.best_loss, t.last_loss))[:3]
        while len(rank) < 3:
            rank.append(self.tracks[len(rank) % len(self.tracks)])
        now = float(self.obj.time_elapsed)
        out = []
        for src in rank:
            for j in range(6):
                z = src.best_z if src.best_z is not None else src.z
                if j:
                    z = z + src.rng.normal(size=z.size) * .2
                out.append(self._new_track(z, now))
        self.tracks = out

    def _replace_worst(self):
        if self.inc_z is None:
            return
        for start in [0, 6, 12]:
            part = self.tracks[start + 1:start + 6]
            worst = max(part, key=lambda t: (not t.last_feasible, t.last_loss))
            z = self.inc_z + worst.rng.normal(size=worst.z.size) * .2
            self._replace_track(worst, z)

    def _replace_track(self, track, z):
        now = float(self.obj.time_elapsed)
        track.reset(z, self.next_id, self.seed, now)
        self.next_id += 1

    def rescue(self):
        k = len(self.tracks) // 3
        worst = sorted(self.tracks, key=lambda t: (
            np.isfinite(t.best_loss),
            -t.best_loss if np.isfinite(t.best_loss) else 0))[:k]
        eligible = [
            t for t in worst
            if self.tracks.index(t) % 6
            and float(self.obj.time_elapsed) - t.mark_time >= .1 * self.total
            and float(self.obj.time_elapsed) - t.born >= .1 * self.total]
        for t in eligible:
            if self.rescue_count % 2 == 0 or not self.archive:
                z = donor_starts(self.obj, self.prior, self.rng,
                                 64, self.donor_count)
                self.donor_count += 1
            else:
                parent = self.archive[
                    (self.rescue_count // 2) % min(3, len(self.archive))][0]
                z = parent + t.rng.normal(size=parent.size) * .2
            self._replace_track(t, z)
            self.rescue_count += 1
        print(f'[gwopt] rescue t={self.obj.time_elapsed:.1f}s '
              f'changed={len(eligible)}', flush=True)

    def confirm(self):
        if self.inc_z is None or self.obj.budget_exceeded or self.time_left() < 2:
            return
        loss, aux = self.obj.value_aux(jnp.asarray(self.inc_z))
        self.keep(self.inc_z, loss, bool(aux['is_feasible']))

    def polish(self, event, allowance):
        if not self.archive or self.obj.budget_exceeded:
            return
        end = min(float(self.obj.time_elapsed) + allowance, self.total - 1.0)
        picks = self.archive[:1 if self.mode == 'repair' else 3]
        share = (end - float(self.obj.time_elapsed)) / len(picks)
        for i, (z, old_loss, ident) in enumerate(picks):
            due = min(end, float(self.obj.time_elapsed) + share)
            best = [np.asarray(z, np.float64).copy(), old_loss]

            def fun(y):
                if self.obj.budget_exceeded or self.obj.time_elapsed >= due:
                    raise StopPolish()
                loss, grad, aux = self.obj.value_and_grad_aux(
                    jnp.asarray(y, jnp.float64))
                val = float(loss)
                if bool(aux['is_feasible']) and np.isfinite(val) and val < best[1]:
                    best[0] = np.asarray(y, np.float64).copy()
                    best[1] = val
                grad = np.where(np.isfinite(np.asarray(grad)),
                                np.asarray(grad), 0)
                return (val if np.isfinite(val) else 1e10,
                        grad.astype(np.float64))

            try:
                minimize(fun, z, jac=True, method='L-BFGS-B',
                         options={'maxls': 10, 'maxiter': 10000, 'maxcor': 10})
            except StopPolish:
                pass
            except (FloatingPointError, ValueError):
                pass
            if (best[1] < old_loss and not self.obj.budget_exceeded
                    and self.time_left() > 1):
                loss, aux = self.obj.value_aux(jnp.asarray(best[0]))
                if bool(aux['is_feasible']) and float(loss) < old_loss:
                    self.keep(best[0], loss, True)
                    target = next(
                        (t for t in self.tracks if t.ident == ident), None)
                    if target is None:
                        target = max(self.tracks, key=lambda t: (
                            not np.isfinite(t.best_loss),
                            t.best_loss if np.isfinite(t.best_loss) else np.inf))
                    self._replace_track(target, best[0])
            if self.obj.budget_exceeded or self.obj.time_elapsed >= end:
                break

    def run(self):
        self.ready()
        self.obj.start_logging()
        self.screen()
        self.descend(.25)
        if self.obj.budget_exceeded:
            return
        self.form_basins()
        if self.mode == 'repair':
            self.descend(.875)
            self.polish('late', .06 * self.total)
        else:
            events = [(.4, 'rescue'), (.5, 'polish'), (.6, 'rescue'),
                      (.75, 'polish'), (.8, 'rescue'), (.9, 'polish')]
            for at, kind in events:
                self.descend(at)
                if self.obj.budget_exceeded:
                    break
                if kind == 'rescue':
                    self.rescue()
                else:
                    self.polish(at, .02 * self.total)
        self.descend(1.01, quiet=True)
        self.confirm()
