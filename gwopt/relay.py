import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .warmstart import (Prior, _fill_from_donor, _fill_from_prior,
                         _get_param_keys, _nearest_donors, find_prior)
from .topology import signature


class StopPolish(Exception):
    pass


def logit(u):
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return np.log(u) - np.log1p(-u)


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


def generate_starts(obj, prior, seed, init=None):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(obj.bounds, np.float64)
    lo, hi = bounds
    pairs = _get_param_keys(obj)
    spec = obj.problem_spec.get('params', obj.problem_spec)
    topo, size = spec.get('topology'), spec.get('size')

    donor_ids = []
    if topo is not None and size is not None:
        donor_ids = _nearest_donors(prior, signature(topo, int(size)), int(size), 64)
    donors = [_fill_from_donor(prior, int(i), pairs, lo, hi, rng) for i in donor_ids]
    if not donors:
        donors = [_fill_from_prior(prior, pairs, lo, hi, rng) for _ in range(16)]

    first = [donors[i % len(donors)].copy() for i in range(16)]
    blocks = param_groups(pairs)
    mixed = [crossover(donors[i % len(donors)],
                        donors[(i + 1 + i // 3) % len(donors)], blocks, rng)
             for i in range(8)]
    sobol = qmc.Sobol(d=len(pairs), scramble=True, seed=seed).random_base2(3)
    broad = [lo + (hi - lo) * row for row in sobol]
    draws = [_fill_from_prior(prior, pairs, lo, hi, rng) for _ in range(4)]

    x = np.clip(np.asarray(first + mixed + broad + draws, np.float64), lo, hi)
    if init is not None:
        p = np.asarray(init, np.float64)
        if p.shape != (obj.n_params,) or not np.all(np.isfinite(p)):
            raise ValueError('init_params must be a finite physical parameter vector')
        x[-1] = np.clip(p, lo, hi)
    return x, blocks, donors


@jax.jit
def gradient_step(params, momentum, variance, count, keys, kind, anchor,
                  grad, span, live, sigma):
    u = jnp.where(kind[:, None] == 0, jax.nn.sigmoid(params),
                  jnp.clip(params, 0, 1))
    g = span * grad
    g = jnp.where(kind[:, None] == 0, g * u * (1 - u), g)
    g = jnp.where(live, jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    norm = jnp.linalg.norm(g, axis=1, keepdims=True)
    g = g / jnp.maximum(norm, 1.0)

    count = count + 1
    momentum = .9 * momentum + .1 * g
    variance = .999 * variance + .001 * g * g
    m_hat = momentum / (1 - .9 ** count[:, None])
    v_hat = variance / (1 - .999 ** count[:, None])
    lr = jnp.where(kind == 0, .1, .01)[:, None]
    move = -lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

    split = jax.vmap(jax.random.split)(keys)
    keys = split[:, 0]
    noise = jax.vmap(lambda key: jax.random.normal(key, (params.shape[1],)))(split[:, 1])
    noise = noise * (sigma * (kind == 0) * (~anchor))[:, None]
    noise_size = jnp.linalg.norm(noise, axis=1, keepdims=True)
    cap = .25 * jnp.linalg.norm(move, axis=1, keepdims=True)
    noise = noise * jnp.minimum(1.0, cap / jnp.maximum(noise_size, 1e-30))

    params = params + jnp.where(live, move + noise, 0.0)
    params = jnp.where(kind[:, None] == 1, jnp.clip(params, 0, 1), params)
    outward = ((params <= 0) & (momentum > 0)) | ((params >= 1) & (momentum < 0))
    momentum = jnp.where((kind[:, None] == 1) & outward, 0.0, momentum)
    return params, momentum, variance, count, keys


class ArchivePoint:
    def __init__(self, x, loss, ident=None):
        self.x = np.asarray(x, np.float64).copy()
        self.loss = float(loss)
        self.ident = ident
        self.radius = .05


class Track:
    def __init__(self, params, kind, ident, seed, now, anchor=False):
        self.p = jnp.asarray(params, jnp.float64)
        self.kind = int(kind)
        self.ident = int(ident)
        self.anchor = bool(anchor)
        self.m = jnp.zeros_like(self.p)
        self.v = jnp.zeros_like(self.p)
        self.count = jnp.asarray(0, jnp.int32)
        key_seed = np.random.SeedSequence([int(seed), int(ident)]).generate_state(1)[0]
        self.key = jax.random.PRNGKey(int(key_seed))
        self.best_x = None
        self.best_loss = np.inf
        self.mark_loss = np.inf
        self.mark_time = float(now)
        self.born = float(now)
        self.last_loss = np.inf
        self.last_feasible = False
        self.last_violation = np.inf
        self.bad = 0


class Relay:
    def __init__(self, obj, seed, init=None, batch=None):
        self.obj = obj
        self.seed = int(seed)
        self.bounds = np.asarray(obj.bounds, np.float64)
        self.lo, self.hi = self.bounds
        self.span = self.hi - self.lo
        self.live = self.span > 0
        self.safe_span = np.where(self.live, self.span, 1)
        self.lo_dev = jnp.asarray(self.lo)
        self.span_dev = jnp.asarray(self.span)
        self.live_dev = jnp.asarray(self.live)
        self.total = float(obj.max_time or 14400)
        self.batch = min(8, max(1, int(batch or os.environ.get('L2D_BATCH', '2'))))
        self.drops = 0
        self.prior = Prior(find_prior())
        self.starts, self.blocks, self.donors = generate_starts(obj, self.prior, seed, init)
        self.rng = np.random.default_rng(seed)
        self.tracks = []
        self.archive = []
        self.inc = None
        self.confirmed = None
        self.next_id = 0
        self.steps = 0
        self.rescues = 0
        self.failures = 0
        self.last_confirm = -np.inf
        self.latencies = []

    def new_track(self, x, kind=0, anchor=False):
        u = np.clip((np.asarray(x) - self.lo) / self.safe_span, 0, 1)
        p = logit(u) if kind == 0 else u
        t = Track(p, kind, self.next_id, self.seed, self.obj.time_elapsed, anchor)
        self.next_id += 1
        return t

    def replace_track(self, track, x, kind=0, anchor=None):
        replacement = self.new_track(x, kind, track.anchor if anchor is None else anchor)
        track.__dict__.update(replacement.__dict__)

    def physical(self, params, kind):
        u = jnp.where(kind[:, None] == 0, jax.nn.sigmoid(params),
                      jnp.clip(params, 0, 1))
        return self.lo_dev + self.span_dev * u

    def distance(self, a, b):
        if not self.live.any():
            return 0.0
        d = (np.asarray(a)[self.live] - np.asarray(b)[self.live]) / self.span[self.live]
        return float(np.sqrt(np.mean(d * d)))

    def keep(self, x, loss, feasible, track=None, confirm=False):
        loss = float(loss)
        if track is not None:
            track.last_loss = loss
            track.last_feasible = bool(feasible)
        if not feasible or not np.isfinite(loss):
            return
        x = np.asarray(x, np.float64).copy()
        if track is not None and loss < track.best_loss:
            track.best_x = x.copy()
            track.best_loss = loss
            if not np.isfinite(track.mark_loss) or track.mark_loss - loss > 1e-4:
                track.mark_loss = loss
                track.mark_time = float(self.obj.time_elapsed)
        if self.inc is None or loss < self.inc.loss:
            self.inc = ArchivePoint(x, loss, track.ident if track else None)
            if confirm or loss < (self.archive[0].loss - 1e-4 if self.archive else np.inf):
                print(f'[gwopt] t={self.obj.time_elapsed:.1f}s feasible={loss:+.8f}',
                      flush=True)
        for q in self.archive:
            if self.distance(x, q.x) <= 1e-3:
                if loss < q.loss:
                    q.x = x
                    q.loss = loss
                    q.ident = track.ident if track else None
                break
        else:
            self.archive.append(ArchivePoint(x, loss, track.ident if track else None))
        self.archive.sort(key=lambda q: q.loss)
        del self.archive[32:]

    def violation(self, aux, size):
        powers = aux.get('power_values')
        limits = self.obj.power_thresholds
        if not isinstance(powers, dict) or not isinstance(limits, dict):
            return np.full(size, np.inf)
        out = np.zeros(size)
        for name, values in powers.items():
            limit = float(limits[name])
            p = np.asarray(values, np.float64).reshape(size, -1)
            if p.shape[1]:
                excess = np.nan_to_num(p / limit - 1, nan=np.inf, posinf=np.inf)
                out = np.maximum(out, np.max(np.maximum(excess, 0), axis=1))
        return out

    def recover(self, exc):
        msg = str(exc).lower()
        is_oom = any(s in msg for s in (
            'resource_exhausted', 'out of memory',
            'cuda_error_out_of_memory', 'allocator-allocation-error'))
        if not is_oom or self.batch <= 1 or self.drops >= 2:
            return False
        self.batch = max(1, self.batch // 2)
        self.drops += 1
        print(f'[gwopt] oom recovery batch={self.batch}', flush=True)
        return True

    def ready(self):
        while True:
            try:
                for b in sorted({1, self.batch}):
                    self.obj.warmup_vmap_value_aux(batch_size=b)
                    self.obj.warmup_vmap_value_and_grad_aux(batch_size=b)
                self.obj.warmup_value_aux()
                self.obj.warmup_value_and_grad_aux()
                for b in sorted({1, self.batch}):
                    p = jnp.zeros((b, self.lo.size), jnp.float64)
                    z = jnp.zeros_like(p)
                    c = jnp.zeros(b, jnp.int32)
                    k = jnp.stack([jax.random.PRNGKey(i) for i in range(b)])
                    kind = jnp.zeros(b, jnp.int32)
                    anch = jnp.zeros(b, bool)
                    out = gradient_step(p, z, z, c, k, kind, anch, z,
                                        self.span_dev, self.live_dev, 0.0)
                    out[0].block_until_ready()
                if hasattr(jax, 'effects_barrier'):
                    jax.effects_barrier()
                break
            except Exception as exc:
                if not self.recover(exc):
                    raise

    def screen(self):
        vals, feas, viol = [], [], []
        i = 0
        while i < len(self.starts) and not self.obj.budget_exceeded:
            b = min(self.batch, len(self.starts) - i)
            x = jnp.asarray(self.starts[i:i + b])
            try:
                v, aux = self.obj.vmap_value_aux(x)
                v = np.asarray(v)
                f = np.asarray(aux['is_feasible'], bool)
                h = self.violation(aux, b)
            except Exception as exc:
                if self.recover(exc):
                    continue
                raise
            for j in range(b):
                self.keep(self.starts[i + j], v[j], f[j])
            vals.extend(v.tolist())
            feas.extend(f.tolist())
            viol.extend(h.tolist())
            i += b

        if not vals:
            raise RuntimeError('screen produced no evaluations')
        vals = np.asarray(vals)
        feas = np.asarray(feas)
        viol = np.asarray(viol)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        viol = np.where(np.isfinite(viol), viol, np.inf)
        if not np.isfinite(vals).any():
            raise FloatingPointError('all screening candidates had nonfinite loss')

        rank = np.lexsort((vals, viol, ~feas))
        chosen = list(rank[:min(12, len(rank))])
        rest = [int(k) for k in rank if k not in chosen]
        while rest and len(chosen) < min(19, len(rank)):
            preferred = [k for k in rest if feas[k]] or rest
            k = max(preferred, key=lambda j: (
                min(self.distance(self.starts[j], self.starts[a]) for a in chosen),
                -int(np.where(rank == j)[0][0])))
            chosen.append(k)
            rest.remove(k)
        for j in chosen:
            t = self.new_track(self.starts[j], anchor=(j == chosen[0]))
            self.tracks.append(t)
            self.keep(self.starts[j], vals[j], feas[j], t)
        self.confirm()

    def confirm(self):
        if (self.inc is None or self.obj.budget_exceeded
                or float(self.obj.time_left or 0) < 2):
            return
        if self.inc.loss >= getattr(self, 'confirmed_loss', np.inf) - 1e-8:
            return
        x = self.inc.x.copy()
        loss, aux = self.obj.value_aux(jnp.asarray(x))
        if bool(aux['is_feasible']) and np.isfinite(float(loss)):
            self.confirmed_loss = float(loss)
            self.keep(x, loss, True, confirm=True)
            self.confirmed = ArchivePoint(x, loss)
            self.last_confirm = float(self.obj.time_elapsed)
        else:
            print('[gwopt] incumbent confirmation failed', flush=True)
            self.inc = self.confirmed

    def descend(self, stop_frac, quiet=False):
        while (not self.obj.budget_exceeded
               and float(self.obj.budget_progress_fraction) < stop_frac):
            done = 0
            while done < len(self.tracks) and not self.obj.budget_exceeded:
                b = min(self.batch, len(self.tracks) - done)
                part = self.tracks[done:done + b]
                p = jnp.stack([t.p for t in part])
                kind = jnp.asarray([t.kind for t in part], jnp.int32)
                x = self.physical(p, kind)
                began = time.perf_counter()
                try:
                    val, grad, aux = self.obj.vmap_value_and_grad_aux(x)
                    vals = np.asarray(val)
                    feasible = np.asarray(aux['is_feasible'], bool)
                    bad = np.asarray(
                        ~jnp.isfinite(val) | ~jnp.all(jnp.isfinite(grad), axis=1))
                except Exception as exc:
                    if self.recover(exc):
                        continue
                    raise
                self.latencies.append(time.perf_counter() - began)
                if len(self.latencies) > 100:
                    del self.latencies[:-100]

                reset = []
                for j, t in enumerate(part):
                    if (feasible[j] and (vals[j] < t.best_loss
                            or self.inc is None or vals[j] < self.inc.loss)):
                        self.keep(np.asarray(x[j]), vals[j], True, t)
                    else:
                        t.last_loss = float(vals[j])
                        t.last_feasible = bool(feasible[j])
                    if bad[j]:
                        t.bad += 1
                    else:
                        t.bad = 0
                    reset.append(t.bad >= 3)

                if self.obj.budget_exceeded:
                    break

                m = jnp.stack([t.m for t in part])
                v = jnp.stack([t.v for t in part])
                count = jnp.stack([t.count for t in part])
                keys = jnp.stack([t.key for t in part])
                anch = jnp.asarray([t.anchor for t in part])
                t = min(1.0, float(self.obj.budget_progress_fraction) / .5)
                sigma = (0.0 if quiet else
                         .3 * 0.5 * (1 + np.cos(np.pi * t)))

                p, m, v, count, keys = gradient_step(
                    p, m, v, count, keys, kind, anch, grad,
                    self.span_dev, self.live_dev, sigma)

                for j, t in enumerate(part):
                    t.p, t.m, t.v = p[j], m[j], v[j]
                    t.count, t.key = count[j], keys[j]
                    if reset[j]:
                        source = (t.best_x if t.best_x is not None
                                  else (self.inc.x if self.inc is not None
                                        else self.starts[0]))
                        self.replace_track(t, source, kind=0)
                        self.failures += 1
                        if self.failures > 24:
                            raise FloatingPointError(
                                'too many nonfinite trajectory restarts')
                done += b
            if done == 0:
                break
            self.steps += 1
            if self.steps % 200 == 0:
                self.confirm()

    def basins(self):
        pool = [q.x for q in self.archive]
        if not pool:
            pool = [t.best_x for t in self.tracks if t.best_x is not None]
        if not pool:
            pool = [self.starts[0]]
        picked = []
        for x in pool:
            if all(self.distance(x, old) > .03 for old in picked):
                picked.append(np.asarray(x))
            if len(picked) == 6:
                break
        out = []
        for i in range(18):
            base = picked[i % len(picked)]
            x = base.copy()
            is_anchor = i < len(picked)
            if not is_anchor:
                u = (x - self.lo) / self.safe_span
                z = logit(u) + self.rng.normal(0, .2, len(u))
                x = self.lo + self.span / (1 + np.exp(-np.clip(z, -700, 700)))
            out.append(self.new_track(x, anchor=is_anchor))
        self.tracks = out
        print(f'[gwopt] basins={len(picked)} t={self.obj.time_elapsed:.1f}s',
              flush=True)

    def mutate(self, x):
        u = np.clip((np.asarray(x) - self.lo) / self.safe_span, 0, 1)
        chosen = self.rng.choice(len(self.blocks), max(1, len(self.blocks) // 4),
                                 replace=False)
        for j in chosen:
            block = self.blocks[j]
            u[block] += self.rng.normal(0, .03, len(block))
        return self.lo + self.span * np.clip(u, 0, 1)

    def rescue(self):
        now = float(self.obj.time_elapsed)
        ready = [t for t in self.tracks
                 if not t.anchor
                 and now - t.born >= .1 * self.total
                 and now - t.mark_time >= .1 * self.total]
        ready.sort(key=lambda t: (
            np.isfinite(t.best_loss),
            -t.best_loss if np.isfinite(t.best_loss) else 0,
            t.last_feasible, -t.last_loss))
        changed = 0
        for t in ready[:len(self.tracks) // 3]:
            if self.rescues % 2 == 0 or not self.archive:
                x = self.donors[(16 + self.rescues // 2) % len(self.donors)]
            else:
                x = self.mutate(
                    self.archive[(self.rescues // 2) % min(6, len(self.archive))].x)
            self.replace_track(t, x)
            self.rescues += 1
            changed += 1
        print(f'[gwopt] rescue={changed} t={now:.1f}s', flush=True)

    def polish(self, event):
        if not self.archive or self.obj.budget_exceeded:
            return
        start = float(self.obj.time_elapsed)
        end = min(start + .025 * self.total, self.total - 2)
        picks = []
        for q in self.archive:
            if all(self.distance(q.x, old.x) > 1e-3 for old in picks):
                picks.append(q)
            if len(picks) == 3:
                break
        if not picks or end <= start:
            return
        share = (end - start) / len(picks)
        for i, q in enumerate(picks):
            due = min(end, start + (i + 1) * share)
            old_x, old_loss = q.x.copy(), q.loss
            u0 = np.clip((old_x - self.lo) / self.safe_span, 0, 1)
            lower = np.maximum(0, u0 - q.radius)
            upper = np.minimum(1, u0 + q.radius)
            lower[~self.live] = upper[~self.live] = .5
            best_u, best_loss = u0.copy(), old_loss

            def fun(u):
                nonlocal best_u, best_loss
                if self.obj.budget_exceeded or float(self.obj.time_elapsed) >= due:
                    raise StopPolish()
                x = self.lo + self.span * u
                val, grad, aux = self.obj.value_and_grad_aux(jnp.asarray(x))
                loss = float(val)
                feasible = bool(aux['is_feasible'])
                if not np.isfinite(loss):
                    raise FloatingPointError('nonfinite polish loss')
                if feasible:
                    self.keep(x, loss, True)
                    if loss < best_loss:
                        best_u = np.asarray(u).copy()
                        best_loss = loss
                g = np.asarray(grad, np.float64) * self.span
                if not np.all(np.isfinite(g)):
                    raise FloatingPointError('nonfinite polish gradient')
                g[~self.live] = 0
                return loss, g

            try:
                minimize(fun, u0, jac=True,
                         bounds=list(zip(lower, upper)), method='L-BFGS-B',
                         options={'maxcor': 10, 'maxls': 6, 'ftol': 1e-12,
                                  'gtol': 1e-8, 'maxfun': 10000, 'maxiter': 10000})
            except StopPolish:
                pass
            except (FloatingPointError, ValueError):
                pass

            if (best_loss < old_loss - 1e-8 and not self.obj.budget_exceeded
                    and float(self.obj.time_left or 0) > 2):
                x = self.lo + self.span * best_u
                loss, aux = self.obj.value_aux(jnp.asarray(x))
                if bool(aux['is_feasible']) and float(loss) < old_loss - 1e-8:
                    self.keep(x, loss, True, confirm=True)
                    anchors = [t for t in self.tracks if t.anchor]
                    target = (min(anchors, key=lambda t: self.distance(
                        old_x, t.best_x if t.best_x is not None
                        else np.asarray(self.physical(
                            t.p[None], jnp.asarray([t.kind]))[0])))
                        if anchors else self.tracks[0])
                    self.replace_track(target, x, kind=1, anchor=True)
                    target.best_x = x.copy()
                    target.best_loss = float(loss)
                    target.mark_loss = float(loss)
                    target.mark_time = float(self.obj.time_elapsed)
                    edge = np.any(self.live & (
                        (best_u <= lower + 1e-5) | (best_u >= upper - 1e-5)))
                    radius = min(.25, q.radius * 2) if edge else q.radius
                    nearest = min(self.archive,
                                  key=lambda item: self.distance(item.x, x))
                    nearest.radius = radius
                else:
                    q.radius = max(1e-3, q.radius / 2)
            else:
                q.radius = max(1e-3, q.radius / 2)
            if self.obj.budget_exceeded or float(self.obj.time_elapsed) >= end:
                break

    def finish(self):
        for t in self.tracks:
            if t.anchor and t.best_x is not None:
                self.replace_track(t, t.best_x, kind=t.kind, anchor=True)
        reserve = (max(10.0, 2 * np.percentile(self.latencies, 95))
                   if self.latencies else 10.0)
        self.descend(min(1.0, 1 - reserve / self.total), quiet=True)
        self.confirm()

    def run(self):
        self.ready()
        self.obj.start_logging()
        self.screen()
        self.descend(.25)
        if self.obj.budget_exceeded:
            return
        self.basins()
        events = ((.35, 'polish'), (.4, 'rescue'), (.55, 'polish'),
                  (.6, 'rescue'), (.75, 'polish'), (.8, 'rescue'), (.9, 'polish'))
        for at, kind in events:
            self.descend(at)
            if self.obj.budget_exceeded:
                return
            if kind == 'rescue':
                self.rescue()
            else:
                self.polish(at)
        self.finish()
