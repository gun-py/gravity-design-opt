import os
import sys
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from dfbench import OptimizationAlgorithm
from warmstart import Prior, build_starts, find_prior, _get_param_keys, _get_bounds, _to_logit, _fill_from_donor, _nearest_donors
from topology import signature
from helpers import (Tracker, make_adam_step, reset_adam_rows, select_basins,
                     cosine_anneal, confirm_best, clean_nans,
                     param_groups, crossover)

SCREEN_SIZE = 36
POPULATION = 19
NUM_BASINS = 3
REPS_PER_BASIN = 6
EXPLORE_END = 0.25
EXPLOIT_END = 0.875


class StopPolish(Exception):
    pass


class GWOptimizer(OptimizationAlgorithm):
    algorithm_str = "gwopt"

    def __init__(self):
        pass

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, initial_noise=0.3, noise_cap=0.25,
                 anneal_frac=0.5, confirm_every=200, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        n = obj.n_params
        bounds = np.asarray(obj.bounds, np.float64)
        lo, hi = bounds[0], bounds[1]
        span = hi - lo
        live = span > 0
        safe_span = np.where(live, span, 1.0)

        prior = Prior(find_prior())
        candidates = np.asarray(build_starts(obj, k=SCREEN_SIZE, prior=prior,
                                             rng_seed=int(seed)))
        rng = np.random.default_rng(int(seed))

        param_keys = None
        blocks = None
        try:
            param_keys = [(re.sub(r'sus$', '', name), prop) for name, prop in
                          [(lambda p: (p[0], p[1]) if len(p) == 2
                            else (p[0][0], p[0][1]))(
                              [p] if isinstance(p, str) else p)
                           for p in obj.optimization_pairs]]
            blocks = param_groups([(name, prop) for name, prop in param_keys])
        except Exception:
            pass

        step_fn, opt = make_adam_step(lr, grad_clip=1.0, noise_cap=noise_cap)

        obj.warmup_vmap_value(batch_size=SCREEN_SIZE)
        obj.warmup_vmap_value_and_grad_aux(batch_size=POPULATION)
        obj.warmup_vmap_value_and_grad_aux(batch_size=NUM_BASINS * REPS_PER_BASIN)
        if getattr(obj, 'batch', 1) > 1:
            obj.warmup_vmap_value(batch_size=1)
            obj.warmup_vmap_value_and_grad_aux(batch_size=1)
        obj.start_logging()

        tracker = Tracker()

        # --- screen candidates ---
        screen_vals = np.asarray(obj.vmap_value(jnp.asarray(candidates)))
        screen_vals = np.where(np.isfinite(screen_vals), screen_vals, np.inf)

        rank = np.argsort(screen_vals)
        chosen = list(rank[:min(12, len(rank))])
        rest = [int(k) for k in rank if k not in chosen]
        while rest and len(chosen) < POPULATION:
            best_k = max(rest, key=lambda j: min(
                np.sqrt(np.mean((jax.nn.sigmoid(jnp.asarray(candidates[j]))
                                 - jax.nn.sigmoid(jnp.asarray(candidates[a])))**2))
                for a in chosen))
            chosen.append(best_k)
            rest.remove(best_k)

        population = jnp.asarray(candidates[chosen])
        state = opt.init(population)
        personal_loss = np.full(POPULATION, np.inf)
        personal_x = np.asarray(population).copy()

        # --- explore with noisy gradient descent ---
        while (not obj.budget_exceeded
               and float(obj.budget_progress_fraction) < EXPLORE_END):
            vals, grads, aux = obj.vmap_value_and_grad_aux(population)
            vals, grads = clean_nans(vals, grads)
            points = np.asarray(population)
            feasible = np.asarray(aux["is_feasible"])
            tracker.update(points, np.asarray(vals), feasible)

            improved = (feasible & np.isfinite(np.asarray(vals))
                        & (np.asarray(vals) < personal_loss))
            personal_loss[improved] = np.asarray(vals)[improved]
            personal_x[improved] = points[improved]

            sigma = cosine_anneal(float(obj.budget_progress_fraction),
                                  initial_noise, 0.0, anneal_frac)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, population.shape) * sigma
            noise = noise.at[0].set(0.0)
            population, state = step_fn(population, state, grads, noise)

            if tracker.iter % confirm_every == 0 and tracker.iter > 0:
                confirm_best(obj, tracker)

        if obj.budget_exceeded:
            confirm_best(obj, tracker)
            return

        # --- form basins ---
        vals, _, aux = obj.vmap_value_and_grad_aux(population)
        v_np = np.asarray(vals)
        f_np = np.asarray(aux["is_feasible"])
        score = np.where(f_np, v_np, v_np + 10.0)
        basins = select_basins(personal_x, personal_loss,
                               np.asarray(population), score, NUM_BASINS)

        basin_pop = np.empty((NUM_BASINS * REPS_PER_BASIN, n), np.float64)
        for b in range(NUM_BASINS):
            key, subkey = jax.random.split(key)
            noise = np.asarray(jax.random.normal(subkey, (REPS_PER_BASIN, n))) * 0.2
            noise[0] = 0.0
            basin_pop[b * REPS_PER_BASIN:(b + 1) * REPS_PER_BASIN] = basins[b] + noise
        population = jnp.asarray(basin_pop)
        state = opt.init(population)

        # --- exploit basins ---
        while (not obj.budget_exceeded
               and float(obj.budget_progress_fraction) < EXPLOIT_END):
            vals, grads, aux = obj.vmap_value_and_grad_aux(population)
            vals, grads = clean_nans(vals, grads)
            tracker.update(np.asarray(population), np.asarray(vals),
                           np.asarray(aux["is_feasible"]))

            sigma = cosine_anneal(float(obj.budget_progress_fraction),
                                  initial_noise, 0.0, anneal_frac)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, population.shape) * sigma
            for b in range(NUM_BASINS):
                anchor_idx = b * REPS_PER_BASIN
                if anchor_idx < population.shape[0]:
                    noise = noise.at[anchor_idx].set(0.0)
            population, state = step_fn(population, state, grads, noise)

            if tracker.iter % 300 == 0 and tracker.iter > 0:
                pop_np = np.asarray(population).copy()
                v_np = np.asarray(vals)
                replaced = []
                for b in range(NUM_BASINS):
                    sl = slice(b * REPS_PER_BASIN,
                               min((b + 1) * REPS_PER_BASIN, pop_np.shape[0]))
                    if sl.stop - sl.start <= 1:
                        continue
                    worst = int(np.argmax(v_np[sl][1:]) + 1)
                    idx = sl.start + worst

                    if tracker.archive and blocks and len(tracker.archive) >= 2:
                        parent_a = tracker.archive[0][0]
                        parent_b = tracker.archive[min(1, len(tracker.archive) - 1)][0]
                        pop_np[idx] = crossover(parent_a, parent_b, blocks, rng)
                    elif tracker.anchor() is not None:
                        key, subkey = jax.random.split(key)
                        jitter = np.asarray(jax.random.normal(subkey, (n,))) * 0.2
                        pop_np[idx] = tracker.anchor() + jitter
                    replaced.append(idx)

                if replaced:
                    state = reset_adam_rows(state, replaced)
                    population = jnp.asarray(pop_np)

            if tracker.iter % confirm_every == 0 and tracker.iter > 0:
                confirm_best(obj, tracker)

        # --- L-BFGS-B polish in bounded unit space ---
        if not obj.budget_exceeded:
            self._polish_bounded(obj, tracker, lo, hi, span, live)
            if (tracker.best_feasible_x is not None
                    and not obj.budget_exceeded):
                population = population.at[0].set(
                    jnp.asarray(tracker.best_feasible_x))
                state = reset_adam_rows(state, [0])

        # --- quiet descent to finish ---
        while not obj.budget_exceeded:
            vals, grads, aux = obj.vmap_value_and_grad_aux(population)
            vals, grads = clean_nans(vals, grads)
            tracker.update(np.asarray(population), np.asarray(vals),
                           np.asarray(aux["is_feasible"]))
            population, state = step_fn(population, state, grads,
                                        jnp.zeros_like(population))
            if tracker.iter % 50 == 0 and tracker.iter > 0:
                confirm_best(obj, tracker)
        confirm_best(obj, tracker)

    def _polish_bounded(self, obj, tracker, lo, hi, span, live):
        safe_span = np.where(live, span, 1.0)

        picks = []
        if tracker.best_feasible_x is not None:
            picks.append(tracker.best_feasible_x)
        for ax, al in tracker.archive[:2]:
            if not any(np.array_equal(ax, p) for p in picks):
                picks.append(ax)
        if not picks:
            return

        stop_frac = min(0.985, float(obj.budget_progress_fraction) + 0.06)
        deadline = (0.985 * float(obj.max_time)
                    if obj.max_time else float('inf'))
        entered = float(obj.time_elapsed)
        update_deadline = max(entered, deadline - 10.0)
        share = (update_deadline - entered) / len(picks)

        for i, z in enumerate(picks):
            due = min(update_deadline, entered + (i + 1) * share)
            x_phys = lo + span / (1 + np.exp(-np.clip(np.asarray(z), -700, 700)))
            u0 = np.clip((x_phys - lo) / safe_span, 0, 1)
            radius = 0.05
            lower = np.maximum(0, u0 - radius)
            upper = np.minimum(1, u0 + radius)
            lower[~live] = upper[~live] = 0.5
            best_u, best_loss = u0.copy(), float('inf')

            def fun(u):
                nonlocal best_u, best_loss
                if obj.budget_exceeded or float(obj.time_elapsed) >= due:
                    raise StopPolish()
                x = lo + span * u
                val, grad, aux = obj.value_and_grad_aux(jnp.asarray(x, jnp.float64))
                loss = float(val)
                if not np.isfinite(loss):
                    raise FloatingPointError('nonfinite')
                if bool(aux['is_feasible']) and loss < best_loss:
                    best_u = np.asarray(u).copy()
                    best_loss = loss
                g = np.asarray(grad, np.float64) * span
                g = np.where(np.isfinite(g), g, 0.0)
                g[~live] = 0
                return loss, g

            try:
                minimize(fun, u0, jac=True, bounds=list(zip(lower, upper)),
                         method='L-BFGS-B',
                         options={'maxcor': 10, 'maxls': 10, 'ftol': 1e-12,
                                  'gtol': 1e-8, 'maxiter': 10000})
            except StopPolish:
                pass
            except (FloatingPointError, ValueError):
                pass

            if best_loss < float('inf') and not obj.budget_exceeded:
                x_best = lo + span * best_u
                z_best = np.log(np.clip(best_u, 1e-12, 1 - 1e-12)
                                / (1 - np.clip(best_u, 1e-12, 1 - 1e-12)))
                try:
                    loss, aux = obj.value_aux(jnp.asarray(x_best, jnp.float64))
                    if bool(aux['is_feasible']) and float(loss) < tracker.best_feasible_loss:
                        tracker.best_feasible_loss = float(loss)
                        tracker.best_feasible_x = z_best.copy()
                except Exception:
                    pass

            if obj.budget_exceeded or float(obj.time_elapsed) >= update_deadline:
                break
