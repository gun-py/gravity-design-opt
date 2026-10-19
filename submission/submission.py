import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm
from warmstart import Prior, build_starts, find_prior
from helpers import (Tracker, make_adam_step, reset_adam_rows, select_basins,
                     linear_anneal, confirm_best, clean_nans)

SCREEN_SIZE = 36
POPULATION = 19
NUM_BASINS = 3
REPS_PER_BASIN = 6
EXPLORE_END = 0.25
EXPLOIT_END = 0.875


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

        prior = Prior(find_prior())
        candidates = np.asarray(build_starts(obj, k=SCREEN_SIZE, prior=prior,
                                             rng_seed=int(seed)))

        step_fn, opt = make_adam_step(lr, grad_clip=1.0, noise_cap=noise_cap)

        obj.warmup_vmap_value(batch_size=SCREEN_SIZE)
        obj.warmup_vmap_value_and_grad_aux(batch_size=POPULATION)
        obj.warmup_vmap_value_and_grad_aux(batch_size=NUM_BASINS * REPS_PER_BASIN)
        if getattr(obj, 'batch', 1) > 1:
            obj.warmup_vmap_value(batch_size=1)
            obj.warmup_vmap_value_and_grad_aux(batch_size=1)
        obj.start_logging()

        tracker = Tracker()

        # --- phase 1: screen candidates by value only ---
        screen_vals = np.asarray(obj.vmap_value(jnp.asarray(candidates)))
        screen_vals = np.where(np.isfinite(screen_vals), screen_vals, np.inf)
        population = jnp.asarray(candidates[np.argsort(screen_vals)[:POPULATION]])
        state = opt.init(population)
        personal_loss = np.full(POPULATION, np.inf)
        personal_x = np.asarray(population).copy()

        # --- phase 2: explore with noisy gradient descent ---
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

            sigma = linear_anneal(float(obj.budget_progress_fraction),
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

        # --- phase 3: form basins from best solutions ---
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

        # --- phase 4: exploit basins with periodic reinjection ---
        while (not obj.budget_exceeded
               and float(obj.budget_progress_fraction) < EXPLOIT_END):
            vals, grads, aux = obj.vmap_value_and_grad_aux(population)
            vals, grads = clean_nans(vals, grads)
            tracker.update(np.asarray(population), np.asarray(vals),
                           np.asarray(aux["is_feasible"]))

            sigma = linear_anneal(float(obj.budget_progress_fraction),
                                  initial_noise, 0.0, anneal_frac)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, population.shape) * sigma
            for b in range(NUM_BASINS):
                anchor_idx = b * REPS_PER_BASIN
                if anchor_idx < population.shape[0]:
                    noise = noise.at[anchor_idx].set(0.0)
            population, state = step_fn(population, state, grads, noise)

            if tracker.iter % 300 == 0 and tracker.iter > 0:
                anchor = tracker.anchor()
                if anchor is not None:
                    pop_np = np.asarray(population).copy()
                    v_np = np.asarray(vals)
                    for b in range(NUM_BASINS):
                        sl = slice(b * REPS_PER_BASIN,
                                   min((b + 1) * REPS_PER_BASIN, pop_np.shape[0]))
                        if sl.stop - sl.start <= 1:
                            continue
                        worst = int(np.argmax(v_np[sl][1:]) + 1)
                        key, subkey = jax.random.split(key)
                        jitter = np.asarray(jax.random.normal(subkey, (n,))) * 0.2
                        pop_np[sl.start + worst] = anchor + jitter
                        state = reset_adam_rows(state, [sl.start + worst])
                    population = jnp.asarray(pop_np)

            if tracker.iter % confirm_every == 0 and tracker.iter > 0:
                confirm_best(obj, tracker)

        # --- phase 5: L-BFGS polish on best solution ---
        anchor = tracker.anchor()
        if anchor is not None and not obj.budget_exceeded:
            self._polish(obj, anchor, tracker)
            if (tracker.best_feasible_x is not None
                    and not obj.budget_exceeded):
                population = population.at[0].set(
                    jnp.asarray(tracker.best_feasible_x))
                state = reset_adam_rows(state, [0])

        # --- phase 6: quiet descent to finish ---
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

    def _polish(self, obj, x0, tracker):
        raw_fn = obj.value_function_aux(unbounded=True)
        if raw_fn is None:
            return

        def loss_only(xi):
            return raw_fn(xi)[0]

        vg = jax.value_and_grad(raw_fn, has_aux=True)
        opt = optax.lbfgs()
        x = jnp.asarray(x0)
        state = opt.init(x)

        @jax.jit
        def step(point, state):
            (loss, aux), grad = vg(point)
            update, new_state = opt.update(
                grad, state, point, value=loss, grad=grad,
                value_fn=loss_only)
            return optax.apply_updates(point, update), new_state, loss, grad, aux

        stop_frac = min(0.985, float(obj.budget_progress_fraction) + 0.06)
        deadline = (0.985 * float(obj.max_time)
                    if obj.max_time else float('inf'))
        entered = float(obj.time_elapsed)
        update_deadline = max(entered, deadline - 10.0)

        try:
            while (not obj.budget_exceeded
                   and float(obj.budget_progress_fraction) < stop_frac
                   and float(obj.time_elapsed) < update_deadline):
                prev = x
                x, state, loss, grad, aux = step(x, state)
                jax.tree_util.tree_map(
                    lambda leaf: (leaf.block_until_ready()
                                  if hasattr(leaf, 'block_until_ready')
                                  else leaf),
                    (x, loss, grad, aux))
                if float(obj.time_elapsed) >= deadline:
                    break
                if (not np.all(np.isfinite(np.asarray(prev)))
                        or not np.all(np.isfinite(np.asarray(grad)))
                        or not np.isfinite(float(loss))):
                    break
                before = int(obj.eval_count)
                obj.log_evaluation(params=prev, loss=loss, grad=grad, aux=aux)
                if (int(obj.eval_count) > before
                        and bool(aux['is_feasible'])):
                    val = float(loss)
                    if val < tracker.best_feasible_loss:
                        tracker.best_feasible_loss = val
                        tracker.best_feasible_x = np.asarray(prev).copy()
                tracker.iter += 1
                if not np.all(np.isfinite(np.asarray(x))):
                    break
        except Exception:
            pass
