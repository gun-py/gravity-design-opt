import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from dfbench import OptimizationAlgorithm

from warm import Prior, build_starts, find_prior_path
from core import (Tracker, make_adam_step, make_independent_adam_step,
                  reset_adam_rows, select_basins, anneal, late_lr_scale, insure_incumbent,
                  nan_guard, run_to_wire)
from rescue import BasinRescue

SCREEN, GRAD, BASINS, REPS = 36, 19, 3, 6
END_B, END_C = 0.25, 0.875
VARIANT = "control"  # Packaging replaces this exact line for each isolated ZIP.


class ChampionV3(OptimizationAlgorithm):
    def __init__(self):
        pass

    algorithm_str = "champion_v3_" + VARIANT

    def optimize(self, objective, init_params=None, random_seed=None,
                 lr=0.1, s0=0.3, cap_noise=0.25, anneal_frac=0.5, insure_every=200,
                 features=None, **kw):
        obj = objective
        seed, key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        n = obj.n_params
        if features is not None:
            raise ValueError('experimental features are fixed in the archive')
        if VARIANT not in {'control', 'rescue', 'late_lr', 'polish'}:
            raise ValueError('unknown packaged Champion variant')
        enabled = {'screen', 'basin', 'state', 'reinject'}
        self.diagnostics = {'variant': VARIANT, 'phase_end_seconds': {},
                            'phase_best_feasible': {}, 'rescue_events': [],
                            'rescue_checks': [], 'polish': {}}

        def snapshot(name):
            self.diagnostics['phase_end_seconds'][name] = float(obj.time_elapsed)
            self.diagnostics['phase_best_feasible'][name] = (
                float(tr.best_feasible_loss) if np.isfinite(tr.best_feasible_loss) else None)

        def scale():
            return late_lr_scale(obj.budget_progress_fraction) if VARIANT == 'late_lr' else 1.0

        prior = Prior(find_prior_path())
        cand = np.asarray(build_starts(obj, k=SCREEN, prior=prior, rng_seed=int(seed)))

        step, opt = ((make_independent_adam_step if 'state' in enabled else make_adam_step)
                     (lr, clip_grad=1.0, cap_noise_frac=cap_noise))

        obj.warmup_vmap_value(batch_size=SCREEN)
        obj.warmup_vmap_value_and_grad_aux(batch_size=GRAD)
        obj.warmup_vmap_value_and_grad_aux(batch_size=BASINS * REPS)
        obj.start_logging()

        tr = Tracker()

        # A: value-only screen
        try:
            first = np.asarray(obj.vmap_value(jnp.asarray(cand)))
            v = np.where(np.isfinite(first), first, np.inf)
            if 'screen' not in enabled:
                second = np.asarray(obj.vmap_value(jnp.asarray(cand)))
                v = np.where(np.isfinite(first), second, np.inf)
            x = jnp.asarray(cand[np.argsort(v)[:GRAD]])
        except Exception:
            x = jnp.asarray(cand[:GRAD])
        snapshot('screen')
        st = opt.init(x)
        personal_loss = np.full(GRAD, np.inf)
        personal_x = np.asarray(x).copy()

        # B: 19-wide capped noise
        def iter_b(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            values = np.asarray(v)
            points = np.asarray(x)
            feasible = np.asarray(aux["is_feasible"])
            tr.update(points, values, feasible)
            if 'basin' in enabled:
                candidate = np.where(feasible & np.isfinite(values), values, np.inf)
                improved = candidate < personal_loss
                personal_loss[improved] = candidate[improved]
                personal_x[improved] = points[improved]
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma; nz = nz.at[0].set(0.0)
            x, st = step(x, st, g, nz, scale())

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < END_B:
            try: iter_b(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)
        snapshot('search_b')

        if obj.budget_exceeded:
            insure_incumbent(obj, tr); return

        # C: replicate 3 best basins
        rescue = None
        try:
            v_l, _, aux_l = obj.vmap_value_and_grad_aux(x)
            vl, fl = np.asarray(v_l), np.asarray(aux_l["is_feasible"])
            score = np.where(fl, vl, vl + 10.0)
            basins = (select_basins(personal_x, personal_loss, np.asarray(x), score, BASINS)
                      if 'basin' in enabled else np.asarray(x)[np.argsort(score)[:BASINS]])

            xc = np.empty((BASINS * REPS, n), np.float64)
            for b in range(BASINS):
                key, sk = jax.random.split(key)
                noise = np.asarray(jax.random.normal(sk, (REPS, n))) * 0.2
                noise[0] = 0.0
                xc[b * REPS:(b + 1) * REPS] = basins[b] + noise
            x = jnp.asarray(xc); st = opt.init(x)
            if VARIANT == 'rescue':
                rescue = BasinRescue(n, float(obj.max_time), float(obj.time_elapsed),
                                     int(seed), thresholds=obj.power_thresholds)
        except Exception:
            pass

        def iter_c(_):
            nonlocal x, st, key
            v, g, aux = obj.vmap_value_and_grad_aux(x)
            if rescue is not None:
                rescue.record(np.asarray(x), np.asarray(v), aux, float(obj.time_elapsed))
            v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            sigma = anneal(float(obj.budget_progress_fraction), s0, 0.0, anneal_frac)
            key, sk = jax.random.split(key)
            nz = jax.random.normal(sk, x.shape) * sigma
            for b in range(BASINS):
                if b * REPS < x.shape[0]: nz = nz.at[b * REPS].set(0.0)
            x, st = step(x, st, g, nz, scale())

            if rescue is not None:
                updated, replaced = rescue.maybe_rescue(np.asarray(x),
                                    float(obj.budget_progress_fraction), float(obj.time_elapsed))
                self.diagnostics['rescue_checks'] = list(rescue.checks)
                if replaced:
                    x = jnp.asarray(updated)
                    st = reset_adam_rows(st, replaced)
                    self.diagnostics['rescue_events'] = list(rescue.events)
            elif tr.iter % 300 == 0 and tr.iter > 0 and tr.anchor() is not None:
                a = tr.anchor(); xn = np.asarray(x).copy(); vn = np.asarray(v)
                for b in range(BASINS):
                    sl = slice(b * REPS, min((b + 1) * REPS, xn.shape[0]))
                    if sl.stop - sl.start <= 1: continue
                    j = int(np.argmax(vn[sl][1:]) + 1)
                    key, sk = jax.random.split(key)
                    xn[sl.start + j] = a + np.asarray(jax.random.normal(sk, (n,))) * 0.2
                    if 'state' in enabled:
                        st = reset_adam_rows(st, [sl.start + j])
                x = jnp.asarray(xn)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < END_C:
            try: iter_c(tr.iter)
            except Exception: pass
            if tr.iter % insure_every == 0 and tr.iter > 0: insure_incumbent(obj, tr)
        snapshot('search_c')

        # D: L-BFGS polish on incumbent, then noise=0 descent to the wire
        polish_start = float(obj.time_elapsed)
        polish_before = tr.best_feasible_loss
        try:
            a = tr.anchor()
            if a is not None and not obj.budget_exceeded:
                before = tr.best_feasible_loss
                self._polish(obj, a, tr)
                if ('reinject' in enabled and tr.best_feasible_x is not None and
                        tr.best_feasible_loss < before and not obj.budget_exceeded):
                    x = x.at[0].set(jnp.asarray(tr.best_feasible_x))
                    if 'state' in enabled:
                        st = reset_adam_rows(st, [0])
                    else:
                        st = opt.init(x)
        except Exception:
            pass
        self.diagnostics['polish'].update({
            'elapsed_seconds': float(obj.time_elapsed) - polish_start,
            'before_loss': float(polish_before) if np.isfinite(polish_before) else None,
            'after_loss': float(tr.best_feasible_loss) if np.isfinite(tr.best_feasible_loss) else None,
            'gain': (float(polish_before - tr.best_feasible_loss)
                     if np.isfinite(polish_before) and np.isfinite(tr.best_feasible_loss) else None)})
        snapshot('polish')

        def iter_d(_):
            nonlocal x, st
            v, g, aux = obj.vmap_value_and_grad_aux(x); v, g = nan_guard(v, g)
            tr.update(np.asarray(x), np.asarray(v), np.asarray(aux["is_feasible"]))
            x, st = step(x, st, g, jnp.zeros_like(x), scale())

        run_to_wire(obj, iter_d, tr, insure_every=50)
        snapshot('search_d')

    def _polish(self, obj, x0, tr):
        if VARIANT == 'polish':
            return self._polish_efficient(obj, x0, tr)
        f_aux = obj.value_function_aux(unbounded=True)
        if f_aux is None: return

        def loss_only(xi):
            l, _ = f_aux(xi); return l
        vg = jax.value_and_grad(loss_only)
        opt = optax.lbfgs(); x = jnp.asarray(x0); st = opt.init(x)

        @jax.jit
        def step(x, st):
            l, g = vg(x)
            u, st2 = opt.update(g, st, x, value=l, grad=g, value_fn=loss_only)
            return optax.apply_updates(x, u), st2, l, g

        step(x, st)
        aux_keys = ("sensitivity_loss", "penalty", "is_feasible", "power_values", "violations")
        stop = min(0.985, float(obj.budget_progress_fraction) + 0.06)

        while not obj.budget_exceeded and float(obj.budget_progress_fraction) < stop:
            xp = x; x, st, loss, g = step(x, st)
            try:
                _, aux = f_aux(xp)
                obj.log_evaluation(params=xp, loss=loss, grad=g, aux={kk: aux[kk] for kk in aux_keys})
                if bool(aux["is_feasible"]) and float(loss) < tr.best_feasible_loss:
                    tr.best_feasible_loss = float(loss); tr.best_feasible_x = np.asarray(xp)
            except Exception:
                pass
            tr.iter += 1

    def _polish_efficient(self, obj, x0, tr):
        f_aux = obj.value_function_aux(unbounded=True)
        if f_aux is None or obj.max_time is None:
            return
        entered = float(obj.time_elapsed)
        deadline = min(.985 * float(obj.max_time), entered + .06 * float(obj.max_time))
        update_deadline = max(entered, deadline - 10.0)

        def loss_only(xi):
            return f_aux(xi)[0]

        vg = jax.value_and_grad(f_aux, has_aux=True)
        opt = optax.lbfgs()
        x = jnp.asarray(x0)
        state = opt.init(x)

        @jax.jit
        def step(point, state):
            (loss, aux), grad = vg(point)
            update, next_state = opt.update(grad, state, point, value=loss,
                                            grad=grad, value_fn=loss_only)
            return optax.apply_updates(point, update), next_state, loss, grad, aux

        count = 0
        next_unconfirmed = None
        failed = False
        try:
            while not obj.budget_exceeded and float(obj.time_elapsed) < update_deadline:
                xp = x
                x, state, loss, grad, aux = step(x, state)
                jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready()
                                       if hasattr(leaf, 'block_until_ready') else leaf,
                                       (x, loss, grad, aux))
                if (not np.all(np.isfinite(np.asarray(xp))) or
                        not np.all(np.isfinite(np.asarray(grad))) or
                        not np.isfinite(float(loss))):
                    self.diagnostics['polish']['stop_reason'] = 'nonfinite'
                    failed = True
                    break
                if obj.budget_exceeded:
                    self.diagnostics['polish']['stop_reason'] = 'budget_exceeded'
                    break
                before_count = int(obj.eval_count)
                obj.log_evaluation(params=xp, loss=loss, grad=grad, aux=aux)
                if int(obj.eval_count) > before_count and bool(aux['is_feasible']):
                    value = float(loss)
                    if value < tr.best_feasible_loss:
                        tr.best_feasible_loss = value
                        tr.best_feasible_x = np.asarray(xp).copy()
                tr.iter += 1
                count += 1
                if not np.all(np.isfinite(np.asarray(x))):
                    self.diagnostics['polish']['stop_reason'] = 'nonfinite_proposal'
                    failed = True
                    break
                next_unconfirmed = x
        except Exception as error:
            self.diagnostics['polish']['stop_reason'] = f'{type(error).__name__}: {error}'[:240]
            failed = True
        self.diagnostics['polish']['steps'] = count
        if (not failed and next_unconfirmed is not None and not obj.budget_exceeded and
                float(obj.time_elapsed) < deadline):
            try:
                before_count = int(obj.eval_count)
                loss, aux = obj.value_aux(next_unconfirmed)
                if int(obj.eval_count) > before_count and bool(aux['is_feasible']):
                    value = float(loss)
                    if np.isfinite(value) and value < tr.best_feasible_loss:
                        tr.best_feasible_loss = value
                        tr.best_feasible_x = np.asarray(next_unconfirmed).copy()
                self.diagnostics['polish']['terminal_confirmation'] = True
            except Exception as error:
                self.diagnostics['polish']['confirmation_error'] = f'{type(error).__name__}: {error}'[:240]
