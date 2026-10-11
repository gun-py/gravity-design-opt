"""Budget-scheduled, basin-preserving restarts for the rescue ablation."""

import numpy as np


CHECKS = (0.40, 0.55, 0.70, 0.82)
SEPARATION = 0.03
MEANINGFUL_GAIN = 1e-4


def _unit(x):
    v = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-v))


def _distance(a, b):
    return float(np.sqrt(np.mean((_unit(a) - _unit(b)) ** 2)))


def _violations(aux, thresholds, losses, n_rows):
    power = aux.get('power_values') if isinstance(aux, dict) else None
    if isinstance(power, dict) and isinstance(thresholds, dict):
        parts = []
        for group in ('hard', 'soft', 'detector'):
            threshold = thresholds.get(group)
            if group not in power or threshold is None or threshold <= 0:
                break
            values = np.asarray(power[group], dtype=np.float64)
            if values.shape[0] != n_rows:
                break
            parts.append(np.maximum(values.reshape(n_rows, -1) / threshold - 1.0, 0.0))
        if len(parts) == 3:
            violation = np.max(np.concatenate(parts, axis=1), axis=1)
            return np.where(np.isfinite(violation), violation, np.inf), 'power'
    return np.where(np.isfinite(losses), losses, np.inf), 'penalized_loss'


class BasinRescue:
    def __init__(self, n_params, budget_seconds, start_seconds, seed, thresholds=None):
        self.budget_seconds = float(budget_seconds)
        self.thresholds = thresholds
        self.rng = np.random.default_rng(int(seed) ^ 0x6A09E667)
        self.best_loss = np.full(18, np.inf)
        self.best_x = [None] * 18
        self.lowest_violation = np.full(18, np.inf)
        self.lowest_x = [None] * 18
        self.born = np.full(18, float(start_seconds))
        self.last_gain = self.born.copy()
        self.generation = np.zeros(18, dtype=np.int32)
        self.basin_loss = np.full(3, np.inf)
        self.basin_x = [None] * 3
        self.basin_violation = np.full(3, np.inf)
        self.basin_violation_x = [None] * 3
        self.archive = []
        self.next_check = 0
        self.events = []
        self.checks = []
        self.violation_source = 'penalized_loss'
        self.n_params = int(n_params)

    def record(self, points, losses, aux, elapsed):
        points = np.asarray(points, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64).reshape(-1)
        feasible = np.asarray(aux['is_feasible'], dtype=bool).reshape(-1)
        if points.shape != (18, self.n_params) or losses.shape != (18,) or feasible.shape != (18,):
            raise ValueError('rescue observations need 18 points, losses and flags')
        violation, self.violation_source = _violations(aux, self.thresholds, losses, 18)
        for row in range(18):
            if not (np.isfinite(losses[row]) and np.all(np.isfinite(points[row]))):
                continue
            basin = row // 6
            if violation[row] < self.lowest_violation[row]:
                self.lowest_violation[row] = violation[row]
                self.lowest_x[row] = points[row].copy()
            if violation[row] < self.basin_violation[basin]:
                self.basin_violation[basin] = violation[row]
                self.basin_violation_x[basin] = points[row].copy()
            if feasible[row]:
                previous = self.best_loss[row]
                if losses[row] < previous:
                    self.best_loss[row] = losses[row]
                    self.best_x[row] = points[row].copy()
                    if not np.isfinite(previous) or previous - losses[row] >= MEANINGFUL_GAIN:
                        self.last_gain[row] = float(elapsed)
                if losses[row] < self.basin_loss[basin]:
                    self.basin_loss[basin] = losses[row]
                    self.basin_x[basin] = points[row].copy()

    def _refresh_archive(self):
        pool = list(self.archive)
        pool.extend((float(loss), point) for loss, point in zip(self.best_loss, self.best_x)
                    if point is not None and np.isfinite(loss))
        pool.extend((float(loss), point) for loss, point in zip(self.basin_loss, self.basin_x)
                    if point is not None and np.isfinite(loss))
        chosen = []
        for loss, point in sorted(pool, key=lambda item: item[0]):
            if all(_distance(point, other) >= SEPARATION for _, other in chosen):
                chosen.append((loss, np.asarray(point).copy()))
                if len(chosen) == 32:
                    break
        self.archive = chosen

    def _eligible(self, basin, elapsed):
        stalled = []
        for row in range(basin * 6 + 1, basin * 6 + 6):
            if min(elapsed - self.born[row], elapsed - self.last_gain[row]) <= .08 * self.budget_seconds:
                continue
            never_feasible = not np.isfinite(self.best_loss[row])
            # Never-feasible rows take priority; within that class, row order is stable.
            severity = 0.0 if never_feasible else self.best_loss[row]
            stalled.append((0 if never_feasible else 1, -severity, row))
        return min(stalled)[2] if stalled else None

    def maybe_rescue(self, points, progress, elapsed):
        due = []
        while self.next_check < len(CHECKS) and progress >= CHECKS[self.next_check]:
            due.append(self.next_check)
            self.next_check += 1
        if not due:
            return points, []
        check_index = due[-1]  # Missed checks do not trigger several back-to-back restarts.
        self._refresh_archive()
        eligible = [(basin, self._eligible(basin, elapsed)) for basin in range(3)]
        eligible = [(basin, row) for basin, row in eligible if row is not None]
        if not eligible:
            self.checks.append({'check_fraction': CHECKS[check_index],
                                'elapsed': float(elapsed), 'eligible': [],
                                'replaced': [], 'archive_size': len(self.archive)})
            return points, []
        scout_basin = eligible[check_index % len(eligible)][0]
        updated = np.asarray(points, dtype=np.float64).copy()
        replaced = []
        for basin, row in eligible:
            source = self.basin_x[basin]
            origin = 'own_basin'
            if basin == scout_basin:
                distinct = next((point for _, point in self.archive
                                 if source is None or _distance(point, source) >= SEPARATION), None)
                if distinct is not None:
                    source, origin = distinct, 'archive'
            if source is None:
                source, origin = self.basin_violation_x[basin], 'least_violating'
            if source is None:
                continue
            updated[row] = source + self.rng.normal(0.0, 0.2, self.n_params)
            previous = self.best_loss[row]
            self.best_loss[row] = np.inf
            self.best_x[row] = None
            self.lowest_violation[row] = np.inf
            self.lowest_x[row] = None
            self.born[row] = self.last_gain[row] = float(elapsed)
            self.generation[row] += 1
            replaced.append(row)
            self.events.append({'check_fraction': CHECKS[check_index], 'elapsed': float(elapsed),
                                'basin': basin, 'row': row, 'source': origin,
                                'previous_feasible_loss': float(previous) if np.isfinite(previous) else None,
                                'generation': int(self.generation[row]),
                                'violation_source': self.violation_source})
        self.checks.append({'check_fraction': CHECKS[check_index], 'elapsed': float(elapsed),
                            'eligible': [row for _, row in eligible], 'replaced': replaced,
                            'archive_size': len(self.archive), 'scout_basin': scout_basin})
        return updated, replaced
