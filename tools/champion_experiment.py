"""Staged, resumable Champion V3 screening under one global wall-clock budget."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

from colab_20gb_compare import (check_archive, gpu_card, pool_size, probe_gpu,
                                run_environment, run_method, save_json, seed_pair,
                                sha256, versions)


VARIANTS = ('rescue', 'late_lr', 'polish')
STAGES = ('discovery', 'cross_check', 'validation')


def valid(outcome):
    if outcome is None:
        return False
    row, guard = outcome
    return (row.get('status') == 'complete' and guard.get('status') == 'within_cap'
            and guard.get('process_exit_code') == 0
            and row.get('best_feasible_loss') is not None)


def gain(outcomes, stage, variant):
    control, trial = outcomes.get((stage, 'control')), outcomes.get((stage, variant))
    if not (valid(control) and valid(trial)):
        return None
    return float(control[0]['best_feasible_loss']) - float(trial[0]['best_feasible_loss'])


def discover(outcomes):
    wins = [(name, gain(outcomes, 'discovery', name)) for name in VARIANTS]
    return [name for name, value in sorted(wins, key=lambda item:
            (-item[1] if item[1] is not None else float('inf'), item[0]))
            if value is not None and value > 0][:2]


def select(outcomes, advancing):
    candidates = []
    for name in advancing:
        first = gain(outcomes, 'discovery', name)
        second = gain(outcomes, 'cross_check', name)
        if first is None or second is None or first <= 0 or second <= 0:
            continue
        mean = (first + second) / 2
        if mean >= .02:
            candidates.append((name, mean, min(first, second)))
    return (sorted(candidates, key=lambda item: (-item[1], -item[2], item[0]))[0][0]
            if candidates else None)


def resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_plan(path, args):
    spec = json.loads(path.read_text())
    root = path.resolve().parents[1]
    if (spec.get('version') != 1 or set(spec.get('archives', {})) != {'control', *VARIANTS}
            or set(spec.get('stages', {})) != set(STAGES)):
        raise ValueError('experiment plan requires version 1, four archives and three stages')
    if args.minutes != 60 or args.startup_allowance_minutes != 15 or args.cap_mib != 19000:
        raise ValueError('experiment requires 60 scored minutes, 15 startup minutes and 19000 MiB cap')
    if args.pool_limit_mib != 16000 or args.max_wall_hours > 12 or args.max_wall_hours <= 0:
        raise ValueError('experiment requires a 16000 MiB pool limit and at most 12 wall hours')
    archives = {name: check_archive(resolve(root, entry)) for name, entry in spec['archives'].items()}
    stages = {}
    expected = {'discovery': 8, 'cross_check': 4, 'validation': 5}
    for name in STAGES:
        entry = spec['stages'][name]
        if entry['seed_index'] != expected[name]:
            raise ValueError(f'{name} seed index differs from frozen plan')
        seed_file = resolve(root, entry['seed_file'])
        pair = seed_pair(seed_file, entry['seed_index'])
        stages[name] = {'seed_file': str(seed_file), 'seed_file_sha256': sha256(seed_file),
                        'seed_index': entry['seed_index'], 'seeds': pair}
    if stages['validation']['seeds'] != (24610084, 818001683):
        raise ValueError('held-out seed pair differs from frozen plan')
    if stages['discovery']['seeds'] == stages['cross_check']['seeds']:
        raise ValueError('development seeds must differ')
    return {'archives': archives, 'stages': stages, 'settings': {
        'minutes': 60, 'startup_allowance_minutes': 15, 'max_wall_hours': args.max_wall_hours,
        'reserve_minutes': 45, 'cap_mib': 19000, 'pool_limit_mib': 16000,
        'frequencies': 50, 'float64': True, 'objective_microbatch': 1},
        'plan_sha256': sha256(path), 'environment_versions': versions()}


def reject_measured_holdout(out):
    """Guard against accidentally reusing a known local holdout measurement."""
    roots = [Path(__file__).resolve().parents[1] / 'results']
    if not (out / 'experiment_lock.json').exists():
        roots.append(out)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('run.json'):
            try:
                row = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if ((row.get('topology_seed'), row.get('optimizer_seed')) ==
                    (24610084, 818001683) and row.get('best_feasible_loss') is not None):
                raise ValueError(f'holdout seed 5 has a previous measurement: {path}')


def outcome_record(out, stage, name, outcome):
    row, guard = outcome
    return {'stage': stage, 'variant': name,
            'status': ('complete' if valid(outcome) else 'incomplete'),
            'loss': row.get('best_feasible_loss'), 'evaluations': row.get('evaluation_count'),
            'feasible_candidates': row.get('feasible_candidate_count'),
            'objective_seconds': row.get('objective_seconds'),
            'process_wall_seconds': guard.get('process_wall_seconds'),
            'peak_gpu_used_mib': guard.get('peak_used_mib'),
            'guard_status': guard.get('status'), 'exit_code': guard.get('process_exit_code'),
            'error': row.get('error'), 'algorithm_diagnostics': row.get('algorithm_diagnostics'),
            'run_dir': str(out / stage / 'runs' / name)}


def save_report(out, plan, outcomes, advancing, selected, status, started_epoch):
    rows = [outcome_record(out, stage, name, outcomes[(stage, name)])
            for stage in STAGES for name in ('control', *VARIANTS)
            if (stage, name) in outcomes]
    report = {'status': status, 'started_epoch': started_epoch,
              'elapsed_wall_seconds': time.time() - started_epoch,
              'plan': plan, 'advancing_variants': advancing, 'selected_variant': selected,
              'discovery_gains': {name: gain(outcomes, 'discovery', name) for name in VARIANTS},
              'cross_check_gains': {name: gain(outcomes, 'cross_check', name) for name in VARIANTS},
              'holdout_gain': gain(outcomes, 'validation', selected) if selected else None,
              'research_recommendation': (selected if selected and
                                          (gain(outcomes, 'validation', selected) or 0) > 0
                                          else 'control'), 'runs': rows,
              'note': 'Screening only. Four-hour promotion testing remains required.'}
    save_json(out / 'experiment_report.json', report)
    lines = ['# Champion V3 isolated experiments', '', f'Status: {status}.',
             f'Advancing: {", ".join(advancing) or "none"}; selected: {selected or "none"}.',
             '', '| Stage | Variant | Status | Best feasible loss | Gain vs control | Evaluations | Peak MiB |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        g = gain(outcomes, row['stage'], row['variant']) if row['variant'] != 'control' else None
        fmt = lambda value: '—' if value is None else f'{value:.6f}'
        lines.append(f"| {row['stage']} | {row['variant']} | {row['status']} | "
                     f"{fmt(row['loss'])} | {fmt(g)} | {row['evaluations'] or '—'} | "
                     f"{row['peak_gpu_used_mib'] or '—'} |")
    lines += ['', f"Held-out gain: {report['holdout_gain'] if report['holdout_gain'] is not None else 'not measured'}.",
              f"Research recommendation: {report['research_recommendation']}.",
              'A positive held-out gain is required before recommending a variant for four-hour testing.']
    (out / 'experiment_report.md').write_text('\n'.join(lines) + '\n')
    return report


def main(args):
    path = args.experiment_plan.resolve()
    plan = load_plan(path, args)
    if args.dry_run:
        reject_measured_holdout(args.out.resolve())
        print(json.dumps(plan, indent=2))
        return 0
    args.out = args.out.resolve()
    reject_measured_holdout(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / 'experiment_lock.json'
    if lock.exists():
        state = json.loads(lock.read_text())
        if state['plan'] != plan:
            raise ValueError('experiment plan/archive/environment changed; use a new output directory')
        if args.started_epoch is not None and abs(state['started_epoch'] - args.started_epoch) > 1:
            raise ValueError('existing experiment clock differs from --started-epoch')
    else:
        started_epoch = args.started_epoch if args.started_epoch is not None else time.time()
        if started_epoch > time.time() or time.time() - started_epoch >= args.max_wall_hours * 3600:
            raise ValueError('experiment wall clock is invalid or already exhausted')
        state = {'plan': plan, 'started_epoch': started_epoch, 'selected_variant': None}
        save_json(lock, state)
    started = state['started_epoch']
    card = gpu_card()
    pool = pool_size(card, args.cap_mib, args.pool_limit_mib)
    if 'gpu' in state and state['gpu'] != card['name']:
        raise ValueError('resumed experiment must use the same GPU model')
    if 'gpu' not in state:
        state['gpu'] = card['name']
        save_json(lock, state)
    env = run_environment(card, pool)
    print('GPU:', card, 'JAX pool:', pool, 'MiB', flush=True)
    print('float64 probe:', probe_gpu(env), flush=True)
    outcomes = {}

    def run(stage, name):
        seed = plan['stages'][stage]
        folder = args.out / stage
        folder.mkdir(parents=True, exist_ok=True)
        method = {'key': name, 'label': f'{stage}: {name}',
                  'archive': plan['archives'][name], 'experiment': stage}
        job = SimpleNamespace(out=folder, seed_file=Path(seed['seed_file']),
                              seed_index=seed['seed_index'], minutes=60, cap_mib=19000,
                              startup_allowance_minutes=15)
        # A completed matching job costs no GPU time and is safe to resume.
        from colab_20gb_compare import reusable, BENCH
        expected = {'archive_sha256': method['archive']['sha256'],
                    'topology_seed': seed['seeds'][0], 'optimizer_seed': seed['seeds'][1],
                    'seed_file_sha256': seed['seed_file_sha256'], 'seed_index': seed['seed_index'],
                    'minutes': 60, 'frequencies': 50, 'batch': 1, 'cap_mib': 19000,
                    'pool_mib': pool, 'gpu_name': card['name'], 'gpu_driver': card['driver'],
                    'versions': versions(), 'bench_sha256': sha256(BENCH), 'experiment': stage}
        run_dir = folder / 'runs' / name
        existing = reusable(run_dir, expected)
        if existing is None and (run_dir / 'meta.json').is_file():
            if json.loads((run_dir / 'meta.json').read_text()) == expected:
                row_path, guard_path = run_dir / 'run.json', run_dir / 'memory_guard.json'
                row = json.loads(row_path.read_text()) if row_path.is_file() else {
                    'status': 'incomplete', 'error': 'interrupted before result'}
                guard = json.loads(guard_path.read_text()) if guard_path.is_file() else {
                    'status': 'incomplete', 'process_exit_code': None}
                existing = (row, guard)
                print(f'[reuse incomplete] {stage}/{name}; no automatic retry', flush=True)
        if existing is None:
            remaining = args.max_wall_hours * 3600 - (time.time() - started)
            if remaining < 75 * 60 + 45 * 60:
                print(f'[budget] Skipping {stage}/{name}: {remaining/60:.1f} minutes remain; '
                      'a full 75-minute job plus reserve does not fit.', flush=True)
                return False
            print(f'[start] {stage}/{name}; {remaining/60:.1f} minutes remain; '
                  f'logs: {run_dir / "driver.log"}', flush=True)
        outcomes[(stage, name)] = (existing if existing is not None else
                                   run_method(job, method, seed['seeds'], card, pool, env))
        save_report(args.out, plan, outcomes, discover(outcomes), state['selected_variant'],
                    'running', started)
        return True

    for name in ('control', *VARIANTS):
        if not run('discovery', name):
            save_report(args.out, plan, outcomes, [], None, 'budget_exhausted', started)
            return 1
    advancing = discover(outcomes)
    if not advancing:
        save_report(args.out, plan, outcomes, [], None, 'stopped_no_discovery_gain', started)
        return 0 if all(valid(item) for item in outcomes.values()) else 1
    for name in ('control', *advancing):
        if not run('cross_check', name):
            save_report(args.out, plan, outcomes, advancing, None, 'budget_exhausted', started)
            return 1
    selected = select(outcomes, advancing)
    if selected is None:
        save_report(args.out, plan, outcomes, advancing, None, 'stopped_no_cross_check_gain', started)
        return 0 if all(valid(item) for item in outcomes.values()) else 1
    # Persist the archive decision before the first held-out evaluation.
    if state['selected_variant'] is None:
        state['selected_variant'] = selected
        state['selected_archive_sha256'] = plan['archives'][selected]['sha256']
        save_json(lock, state)
    elif state['selected_variant'] != selected:
        raise ValueError('frozen validation selection differs from this resumed run')
    for name in ('control', selected):
        if not run('validation', name):
            save_report(args.out, plan, outcomes, advancing, selected, 'budget_exhausted', started)
            return 1
    report = save_report(args.out, plan, outcomes, advancing, selected, 'complete', started)
    return 0 if all(valid(outcome) for outcome in outcomes.values()) else 1
