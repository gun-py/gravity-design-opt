import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from official.validate_submission import validate_zip


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench.py'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_pairs(path, indices):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'topology_seed', 'optimizer_seed'} <= set(rows[0]):
        raise ValueError('seed file needs topology_seed and optimizer_seed columns')
    if max(indices) >= len(rows):
        raise ValueError('requested seed index is missing from seed file')
    return {index: (int(rows[index]['topology_seed']), int(rows[index]['optimizer_seed']))
            for index in indices}


def gpu_name():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=8)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else None
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def runtime_versions():
    versions = {'python': platform.python_version()}
    for package in ('jax', 'jaxlib', 'dfbench', 'optax', 'scipy', 'osqp'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cached(folder, expected):
    meta, result = folder / 'meta.json', folder / 'run.json'
    if not meta.is_file() or not result.is_file():
        return None
    try:
        if json.loads(meta.read_text()) != expected:
            return None
        row = json.loads(result.read_text())
        if (row.get('status') == 'complete' and row.get('sha256') == expected['archive_sha256']
                and row.get('topology_seed') == expected['topology_seed']
                and row.get('optimizer_seed') == expected['optimizer_seed']
                and row.get('minutes') == expected['minutes']
                and row.get('frequencies') == 50
                and row.get('adapted_baseline') == expected['adapted_baseline']
                and row.get('champion_features') == expected['features']):
            return row
    except (ValueError, KeyError):
        return None
    return None


def run_one(args, name, index, seeds, archive, archive_hash, gpu, start):
    topo, opt = seeds[index]
    adapted = name == 'baseline' and args.adapt_baseline
    features = args.candidate_features if name == 'candidate' else None
    expected = {'archive_sha256': archive_hash, 'topology_seed': topo, 'optimizer_seed': opt,
                'minutes': args.minutes, 'batch': args.batch, 'adapted_baseline': adapted,
                'features': features, 'runner_sha256': digest(BENCH), 'gpu': gpu,
                'runtime_versions': runtime_versions()}
    folder = args.out / name / f'index_{index}'
    old = cached(folder, expected)
    if old is not None:
        print(f'[reuse] {name} seed {index}: {old["best_feasible_loss"]}', flush=True)
        return old
    if (folder / 'run.json').exists():
        archive_dir = folder / f'previous_{int(time.time())}'
        archive_dir.mkdir(parents=True, exist_ok=True)
        for entry in folder.iterdir():
            if entry.is_file():
                shutil.move(str(entry), str(archive_dir / entry.name))
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'meta.json', expected)
    remaining = args.max_hours * 3600 - (time.monotonic() - start)
    cutoff = (datetime.fromisoformat(args.cutoff).timestamp() - time.time()
              if args.cutoff else float('inf'))
    run_limit = args.minutes * 60 + args.startup_allowance_minutes * 60
    if min(remaining, cutoff) < run_limit:
        row = {'status': 'skipped', 'error': 'run cannot finish within cap or cutoff', **expected}
        save(folder / 'run.json', row)
        return row
    command = [sys.executable, str(BENCH), '--worker', '--archive', str(archive),
               '--out', str(folder), '--minutes', str(args.minutes), '--batch', str(args.batch),
               '--topology-seed', str(topo), '--optimizer-seed', str(opt),
               '--frequencies', '50',
               '--progress-seconds', '60']
    if adapted:
        command.append('--adapt-baseline')
    env = os.environ.copy()
    env['JAX_ENABLE_X64'] = 'true'
    env['L2D_BATCH'] = str(args.batch)
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    env.pop('L2D_CHAMPION_FEATURES', None)
    if features is not None:
        env['L2D_CHAMPION_FEATURES'] = features
    with (folder / 'driver.log').open('w') as log:
        try:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=run_limit)
            if (folder / 'run.json').is_file():
                row = json.loads((folder / 'run.json').read_text())
            else:
                row = {'status': 'error', 'error': f'runner exited {process.returncode}', **expected}
        except subprocess.TimeoutExpired:
            row = {'status': 'timeout', 'error': 'optimization plus startup allowance exceeded', **expected}
    if row.get('status') == 'complete' and row.get('champion_features') != features:
        row = {'status': 'invalid', 'error': 'feature configuration mismatch', **expected}
    save(folder / 'run.json', row)
    print(f'{name} seed {index}: {row["status"]} {row.get("best_feasible_loss")}', flush=True)
    return row


def decision(args, archives, hashes, seeds, results, gpu):
    rows = []
    for index in args.indices:
        a, b = results[('baseline', index)], results[('candidate', index)]
        complete = (a.get('status') == b.get('status') == 'complete'
                    and a.get('best_feasible_loss') is not None
                    and b.get('best_feasible_loss') is not None)
        rows.append({'seed_index': index, 'topology_seed': seeds[index][0],
                     'optimizer_seed': seeds[index][1], 'baseline': a.get('best_feasible_loss'),
                     'candidate': b.get('best_feasible_loss'),
                     'improvement': a['best_feasible_loss'] - b['best_feasible_loss'] if complete else None,
                     'complete': complete})
    complete = all(r['complete'] for r in rows)
    gains = [r['improvement'] for r in rows if r['complete']]
    mean_gain = sum(gains) / len(gains) if complete else None
    wins = sum(gain > 0 for gain in gains) if complete else None
    worst_regression = max(-gain for gain in gains) if complete else None
    full_budget = args.minutes == 240
    performance_gate = (complete and full_budget and len(rows) >= 3 and mean_gain >= .02
                        and wins > len(rows) / 2 and worst_regression <= .10)
    passed = performance_gate and args.candidate_features is None
    report = {'baseline': {'path': str(archives['baseline']), 'sha256': hashes['baseline'],
                           'portal_provenance_verified': args.baseline_verified},
              'candidate': {'path': str(archives['candidate']), 'sha256': hashes['candidate'],
                            'features': args.candidate_features},
              'seed_file': str(args.seed_file), 'seed_file_sha256': digest(args.seed_file),
              'gpu': gpu, 'minutes_per_run': args.minutes,
              'baseline_adapted': args.adapt_baseline, 'paired': rows,
              'complete': complete, 'full_budget': full_budget,
              'mean_improvement': mean_gain, 'wins': wins,
              'worst_regression': worst_regression,
              'performance_gate_passed': performance_gate,
              'promotion_gate_passed': passed,
              'ablation_requires_packaged_confirmation': args.candidate_features is not None,
              'recommendation': 'candidate' if passed else 'baseline'}
    save(args.out / 'decision.json', report)
    selected_path = args.out / 'selected.zip'
    if selected_path.exists():
        selected_path.unlink()
    if complete and full_budget:
        shutil.copyfile(archives[report['recommendation']], selected_path)
        report['selected_sha256'] = digest(selected_path)
        save(args.out / 'decision.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT / 'artifacts/baselines/local_champion.zip')
    parser.add_argument('--candidate', type=Path, default=ROOT / 'dist/champion_v2.zip')
    parser.add_argument('--baseline-verified', action='store_true')
    parser.add_argument('--candidate-features', help='comma-separated ablation features; default uses all')
    parser.add_argument('--seed-file', type=Path, default=ROOT / 'analysis/final_holdout_seeds.csv')
    parser.add_argument('--indices', default='0,5,8')
    parser.add_argument('--minutes', type=float, default=240)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--adapt-baseline', action='store_true')
    parser.add_argument('--allow-other-gpu', action='store_true')
    parser.add_argument('--max-hours', type=float, default=30)
    parser.add_argument('--startup-allowance-minutes', type=float, default=30)
    parser.add_argument('--cutoff', help='ISO timestamp with timezone, if a stop time is desired')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/final_comparison')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        args.indices = [int(v) for v in args.indices.split(',')]
        if not args.indices or len(set(args.indices)) != len(args.indices) or min(args.indices) < 0:
            raise ValueError('indices must be unique nonnegative integers')
        if args.minutes <= 0 or args.max_hours <= 0 or args.startup_allowance_minutes < 0:
            raise ValueError('minutes, max-hours and startup allowance must be valid')
        if args.cutoff and datetime.fromisoformat(args.cutoff).tzinfo is None:
            raise ValueError('cutoff needs an explicit timezone')
        if args.candidate_features is not None:
            features = set(filter(None, args.candidate_features.split(',')))
            if not features <= {'screen', 'basin', 'state', 'reinject'}:
                raise ValueError('unknown candidate feature')
        args.seed_file = args.seed_file.resolve()
        seeds = seed_pairs(args.seed_file, args.indices)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    archives = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    hashes = {}
    for name, path in archives.items():
        if not path.is_file():
            parser.error(f'missing {name} archive: {path}')
        failures = [x.render() for x in validate_zip(str(path)) if x.severity == 'FAIL']
        if failures:
            parser.error(f'invalid {name} archive: {"; ".join(failures)}')
        hashes[name] = digest(path)
    planned = 2 * len(args.indices) * (args.minutes + args.startup_allowance_minutes) / 60
    plan = {'archives': {n: {'path': str(p), 'sha256': hashes[n]} for n, p in archives.items()},
            'seed_file': str(args.seed_file), 'seeds': seeds, 'minutes': args.minutes,
            'gpu_hours_with_startup_allowance': planned, 'cap_hours': args.max_hours,
            'baseline_adapted': args.adapt_baseline, 'candidate_features': args.candidate_features}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    gpu = gpu_name()
    if gpu is None:
        parser.error('an NVIDIA GPU is required')
    if not args.allow_other_gpu and 'H100' not in gpu:
        parser.error(f'expected an H100 GPU, found {gpu}; use --allow-other-gpu for local testing')
    if planned > args.max_hours:
        parser.error(f'planned jobs may need {planned:.1f} hours, cap is {args.max_hours:.1f}; adjust indices or cap')
    args.out.mkdir(parents=True, exist_ok=True)
    old_selection = args.out / 'selected.zip'
    if old_selection.exists():
        old_selection.rename(args.out / f'previous_selected_{time.time_ns()}.zip')
    save(args.out / 'plan.json', plan)
    start = time.monotonic()
    results = {}
    for index in args.indices:
        for name in ('baseline', 'candidate'):
            results[(name, index)] = run_one(args, name, index, seeds, archives[name], hashes[name], gpu, start)
    report = decision(args, archives, hashes, seeds, results, gpu)
    print(json.dumps({k: report[k] for k in ('complete', 'mean_improvement', 'wins',
                                            'promotion_gate_passed', 'recommendation')}, indent=2))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
