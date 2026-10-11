import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from official.validate_submission import validate_zip


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools/bench.py'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_card():
    command = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'could not read NVIDIA GPU memory: {error}') from error
    if result.returncode or len(result.stdout.strip().splitlines()) != 1:
        raise RuntimeError('one NVIDIA GPU is required')
    name, driver, total, used = (part.strip() for part in result.stdout.strip().split(','))
    return {'name': name, 'driver': driver, 'total_mib': int(total), 'used_mib': int(used)}


def pool_size(card, cap_mib, maximum_pool_mib):
    if card['used_mib'] >= cap_mib - 3000:
        raise RuntimeError('GPU memory is already too full; use a fresh Colab runtime')
    pool = min(maximum_pool_mib, int(card['total_mib'] * .78),
               cap_mib - card['used_mib'] - 2500)
    if pool < 6000:
        raise RuntimeError('less than 6 GiB can be reserved under the cap')
    return pool


def versions():
    result = {'python': platform.python_version()}
    for package in ('jax', 'jaxlib', 'dfbench', 'optax', 'scipy', 'osqp'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def seed_pair(path, index):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'topology_seed', 'optimizer_seed'} <= set(rows[0]):
        raise ValueError('seed CSV needs topology_seed and optimizer_seed columns')
    if index < 0 or index >= len(rows):
        raise ValueError(f'seed index {index} is outside 0..{len(rows) - 1}')
    row = rows[index]
    return int(row['topology_seed']), int(row['optimizer_seed'])


def run_environment(card, pool_mib):
    env = os.environ.copy()
    env['JAX_ENABLE_X64'] = 'true'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    fraction = f"{pool_mib / card['total_mib']:.8f}"
    env['XLA_CLIENT_MEM_FRACTION'] = fraction
    env.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
    env.pop('L2D_CHAMPION_FEATURES', None)
    env['L2D_BATCH'] = '1'
    env.setdefault('MPLCONFIGDIR', '/tmp/l2d-mpl')
    return env


def probe_gpu(env):
    code = (
        'import jax,jax.numpy as jnp; '
        'x=jnp.asarray([1.0],dtype=jnp.float64).block_until_ready(); '
        'assert x.dtype==jnp.float64; '
        'assert jax.devices("gpu"); '
        'print(jax.__version__,jax.devices(),x.dtype)'
    )
    check = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                           text=True, timeout=180)
    if check.returncode:
        raise RuntimeError(f'GPU float64 probe failed: {(check.stdout + check.stderr)[-1500:]}')
    return check.stdout.strip()


def check_archive(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    failures = [issue.render() for issue in validate_zip(str(path)) if issue.severity == 'FAIL']
    if failures:
        raise ValueError(f'invalid archive {path}: {"; ".join(failures)}')
    return {'path': str(path), 'sha256': sha256(path)}


def reusable(folder, expected):
    meta = folder / 'meta.json'
    row_file = folder / 'run.json'
    guard_file = folder / 'memory_guard.json'
    if not all(path.is_file() for path in (meta, row_file, guard_file)):
        return None
    try:
        row, guard = json.loads(row_file.read_text()), json.loads(guard_file.read_text())
        if (json.loads(meta.read_text()) == expected and row.get('status') == 'complete'
                and row.get('sha256') == expected['archive_sha256']
                and row.get('topology_seed') == expected['topology_seed']
                and row.get('optimizer_seed') == expected['optimizer_seed']
                and row.get('frequencies') == 50 and row.get('minutes') == expected['minutes']
                and row.get('microbatch_objective') is True
                and guard.get('status') == 'within_cap'
                and guard.get('process_exit_code') == 0
                and guard.get('peak_used_mib', 999999) < expected['cap_mib']):
            return row, guard
    except (ValueError, KeyError):
        return None
    return None


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_method(args, method, pair, card, pool_mib, env):
    folder = args.out / 'runs' / method['key']
    expected = {'archive_sha256': method['archive']['sha256'],
                'topology_seed': pair[0], 'optimizer_seed': pair[1],
                'seed_file_sha256': sha256(args.seed_file), 'seed_index': args.seed_index,
                'minutes': args.minutes, 'frequencies': 50, 'batch': 1,
                'cap_mib': args.cap_mib, 'pool_mib': pool_mib,
                'gpu_name': card['name'], 'gpu_driver': card['driver'],
                'versions': versions(), 'bench_sha256': sha256(BENCH)}
    if 'experiment' in method:
        expected['experiment'] = method['experiment']
    prior = reusable(folder, expected)
    if prior is not None:
        print(f"[reuse] {method['label']}: {prior[0]['best_feasible_loss']}", flush=True)
        return prior
    if folder.exists():
        folder.rename(folder.with_name(folder.name + '_attempt_' + str(time.time_ns())))
    folder.mkdir(parents=True)
    save_json(folder / 'meta.json', expected)
    command = [sys.executable, str(BENCH), '--worker', '--archive', method['archive']['path'],
               '--out', str(folder), '--topology-seed', str(pair[0]),
               '--optimizer-seed', str(pair[1]), '--minutes', str(args.minutes),
               '--frequencies', '50', '--batch', '1', '--microbatch-objective',
               '--progress-seconds', '60']
    observed = []
    peak = card['used_mib']
    reason = None
    started = time.monotonic()
    with (folder / 'driver.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    used = gpu_card()['used_mib']
                    peak = max(peak, used)
                    observed.append((round(time.monotonic() - started, 2), used))
                    if used >= args.cap_mib:
                        reason = 'memory_cap_exceeded'
                        terminate_group(process)
                        break
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                    reason = 'memory_measurement_failed'
                    terminate_group(process)
                    break
                if time.monotonic() - started >= (args.minutes + args.startup_allowance_minutes) * 60:
                    reason = 'process_timeout'
                    terminate_group(process)
                    break
                time.sleep(.5)
            process.wait()
        finally:
            if process.poll() is None:
                terminate_group(process)
    with (folder / 'memory_trace.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(('elapsed_seconds', 'gpu_used_mib'))
        writer.writerows(observed)
    guard = {'status': reason or ('within_cap' if peak < args.cap_mib else 'memory_cap_exceeded'),
             'peak_used_mib': peak, 'cap_mib': args.cap_mib, 'pool_mib': pool_mib,
             'process_wall_seconds': time.monotonic() - started,
             'process_exit_code': process.returncode}
    save_json(folder / 'memory_guard.json', guard)
    if (folder / 'run.json').is_file():
        row = json.loads((folder / 'run.json').read_text())
    else:
        row = {'status': 'incomplete', 'best_feasible_loss': None,
               'error': reason or 'worker did not write run.json'}
        save_json(folder / 'run.json', row)
    print(f"{method['label']}: {row.get('status')}, best={row.get('best_feasible_loss')}, "
          f'peak={peak} MiB, guard={guard["status"]}', flush=True)
    return row, guard


def write_report(args, methods, pair, card, pool_mib, outcomes):
    results = []
    for method in methods:
        row, guard = outcomes.get(method['key'], ({}, {}))
        completed = (row.get('status') == 'complete' and guard.get('status') == 'within_cap'
                     and guard.get('process_exit_code') == 0
                     and row.get('best_feasible_loss') is not None)
        status = ('complete' if completed else
                  guard.get('status') if guard.get('status') not in (None, 'within_cap') else
                  row.get('status') if row.get('status') not in (None, 'complete') else
                  'worker_failed' if guard.get('process_exit_code') not in (None, 0) else
                  'pending')
        results.append({'method': method['label'], 'key': method['key'],
                        'archive_sha256': method['archive']['sha256'],
                        'origin_verified': method.get('origin_verified', False),
                        'status': status,
                        'best_feasible_loss': row.get('best_feasible_loss') if completed else None,
                        'evaluations': row.get('evaluation_count'),
                        'feasible_candidates': row.get('feasible_candidate_count'),
                        'optimizer_seconds': row.get('objective_seconds'),
                        'process_wall_seconds': guard.get('process_wall_seconds'),
                        'peak_gpu_used_mib': guard.get('peak_used_mib'),
                        'run_dir': str(args.out / 'runs' / method['key'])})
    completed = [item for item in results if item['status'] == 'complete']
    leader = min(completed, key=lambda item: item['best_feasible_loss'])['method'] if completed else None
    report = {'seed_index': args.seed_index, 'topology_seed': pair[0],
              'optimizer_seed': pair[1], 'seed_file': str(args.seed_file),
              'minutes_per_method': args.minutes, 'frequencies': 50,
              'microbatch': 1, 'gpu': card, 'cap_mib': args.cap_mib,
              'jax_pool_mib': pool_mib, 'results': results,
              'best_on_this_single_topology': leader,
              'note': 'One topology and a 60-minute budget cannot establish four-hour final superiority.'}
    save_json(args.out / 'comparison.json', report)
    with (args.out / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = [f'# One-hour comparison on topology seed {pair[0]}', '',
             f'GPU: {card["name"]}; JAX pool: {pool_mib} MiB; observed cap: {args.cap_mib} MiB.',
             'All methods use 50 frequencies, microbatch one, the same optimizer seed, and a 60-minute Objective budget.',
             '', '| Method | Status | Best feasible loss ↓ | Evaluations | Peak GPU MiB |',
             '|---|---|---:|---:|---:|']
    for item in results:
        loss = (f"{item['best_feasible_loss']:.6f}" if item['best_feasible_loss'] is not None else '—')
        lines.append(f"| {item['method']} | {item['status']} | {loss} | "
                     f"{item['evaluations'] or '—'} | {item['peak_gpu_used_mib'] or '—'} |")
    lines += ['', f'Best on this topology: {leader or "no complete run"}.',
              'This is a screening result; confirm promising methods over several full four-hour runs.']
    (args.out / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--phase3', type=Path)
    parser.add_argument('--phase3-label', default='N7 local (Phase 3 identity unverified)')
    parser.add_argument('--phase3-verified', action='store_true')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--seed-file', type=Path)
    parser.add_argument('--seed-index', type=int, default=0)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--cap-mib', type=int, default=19000)
    parser.add_argument('--pool-limit-mib', type=int, default=16000)
    parser.add_argument('--startup-allowance-minutes', type=float, default=15)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--experiment-plan', type=Path,
                        help='run the staged Champion V3 experiment instead of the classic comparison')
    parser.add_argument('--max-wall-hours', type=float, default=12.0,
                        help='global elapsed-time cap for the staged experiment')
    parser.add_argument('--started-epoch', type=float,
                        help='wall-budget clock started before preflight and smoke testing')
    args = parser.parse_args()
    if args.experiment_plan:
        from champion_experiment import main as run_experiment
        return run_experiment(args)
    if not all((args.baseline, args.phase3, args.candidate, args.seed_file)):
        parser.error('classic comparison requires --baseline, --phase3, --candidate and --seed-file')
    if args.minutes <= 0 or args.cap_mib > 19000 or args.cap_mib < 8000:
        parser.error('minutes must be positive and cap-mib must be 8,000–19,000')
    if args.pool_limit_mib < 6000 or args.startup_allowance_minutes < 0:
        parser.error('pool limit and startup allowance must be nonnegative practical values')
    args.seed_file = args.seed_file.resolve()
    try:
        pair = seed_pair(args.seed_file, args.seed_index)
        methods = [
            {'key': 'champion', 'label': 'Champion', 'archive': check_archive(args.baseline)},
            {'key': 'phase3', 'label': args.phase3_label, 'archive': check_archive(args.phase3),
             'origin_verified': args.phase3_verified},
            {'key': 'champion_v2', 'label': 'Champion V2', 'archive': check_archive(args.candidate)},
        ]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    plan = {'seed_file': str(args.seed_file), 'seed_index': args.seed_index,
            'topology_seed': pair[0], 'optimizer_seed': pair[1], 'minutes_per_method': args.minutes,
            'max_gpu_memory_mib': args.cap_mib,
            'methods': [{k: v for k, v in item.items() if k != 'archive'} |
                        {'archive': item['archive']} for item in methods]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    card = gpu_card()
    pool_mib = pool_size(card, args.cap_mib, args.pool_limit_mib)
    env = run_environment(card, pool_mib)
    print('GPU:', card, 'JAX pool:', pool_mib, 'MiB', flush=True)
    print('float64 probe:', probe_gpu(env), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / 'plan.json', plan | {'gpu': card, 'jax_pool_mib': pool_mib,
                                             'environment_versions': versions()})
    outcomes = {}
    for method in methods:
        row, guard = run_method(args, method, pair, card, pool_mib, env)
        outcomes[method['key']] = row, guard
        write_report(args, methods, pair, card, pool_mib, outcomes)
    return 0 if all(row.get('status') == 'complete' and guard.get('status') == 'within_cap'
                    and guard.get('process_exit_code') == 0
                    for row, guard in outcomes.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
