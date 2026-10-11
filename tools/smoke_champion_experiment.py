"""Short fresh-process GPU smoke test before the staged screening budget."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from champion_experiment import load_plan
from colab_20gb_compare import (gpu_card, pool_size, probe_gpu, run_environment,
                                run_method, save_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    settings = SimpleNamespace(minutes=60, startup_allowance_minutes=15, cap_mib=19000,
                               pool_limit_mib=16000, max_wall_hours=12)
    plan = load_plan(args.experiment_plan.resolve(), settings)
    seed = plan['stages']['discovery']
    card = gpu_card()
    pool = pool_size(card, 19000, 16000)
    env = run_environment(card, pool)
    print('float64 probe:', probe_gpu(env), flush=True)
    outcomes = {}
    for name in ('control', 'rescue', 'late_lr', 'polish'):
        print('[smoke]', name, flush=True)
        method = {'key': name, 'label': f'smoke: {name}',
                  'archive': plan['archives'][name], 'experiment': 'smoke'}
        job = SimpleNamespace(out=args.out.resolve(), seed_file=Path(seed['seed_file']),
                              seed_index=8, minutes=2, cap_mib=19000,
                              startup_allowance_minutes=15)
        row, guard = run_method(job, method, seed['seeds'], card, pool, env)
        phases = (row.get('algorithm_diagnostics') or {}).get('phase_end_seconds') or {}
        okay = (guard['status'] == 'within_cap' and row.get('error') is None
                and (row.get('evaluation_count') or 0) > 0 and 'search_c' in phases)
        outcomes[name] = {'pass': okay, 'status': row.get('status'),
                          'evaluations': row.get('evaluation_count'),
                          'error': row.get('error'), 'peak_used_mib': guard['peak_used_mib']}
        save_json(args.out / 'smoke_report.json', outcomes)
        if not okay:
            print(f'[smoke failure] {name}: {json.dumps(outcomes[name])}', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
