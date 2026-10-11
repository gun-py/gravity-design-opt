import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def pairs(round_num, indices):
    path = ROOT / 'competition_data' / f'round{round_num}' / 'seeds.csv'
    rows = list(csv.DictReader(path.open()))
    return [(int(rows[i]['topology_seed']), int(rows[i]['optimizer_seed'])) for i in indices]


def score(obj):
    from official.validate_submission import summarize_objective
    data = summarize_objective(obj)
    losses = obj.loss_history
    flags = obj.is_feasible_history
    best = None
    for a, b in zip(losses, flags):
        if b is None:
            continue
        x = np.asarray(a).reshape(-1)
        f = np.asarray(b, bool).reshape(-1)
        if len(x) != len(f):
            continue
        good = x[np.isfinite(x) & f]
        if good.size:
            val = float(good.min())
            best = val if best is None else min(best, val)
    official = data['best_feasible_loss']
    if official is not None and (best is None or abs(best-official) > 1e-8):
        raise RuntimeError(f'feasible score disagrees with official validator: {best} vs {official}')
    return data


def checkpoints(obj, minutes):
    out = {}
    for mark in [5, 15, 20, 30, 45, 60, 120, 180, 240]:
        if mark > minutes:
            continue
        best = None
        for elapsed, a, b in zip(obj.time_steps, obj.loss_history, obj.is_feasible_history):
            if float(elapsed) > mark*60 or b is None:
                continue
            x = np.asarray(a).reshape(-1)
            f = np.asarray(b, bool).reshape(-1)
            if len(x) != len(f):
                continue
            good = x[np.isfinite(x) & f]
            if good.size:
                val = float(good.min())
                best = val if best is None else min(best, val)
        out[str(mark)] = best
    return out


def gpu_memory():
    try:
        p = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        a = p.stdout.splitlines()[0].split(',')
        return {'used_mb':int(a[0]),'total_mb':int(a[1])}
    except Exception:
        return None


def gpu_details():
    try:
        p = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        a = [v.strip() for v in p.stdout.splitlines()[0].split(',')]
        return {'name':a[0],'driver':a[1],'total_mb':int(a[2])}
    except Exception:
        return None


class MicroObjective:
    def __init__(self, obj, batch):
        self.obj = obj
        self.batch = batch

    def __getattr__(self, name):
        if name.startswith('warmup_vmap_'):
            return lambda batch_size: getattr(self.obj,name)(batch_size=min(batch_size,self.batch))
        if name in ('vmap_value','vmap_value_aux','vmap_value_and_grad_aux'):
            return lambda x: self.call(name,x)
        return getattr(self.obj,name)

    def __setattr__(self,name,value):
        if name in ('obj','batch'):
            object.__setattr__(self,name,value)
        else:
            setattr(self.obj,name,value)

    def call(self, name, x):
        import jax.numpy as jnp
        import jax
        fun = getattr(self.obj,name)
        parts = [fun(x[i:i+self.batch]) for i in range(0,len(x),self.batch)]
        if name == 'vmap_value':
            return jnp.concatenate(parts)
        if name == 'vmap_value_aux':
            return (jnp.concatenate([a for a,_ in parts]),
                    jax.tree_util.tree_map(lambda *v:jnp.concatenate(v), *(b for _,b in parts)))
        return (jnp.concatenate([a for a,_,_ in parts]),
                jnp.concatenate([b for _,b,_ in parts]),
                jax.tree_util.tree_map(lambda *v:jnp.concatenate(v), *(c for _,_,c in parts)))


def worker(args):
    from dfbench import Objective, OptimizationAlgorithm
    from dfbench.problems import UIFOProblem, ConstrainedVoyagerProblem
    from official.validate_submission import load_optimizer_class, safe_extract, validate_zip
    import jax
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive).resolve()
    bad = [f.render() for f in validate_zip(str(archive)) if f.severity == 'FAIL']
    if bad:
        raise RuntimeError('invalid archive: ' + '; '.join(bad))
    t0 = time.time()
    result = {'status':'error','error':None,'traceback':None,'archive':str(archive),
              'sha256':digest(archive),'topology_seed':args.topology_seed,
              'optimizer_seed':args.optimizer_seed,'minutes':args.minutes,
              'frequencies':args.frequencies,
              'batch':args.batch,'gpu':gpu_memory(),'python':platform.python_version(),
              'gpu_details':gpu_details(),
              'jax':jax.__version__,'jaxlib':importlib.metadata.version('jaxlib'),
              'dfbench':importlib.metadata.version('dfbench'),
              'devices':[str(d) for d in jax.devices()],
              'x64':bool(jax.config.jax_enable_x64),
              'adapted_baseline':bool(args.adapt_baseline),
              'microbatch_objective':bool(args.microbatch_objective or args.adapt_baseline)}
    result['champion_features'] = os.environ.get('L2D_CHAMPION_FEATURES')
    obj = None
    algo = None
    stop = threading.Event()
    peak = [0]
    seen = [0]
    best = [None]
    def progress():
        while not stop.wait(args.progress_seconds):
            mem = gpu_memory()
            if mem:
                peak[0] = max(peak[0], mem['used_mb'])
            try:
                if obj is not None:
                    losses = obj.loss_history
                    flags = obj.is_feasible_history
                    end = min(len(losses),len(flags))
                    for a,b in zip(losses[seen[0]:end],flags[seen[0]:end]):
                        if b is None:
                            continue
                        x = np.asarray(a).reshape(-1)
                        f = np.asarray(b,bool).reshape(-1)
                        if len(x) == len(f):
                            good = x[np.isfinite(x) & f]
                            if good.size:
                                v = float(good.min())
                                best[0] = v if best[0] is None else min(best[0],v)
                    seen[0] = end
                row = {'time':time.time(),'elapsed':float(obj.time_elapsed) if obj else None,
                       'evals':int(obj.eval_count) if obj else 0,
                       'best_feasible':best[0],'memory':mem}
                with (out/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print('[progress]',json.dumps(row),flush=True)
            except Exception as exc:
                print(f'[progress] {type(exc).__name__}: {exc}',flush=True)
    thread = None
    try:
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td)
            safe_extract(str(archive), dst)
            sys.path.insert(0, td)
            try:
                cls = load_optimizer_class(dst/'submission.py', OptimizationAlgorithm)
                if args.voyager:
                    problem = ConstrainedVoyagerProblem()
                else:
                    problem = UIFOProblem(size=3, topology_seed=args.topology_seed, n_frequencies=args.frequencies)
                obj = Objective(problem,max_time=args.minutes*60,
                                save=['is_feasible','batched_loss','batched_is_feasible'],
                                display_mode='log',verbose=0)
                thread = threading.Thread(target=progress,daemon=True)
                thread.start()
                algo = cls()
                algo.optimize(MicroObjective(obj,args.batch)
                              if (args.microbatch_objective or args.adapt_baseline) else obj,
                              random_seed=args.optimizer_seed)
                data = score(obj)
                result.update(data)
                result['checkpoints'] = checkpoints(obj,args.minutes)
                result['status'] = ('complete' if data['best_feasible_loss'] is not None
                                    and float(obj.time_elapsed) >= .98*args.minutes*60 else 'invalid')
            finally:
                sys.path.remove(td)
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        if obj is not None:
            try:
                result.update(score(obj))
                result['checkpoints'] = checkpoints(obj,args.minutes)
            except Exception as inner:
                result['score_error'] = f'{type(inner).__name__}: {inner}'
    finally:
        stop.set()
        if thread:
            thread.join(timeout=3)
        if algo is not None and hasattr(algo, 'diagnostics'):
            result['algorithm_diagnostics'] = algo.diagnostics
        result['wall_seconds'] = time.time()-t0
        result['peak_memory_mb'] = peak[0]
        if obj is not None:
            result['objective_seconds'] = float(obj.time_elapsed)
            result['budget_exceeded'] = bool(obj.budget_exceeded)
        write(out/'run.json',result)
        print('[result]',json.dumps({k:v for k,v in result.items() if k != 'traceback'}),flush=True)
    return 0 if result['status']=='complete' else 1


def parent(args):
    archive = Path(args.archive or ROOT/'dist'/f'{args.entry}.zip').resolve()
    if not archive.is_file():
        raise SystemExit(f'missing archive {archive}')
    ids = [int(v) for v in args.indices.split(',')] if args.indices else list(range(args.seeds or (3 if args.fast else 10)))
    if args.voyager:
        ids = [0]
    seed_pairs = pairs(args.round,ids) if not args.voyager else [(0,0)]
    rows = []
    for i,(t,s) in zip(ids,seed_pairs):
        out = (Path(args.out) if args.single_dir else
               Path(args.out or ROOT/'results'/'bench')/f'{archive.stem}_r{args.round}_i{i}_{args.minutes:g}m')
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable,str(Path(__file__).resolve()),'--worker','--archive',str(archive),
               '--out',str(out),'--minutes',str(args.minutes),'--batch',str(args.batch),
               '--topology-seed',str(t),'--optimizer-seed',str(s),
               '--frequencies',str(args.frequencies),
               '--progress-seconds',str(args.progress_seconds)]
        if args.voyager:
            cmd.append('--voyager')
        if args.adapt_baseline:
            cmd.append('--adapt-baseline')
        if args.microbatch_objective:
            cmd.append('--microbatch-objective')
        env = os.environ.copy()
        env['L2D_BATCH'] = str(args.batch)
        env['JAX_ENABLE_X64'] = 'true'
        env.setdefault('MPLCONFIGDIR','/tmp/l2d-mpl')
        env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        print(f'run {archive.name} seed={i} topology={t} optimizer={s}',flush=True)
        try:
            with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
                proc = subprocess.run(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                                      timeout=args.minutes*60+1800)
            row = json.loads((out/'run.json').read_text())
            row['exit_code'] = proc.returncode
        except subprocess.TimeoutExpired:
            row = {'status':'timeout','error':'process exceeded optimization plus 30-minute overhead',
                   'topology_seed':t,'optimizer_seed':s,'best_feasible_loss':None}
            write(out/'run.json',row)
        except Exception as exc:
            row = {'status':'error','error':f'{type(exc).__name__}: {exc}',
                   'topology_seed':t,'optimizer_seed':s,'best_feasible_loss':None}
            write(out/'run.json',row)
        rows.append(row)
        print(json.dumps({'index':i,'status':row['status'],
                          'best_feasible':row.get('best_feasible_loss'),'error':row.get('error')}),flush=True)
    valid = len(rows)==len(ids) and all(r['status']=='complete' for r in rows)
    summary = {'archive':str(archive),'sha256':digest(archive),'round':args.round,
               'indices':ids,'minutes':args.minutes,'batch':args.batch,'runs':rows,
               'complete':valid,'mean_feasible_loss':float(np.mean([r['best_feasible_loss'] for r in rows])) if valid else None}
    dest = Path(args.json or Path(args.out or ROOT/'results'/'bench')/'summary.json')
    write(dest,summary)
    print(json.dumps({'complete':valid,'mean_feasible_loss':summary['mean_feasible_loss'],
                      'summary':str(dest)},indent=2),flush=True)
    return 0 if valid else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('entry',nargs='?')
    p.add_argument('--archive')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--round',type=int,default=2)
    p.add_argument('--indices')
    p.add_argument('--seeds',type=int)
    p.add_argument('--minutes',type=float,default=20)
    p.add_argument('--fast',action='store_true')
    p.add_argument('--full',action='store_true')
    p.add_argument('--voyager',action='store_true')
    p.add_argument('--adapt-baseline',action='store_true')
    p.add_argument('--microbatch-objective',action='store_true')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--frequencies',type=int,default=50)
    p.add_argument('--out')
    p.add_argument('--single-dir',action='store_true')
    p.add_argument('--json')
    p.add_argument('--progress-seconds',type=float,default=60)
    p.add_argument('--topology-seed',type=int)
    p.add_argument('--optimizer-seed',type=int)
    a=p.parse_args()
    if a.full and a.minutes==20:
        a.minutes=240
    if not a.archive and not a.entry:
        p.error('entry or --archive required')
    if a.worker:
        if not a.archive or not a.out:
            p.error('worker needs --archive and --out')
        return worker(a)
    return parent(a)


if __name__=='__main__':
    raise SystemExit(main())
