import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from official.validate_submission import validate_zip

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent/'bench.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def mean(rows, name, ids):
    selected = [rows.get((name,i),{}) for i in ids]
    if len(selected) != len(ids) or any(r.get('status') != 'complete' or r.get('best_feasible_loss') is None for r in selected):
        return None
    return sum(r['best_feasible_loss'] for r in selected)/len(selected)


def run_one(out, stage, name, index, minutes, archive, batch, started, cap):
    folder = out/'runs'/stage/name/f'index_{index}'
    path = folder/'run.json'
    expected = {'sha256':sha(archive),'minutes':minutes,'batch':batch}
    if path.is_file():
        row = json.loads(path.read_text())
        if row.get('status') == 'complete' and all(row.get(k) == v for k,v in expected.items()):
            return row
    folder.mkdir(parents=True, exist_ok=True)
    if time.monotonic()-started + minutes*60+300 > cap:
        row = {'status':'skipped','error':'12-hour cap would be exceeded',
               'best_feasible_loss':None, **expected}
        save(path,row)
        return row
    cmd = [sys.executable,str(SCRIPT),'--archive',str(archive),'--round','2',
           '--indices',str(index),'--minutes',str(minutes),'--batch',str(batch),
           '--out',str(folder),'--single-dir','--json',str(folder/'summary.json'),
           '--progress-seconds','60']
    env = os.environ.copy()
    env['JAX_ENABLE_X64'] = 'true'
    env['L2D_BATCH'] = str(batch)
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    env.setdefault('MPLCONFIGDIR','/tmp/l2d-mpl')
    with (folder/'driver.log').open('w') as log:
        proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,
                                stderr=subprocess.STDOUT,start_new_session=True)
        try:
            proc.wait(timeout=min(minutes*60+300,cap-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            row = {'status':'timeout','error':'wall-time cap','best_feasible_loss':None,**expected}
            save(path,row)
            return row
    if path.is_file():
        row = json.loads(path.read_text())
    else:
        row = {'status':'error','error':f'runner exited {proc.returncode}',
               'best_feasible_loss':None,**expected}
        save(path,row)
    return row


def stage(out, label, jobs, archives, batch, started, cap):
    rows = {}
    for name,index,minutes in jobs:
        row = run_one(out,label,name,index,minutes,archives[name],batch,started,cap)
        rows[(name,index)] = row
        print(f'{label} {name} seed={index} {row["status"]} {row.get("best_feasible_loss")}',flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--n6',type=Path,default=ROOT/'dist'/'n6_relay.zip')
    p.add_argument('--n7',type=Path,default=ROOT/'dist'/'n7_fsq.zip')
    p.add_argument('--out',type=Path,default=ROOT/'results'/'n7_comparison')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--max-hours',type=float,default=12)
    p.add_argument('--baseline-verified',action='store_true')
    p.add_argument('--allow-other-gpu',action='store_true')
    a = p.parse_args()
    if importlib.util.find_spec('osqp') is None:
        p.error('OSQP is required for this comparison; install entries/n7_fsq/requirements.txt')
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True)
    if gpu.returncode != 0 or not gpu.stdout.strip():
        p.error('an NVIDIA GPU is required')
    if not a.allow_other_gpu and 'A100' not in gpu.stdout:
        p.error(f'expected an A100 GPU, found {gpu.stdout.strip()}')
    archives = {'baseline':a.baseline.resolve(),'n6':a.n6.resolve(),'n7':a.n7.resolve()}
    for path in archives.values():
        if not path.is_file():
            p.error(f'missing archive: {path}')
        failures = [x.render() for x in validate_zip(str(path)) if x.severity == 'FAIL']
        if failures:
            p.error(f'invalid archive {path}: {"; ".join(failures)}')
    a.out.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    cap = a.max_hours*3600
    screen_jobs = [(name,i,20) for i in (0,5) for name in ('baseline','n6','n7')]
    screen = stage(a.out,'screen',screen_jobs,archives,a.batch,started,cap)
    candidates = [(mean(screen,name,[0,5]),name) for name in ('n6','n7')]
    valid = [(loss,name) for loss,name in candidates if loss is not None]
    challenger = min(valid)[1] if valid else None
    confirm = {}
    gate = {'passed':False,'reason':'no completed screening challenger'}
    endurance = {}
    if challenger is not None:
        jobs = [(name,i,45) for i in (2,7,9) for name in ('baseline',challenger)]
        confirm = stage(a.out,'confirm',jobs,archives,a.batch,started,cap)
        base = mean(confirm,'baseline',[2,7,9])
        new = mean(confirm,challenger,[2,7,9])
        if base is not None and new is not None:
            gains = {i:confirm[('baseline',i)]['best_feasible_loss']-
                     confirm[(challenger,i)]['best_feasible_loss'] for i in (2,7,9)}
            gate = {'baseline_mean':base,'challenger_mean':new,'gains':gains,
                    'wins':sum(v>0 for v in gains.values()),
                    'max_regression':max(-v for v in gains.values())}
            gate['passed'] = (base-new >= .02 and gate['wins'] >= 2 and gate['max_regression'] <= .10)
            if gate['passed']:
                hard = max((2,7,9),key=lambda i:confirm[(challenger,i)]['best_feasible_loss'])
                endurance = stage(a.out,'endurance',[(challenger,hard,240)],archives,a.batch,started,cap)
                gate['endurance_seed'] = hard
                gate['passed'] = endurance[(challenger,hard)]['status'] == 'complete'
        else:
            gate = {'passed':False,'reason':'confirmation missing feasible completed runs'}
    selected = challenger if gate['passed'] else 'baseline'
    report = {'selected':selected,'challenger':challenger,'gate':gate,
              'screen':{f'{name}_{i}':row for (name,i),row in screen.items()},
              'confirm':{f'{name}_{i}':row for (name,i),row in confirm.items()},
              'endurance':{f'{name}_{i}':row for (name,i),row in endurance.items()},
              'archives':{name:{'path':str(path),'sha256':sha(path)} for name,path in archives.items()},
              'wall_seconds':time.monotonic()-started,'gpu_hour_cap':a.max_hours,
              'gpu':gpu.stdout.strip(),
              'baseline_verified_round2':a.baseline_verified}
    save(a.out/'decision.json',report)
    target = a.out/'selected.zip'
    shutil.copyfile(archives[selected],target)
    report['selected_sha256'] = sha(target)
    save(a.out/'decision.json',report)
    print(json.dumps({'selected':selected,'gate':gate,'zip':str(target),'sha256':sha(target)},indent=2),flush=True)


if __name__ == '__main__':
    main()
