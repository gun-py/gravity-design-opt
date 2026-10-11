import argparse
import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR','/tmp/l2d-mpl')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FREEZE = datetime(2026, 9, 30, 12, 0, tzinfo=ZoneInfo('Asia/Kolkata'))
MAX_WALL = 12 * 3600
NAMES = ('baseline', 'champion_repair', 'n5_rescue')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(ZoneInfo('Asia/Kolkata'))


def seconds_left(start):
    return min((FREEZE - now()).total_seconds(), MAX_WALL - (time.time() - start))


def gate(start, jobs, reserve=1800):
    required = sum(minutes * 60 + 300 for _, _, minutes in jobs) + reserve
    return seconds_left(start) >= required, required


def result_path(out, stage, name, index):
    return out / 'runs' / stage / name / f'index_{index}' / 'run.json'


def one(out, stage, name, archive, index, minutes, batch, adapt, start):
    path = result_path(out, stage, name, index)
    expected = {'sha256': sha(archive), 'minutes': minutes, 'batch': batch,
                'adapted_baseline':bool(adapt and name == 'baseline')}
    if path.is_file():
        try:
            row = json.loads(path.read_text())
            if row.get('status') == 'complete' and all(row.get(k) == v for k, v in expected.items()):
                print(f'[resume] {stage} {name} seed {index}: {row["best_feasible_loss"]:+.8f}', flush=True)
                return row
        except (ValueError, KeyError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = str(int(time.time()))
        for old in ('run.json','progress.jsonl','stdout.log','stderr.log','summary.json','driver.log'):
            source = path.parent/old
            if source.exists():
                source.rename(path.parent/f'attempt_{stamp}_{old}')
    cmd = [sys.executable, str(ROOT/'tools'/'bench.py'), '--archive', str(archive),
           '--round', '2', '--indices', str(index), '--minutes', str(minutes),
           '--batch', str(batch), '--out', str(path.parent), '--single-dir',
           '--json', str(path.parent/'summary.json'), '--progress-seconds', '60']
    if adapt and name == 'baseline':
        cmd.append('--adapt-baseline')
    limit = min(minutes*60+1800, seconds_left(start)-1200)
    if limit <= 0:
        row = {'status':'timeout','error':'GPU cap or internal cutoff reached',
               'best_feasible_loss':None,**expected}
        save(path,row)
        return row
    with (path.parent/'driver.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            row = {'status':'timeout','error':'run hit GPU cap or internal cutoff',
                   'best_feasible_loss':None,**expected}
            save(path,row)
            return row
    if path.is_file():
        row = json.loads(path.read_text())
    else:
        row = {'status': 'error', 'error': f'runner exit {proc.returncode}; see driver.log',
               'best_feasible_loss': None, **expected}
        save(path, row)
    return row


def stage(out, title, jobs, archives, batch, start, adapt):
    waiting = [(n, i, m) for n, i, m in jobs if not complete(result_path(out,title,n,i),archives[n],m,batch,adapt and n == 'baseline')]
    allowed, required = (True,0) if not waiting else gate(start, waiting)
    if not allowed:
        print(f'[skip] {title}: need {required/3600:.2f}h, available {seconds_left(start)/3600:.2f}h', flush=True)
        return None
    rows = {}
    for name, index, minutes in jobs:
        row = one(out,title,name,archives[name],index,minutes,batch,adapt,start)
        rows[(name,index)] = row
        print(f'[{title}] {name} {index}: {row.get("status")} {row.get("best_feasible_loss")}', flush=True)
    return rows


def complete(path, archive, minutes, batch, adapt):
    try:
        row = json.loads(path.read_text())
        return (row['status'] == 'complete' and row['sha256'] == sha(archive)
                and row['minutes'] == minutes and row['batch'] == batch
                and row.get('adapted_baseline',False) == adapt)
    except (FileNotFoundError, KeyError, ValueError):
        return False


def mean(rows, name, ids):
    vals = [rows.get((name,i),{}) for i in ids]
    if len(vals) != len(ids) or any(v.get('status') != 'complete' for v in vals):
        return None
    return sum(v['best_feasible_loss'] for v in vals) / len(vals)


def choose_screen(rows):
    a = mean(rows,'champion_repair',[0,5])
    b = mean(rows,'n5_rescue',[0,5])
    if a is None and b is None:
        return None
    if a is None:
        return 'n5_rescue'
    if b is None or a <= b + .005:
        return 'champion_repair'
    return 'n5_rescue'


def compare(rows, challenger):
    ids = [2,7,9]
    base = mean(rows,'baseline',ids)
    test = mean(rows,challenger,ids)
    info = {'baseline_mean':base,'challenger_mean':test,'improvement':None,
            'wins':0,'max_regression':None,'passed':False}
    if base is None or test is None:
        return info
    diffs = [rows[('baseline',i)]['best_feasible_loss'] - rows[(challenger,i)]['best_feasible_loss'] for i in ids]
    info.update(improvement=base-test,wins=sum(x>0 for x in diffs),
                max_regression=max(-x for x in diffs))
    info['passed'] = (base-test >= .02 and info['wins'] >= 2 and info['max_regression'] <= .10)
    return info


def tables(out, runs):
    rows = []
    for stage_name, vals in runs.items():
        if not vals:
            continue
        for (name,index), row in vals.items():
            rows.append({'stage':stage_name,'algorithm':name,'index':index,
                         'status':row.get('status'),'loss':row.get('best_feasible_loss'),
                         'wall_seconds':row.get('wall_seconds'),'error':row.get('error'),
                         'sha256':row.get('sha256')})
    with (out/'per_seed.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=['stage','algorithm','index','status','loss','wall_seconds','error','sha256'])
        w.writeheader()
        w.writerows(rows)
    return rows


def plots(out, runs):
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for stage_name, vals in runs.items():
        if not vals:
            continue
        for (name,index), row in vals.items():
            cp = row.get('checkpoints') or {}
            pts = sorted((float(k),v) for k,v in cp.items() if v is not None)
            if pts:
                axes[0].plot(*zip(*pts),label=f'{stage_name} {name} {index}',alpha=.7)
            if row.get('feasible_candidate_count') is not None:
                axes[1].scatter(row.get('evaluation_count',0),row['feasible_candidate_count'],label=f'{stage_name} {name} {index}')
    axes[0].set(xlabel='Optimization minute',ylabel='Best feasible loss',title='Convergence')
    axes[1].set(xlabel='Candidate evaluations',ylabel='Feasible candidates',title='Feasibility')
    for ax in axes:
        ax.grid(alpha=.2)
    if axes[0].lines or axes[1].collections:
        axes[1].legend(fontsize=6,loc='best')
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160)
    plt.close(fig)


def validate(archive, out, smoke, adapted=False):
    from official.validate_submission import validate_zip, safe_extract, load_optimizer_class
    from dfbench import OptimizationAlgorithm
    fails = [v.render() for v in validate_zip(str(archive)) if v.severity == 'FAIL']
    extracted = out/'extracted_selected'
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir()
    safe_extract(str(archive),extracted)
    sys.path.insert(0,str(extracted))
    try:
        cls = load_optimizer_class(extracted/'submission.py',OptimizationAlgorithm)
    finally:
        sys.path.remove(str(extracted))
    info = {'static_failures':fails,'class':cls.__name__,'extracted':str(extracted),'execution':'not run'}
    if smoke and not fails:
        if adapted:
            cmd = [sys.executable,str(ROOT/'tools'/'bench.py'),'--archive',str(archive),
                   '--indices','0','--round','2','--minutes','0.2','--batch','2',
                   '--single-dir','--out',str(out/'validation_run'),'--adapt-baseline']
        else:
            cmd = [sys.executable,str(ROOT/'tools'/'official'/'validate_submission.py'),str(archive),
                   '--run-local','--topology-seed','289505736','--optimizer-seed','585204139',
                   '--max-time-seconds','12']
        env = os.environ.copy()
        env.update(JAX_ENABLE_X64='true', L2D_BATCH='2', XLA_PYTHON_CLIENT_PREALLOCATE='false')
        try:
            with (out/'validation.log').open('w') as f:
                p = subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
            if adapted:
                run_path = out/'validation_run'/'run.json'
                row = json.loads(run_path.read_text()) if run_path.is_file() else {}
                okay = row.get('error') is None and row.get('objective_seconds',0)>0
                info['execution'] = 'passed_adapted' if okay else 'failed'
                info['short_run_feasible_loss'] = row.get('best_feasible_loss')
            else:
                info['execution'] = 'passed' if p.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            info['execution'] = 'timeout'
    save(out/'validation.json',info)
    return info


def report(out, decision, rows, archives, elapsed):
    lines = ['# L2D selection', '', f'Selected: **{decision["selected"]}**',
             f'Reason: {decision["reason"]}', f'Baseline provenance: {decision["baseline_provenance"]}',
             f'Upload-ready validation: {decision["upload_ready"]}',
             'Published Round 2 submission #250: rank 9, mean feasible loss 0.167875. This score belongs to the portal upload, not automatically to the local copy.',
             f'GPU wall time: {elapsed/3600:.2f} hours', '',
             '| Stage | Algorithm | Seed index | Status | Best feasible | Error |',
             '|---|---|---:|---|---:|---|']
    for row in rows:
        loss = row['loss']
        lines.append(f'| {row["stage"]} | {row["algorithm"]} | {row["index"]} | {row["status"]} | {loss if loss is not None else "—"} | {row["error"] or ""} |')
    lines.extend(['','## Gates','',f'```json\n{json.dumps(decision["gates"],indent=2)}\n```','',
                  'A100 measurements guide this selection; they do not reproduce the official H100 leaderboard.'])
    (out/'decision.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    p.add_argument('--repair',required=True)
    p.add_argument('--rescue',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--baseline-provenance',default='unverified local copy')
    p.add_argument('--baseline-sha256')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--adapt-baseline',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    archives = dict(zip(NAMES,map(lambda x:Path(x).resolve(),[a.baseline,a.repair,a.rescue])))
    if any(not v.is_file() for v in archives.values()):
        raise SystemExit('missing archive')
    hashes = {k:sha(v) for k,v in archives.items()}
    if a.baseline_sha256 and hashes['baseline'] != a.baseline_sha256:
        raise SystemExit('baseline checksum mismatch')
    save(out/'archives.json',{'paths':{k:str(v) for k,v in archives.items()},'sha256':hashes,
                              'baseline_provenance':a.baseline_provenance})
    clock = out/'run_clock.json'
    if clock.is_file():
        start = float(json.loads(clock.read_text())['gpu_start_epoch'])
    else:
        start = time.time()
        save(clock,{'gpu_start_epoch':start})
    runs = {'screen':None,'confirm':None,'endurance':None}
    challenger = None
    check = {}
    gates = {'screen':False,'confirm':False,'endurance':False}
    reason = 'Evidence incomplete; preserved baseline selected.'
    try:
        screen_jobs = [(n,i,20) for n in NAMES for i in (0,5)]
        confirm_jobs = []
        if not a.dry_run:
            runs['screen'] = stage(out,'screen',screen_jobs,archives,a.batch,start,a.adapt_baseline)
        challenger = choose_screen(runs['screen'] or {})
        gates['screen'] = challenger is not None and mean(runs['screen'],'baseline',[0,5]) is not None
        if gates['screen']:
            confirm_jobs = [(n,i,45) for n in ('baseline',challenger) for i in (2,7,9)]
            runs['confirm'] = stage(out,'confirm',confirm_jobs,archives,a.batch,start,a.adapt_baseline)
        check = compare(runs['confirm'] or {},challenger) if challenger else {}
        gates['confirm'] = check.get('passed',False)
        if gates['confirm']:
            hardest = max((2,7,9),key=lambda i:runs['confirm'][('baseline',i)]['best_feasible_loss'])
            runs['endurance'] = stage(out,'endurance',[(challenger,hardest,240)],archives,a.batch,start,a.adapt_baseline)
            gates['endurance'] = bool(runs['endurance'] and runs['endurance'][(challenger,hardest)]['status']=='complete')
        selected = challenger if all(gates.values()) else 'baseline'
        if selected != 'baseline':
            reason = f'Challenger passed paired confirmation and four-hour endurance: {check}'
    except BaseException as exc:
        selected = 'baseline'
        reason = f'Comparison interrupted: {type(exc).__name__}: {exc}'
        print(reason,flush=True)
    rows = tables(out,runs)
    plots(out,runs)
    decision = {'selected':selected,'challenger':challenger,'reason':reason,'gates':gates,
                'confirmation':check if challenger else {},'baseline_provenance':a.baseline_provenance,
                'archive_sha256':hashes,'freeze_ist':FREEZE.isoformat(),
                'baseline_adapted_on_a100':a.adapt_baseline}
    chosen = out/'selected.zip'
    shutil.copyfile(archives[selected],chosen)
    decision['selected_sha256'] = sha(chosen)
    validation = validate(chosen,out,a.smoke and seconds_left(start)>1500,
                          adapted=selected == 'baseline' and a.adapt_baseline)
    decision['validation'] = validation
    decision['upload_ready'] = (not validation['static_failures'] and
                                validation['execution'] in ('passed','passed_adapted'))
    decision['gpu_wall_seconds'] = time.time()-start
    save(out/'decision.json',decision)
    report(out,decision,rows,archives,decision['gpu_wall_seconds'])
    print(json.dumps({'selected':selected,'selected_zip':str(chosen),'sha256':sha(chosen),
                      'upload_ready':decision['upload_ready'],
                      'validation':validation,'report':str(out/'decision.md')},indent=2),flush=True)
    return 0 if decision['upload_ready'] or not a.smoke else 1


if __name__ == '__main__':
    raise SystemExit(main())
