# run an entry against the round-1 hidden topology seeds (or ConstrainedVoyager as a cheap smoke)
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# from competition_data/round1/seeds.csv
SEEDS = [
    (419164888, 771850990), (910574073, 424237334), (891065062, 929044270),
    (720200356, 174477668), (654200764, 541268051), (1038908541, 1844528431),
    (402803193, 1082025484), (842892044, 1072187562), (136269361, 267997051),
    (2139912011, 1732111275),
]
CKPT = [15, 30, 60, 120, 180, 240]


def load_entry(entry_dir):
    from dfbench import OptimizationAlgorithm
    sub = entry_dir / "submission.py"
    spec = importlib.util.spec_from_file_location(f"sub_{entry_dir.name}", sub)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(entry_dir)); sys.path.insert(0, str(ROOT / "l2d"))
    spec.loader.exec_module(m)
    for a in dir(m):
        o = getattr(m, a)
        if isinstance(o, type) and issubclass(o, OptimizationAlgorithm) and o is not OptimizationAlgorithm:
            return o
    raise SystemExit(f"no OptimizationAlgorithm in {sub}")


def uifo_obj(tseed, seconds):
    from dfbench.problems import UIFOProblem
    from dfbench import Objective
    return Objective(UIFOProblem(size=3, topology_seed=tseed, n_frequencies=50),
                     max_time=seconds, save=["is_feasible"], display_mode="log", verbose=0)


def voyager_obj(seconds):
    from dfbench.problems import ConstrainedVoyagerProblem
    from dfbench import Objective
    return Objective(ConstrainedVoyagerProblem(), max_time=seconds,
                     save=["is_feasible"], display_mode="log", verbose=0)


def best_fez_ckpt(obj, budget_min):
    ls = obj.loss_history_reduced
    if not ls: return [float("nan")] * len(CKPT)
    ts = np.array(obj.time_steps)
    lv = np.array([float(l) for l in ls])
    fe = np.array([bool(x) if x is not None else False for x in obj.is_feasible_history])
    out = []
    for cm in CKPT:
        cs = cm * 60.0
        if cs > budget_min * 60.0: out.append(float("nan")); continue
        m = (ts <= cs) & fe
        out.append(float(np.min(lv[m])) if m.any() else float("inf"))
    return out


def run_one(cls, obj, seed, budget_min, label):
    t0 = time.time()
    alg = cls()
    try:
        alg.optimize(obj, random_seed=seed); err = None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    fl = [l for l, fe in zip(obj.loss_history_reduced, obj.is_feasible_history) if fe]
    return {
        "label": label, "wall_s": time.time() - t0, "err": err,
        "evals": int(obj.eval_count),
        "best": float(obj.best_loss) if obj.best_loss is not None else float("inf"),
        "best_fez": float(min(fl)) if fl else float("inf"),
        "n_fez": len(fl),
        "ckpt": best_fez_ckpt(obj, budget_min),
    }


def show(r):
    if r["err"]:
        print(f"  {r['label']:24s} ERROR: {r['err']}"); return
    ck = " ".join(("%+.3f" % v) if v == v and v != float('inf') else "   .   " for v in r["ckpt"])
    print(f"  {r['label']:24s} evals={r['evals']:>7d} best={r['best']:+.4f} "
          f"best_fez={r['best_fez']:+.4f} n_fez={r['n_fez']:>5d} wall={r['wall_s']:>5.0f}s  ck=[{ck}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--voyager", action="store_true")
    g.add_argument("--fast", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    a = ap.parse_args()

    cls = load_entry(ROOT / "entries" / a.entry)
    print(f"loaded {cls.__name__}")
    rows = []
    if a.voyager:
        rows.append(run_one(cls, voyager_obj((a.minutes or 2.0) * 60), 0, a.minutes or 2.0, "voyager"))
    else:
        m = a.minutes or (20.0 if a.fast else 240.0)
        n = a.seeds or (3 if a.fast else 10)
        for i, (t, s) in enumerate(SEEDS[:n]):
            print(f"\n=== {i+1}/{n} topology_seed={t} optimizer_seed={s}  budget={m:.0f}min ===")
            r = run_one(cls, uifo_obj(t, m * 60), s, m, f"seed{i}_t{t}")
            rows.append(r); show(r)

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    fez = [r["best_fez"] for r in rows if not r["err"]]
    if fez:
        print(f"n={len(fez)}  mean={np.mean(fez):+.4f}  median={np.median(fez):+.4f}  "
              f"min={np.min(fez):+.4f}  max={np.max(fez):+.4f}")
    for r in rows: show(r)


if __name__ == "__main__":
    main()
