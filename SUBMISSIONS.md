# Learn2Design 2026 — Submission Kit

Six ZIPs in `dist/`, ranked by expected score:

| Entry           | ZIP                    | Idea                                                                    |
| --------------- | ---------------------- | ----------------------------------------------------------------------- |
| **champion**    | `dist/champion.zip`    | M1+M2+M3 with row-0 protection, phased schedule, L-BFGS polish. Primary |
| m1_warmpack     | `dist/m1_warmpack.zip` | 12-wide dataset-warm-started Adam multi-start                           |
| m2_probe        | `dist/m2_probe.zip`    | 36→19→18 portfolio → probe → commit schedule                            |
| m3_barrier      | `dist/m3_barrier.zip`  | Soft barrier on `aux["power_values"]` + L-BFGS polish                   |
| m4_priorgen     | `dist/m4_priorgen.zip` | Kernel-mixture generative prior in quantile space (`weights.npz`)       |
| m5_surro        | `dist/m5_surro.zip`    | ReSTIR kNN surrogate + reservoir importance sampling                    |

Upload the one you want scored at <https://submit.learn2design2026.com/competitions/4/>.
Only the last submission before each deadline is evaluated → upload the champion for scoring rounds; keep M1 as the ranked fallback.

---

## Layout

```
L2D_Gravity/
├── SUBMISSIONS.md
├── l2d/                     shared foundation (copied into every ZIP root)
│   ├── name_gen.py          pure-Python UIFO pair generator (no differometor)
│   ├── warm.py              eval-time topology → warm-start vectors
│   ├── core.py              Tracker, Adam+capped-noise step, run_to_wire, insure
│   └── prior.npz            4.7 MB — 444 keys, 3365 donors, per-key top-decile samples
├── tools/
│   ├── build_prior.py       -> l2d/prior.npz  (~2 s)
│   ├── train_priorgen.py    -> entries/m4_priorgen/weights.npz  (~10 s)
│   ├── pack.py              build & validate every dist/*.zip
│   └── bench.py             run any entry on Voyager / round-1 seeds
├── entries/
│   ├── m1_warmpack/submission.py
│   ├── m2_probe/submission.py
│   ├── m3_barrier/submission.py
│   ├── m4_priorgen/{submission.py, weights.npz}
│   ├── m5_surro/submission.py
│   └── champion/submission.py
└── dist/*.zip               6 submission ZIPs
```

Every ZIP has, at root: `submission.py`, `requirements.txt`, `name_gen.py`, `warm.py`, `core.py`, `prior.npz` (+ `weights.npz` for m4).

`requirements.txt` pins only `optax>=0.2.0`. Never pin `jax`/`jaxlib`/`dfbench` — the evaluator ships CUDA-13 builds.

---

## Rebuild everything

```bash
python tools/build_prior.py           # l2d/prior.npz  (~2 s)
python tools/train_priorgen.py        # entries/m4_priorgen/weights.npz  (~10 s)
python tools/pack.py all              # dist/*.zip
```

---

## Validation

Three tiers. Do them in order.

### Tier 1 — structural (seconds, no GPU)

```bash
python tools/pack.py all
```
Checks each entry: exactly one `OptimizationAlgorithm` subclass, no `import differometor`, imports load cleanly from a scratch cwd, ZIP < 50 MB. If any entry fails, the message tells you what.

### Tier 2 — mechanics smoke (2 min each, CPU)

```bash
for e in m1_warmpack m2_probe m3_barrier m4_priorgen m5_surro champion; do
  python tools/bench.py $e --voyager --minutes 2
done
```
Runs each entry on `ConstrainedVoyagerProblem` (small, ~48 params, cheap eval). What to check per entry:
- **evals > 0** — the algorithm actually iterated.
- **No exception in the summary** (`err` field is `None`).
- **`best` is finite** — no NaN escaped `nan_guard`.

CPU + batched vmap means feasibility usually stays at 0 on Voyager in 2 min. That's fine — feasibility on the light problem is *not* a signal of quality. Reach for Tier 3 for that.

### Tier 3 — real UIFO on a GPU (Kaggle H100 or A100)

Install on the GPU box:
```bash
pip install dfbench differometor optax
# jax/jaxlib should already match your CUDA. If not: pip install "jax[cuda12]"
```

Quick iteration (3 seeds × 20 min ≈ 1 GPU-hour):
```bash
python tools/bench.py champion --fast
```

Full-budget run (10 seeds × 4 h = 40 GPU-hours, matches evaluator):
```bash
python tools/bench.py champion --full
```

Every seed prints one row:
```
seedK_t<topology_seed>   evals=NNNN   best=+X.XXXX   best_fez=+X.XXXX   n_fez=NNN   wall=NNNs   ck=[15,30,60,120,180,240 min]
```
Followed by a mean/median/min/max summary across seeds.

Compare **`mean best_fez`** to `competition_data/round1/leaderboard.csv`:
| Bar          | Round-1 mean |
| ------------ | -----------: |
| winner       |       0.0197 |
| rank 3       |       0.0916 |
| rank 10      |        0.365 |
| NAAdamGD     |        0.504 |
| RandomSearch |        4.749 |

**Success criteria:**
- **m1_warmpack** → mean ≤ **0.20** (beats podium rank-3)
- **champion**     → mean ≤ **0.05**, no run > 0.4 (podium)

If m1_warmpack's mean sits between 1–3, check that `problem_spec` reveals the topology — sometimes the eval container's Objective wrapper is slightly different. Run:
```bash
python -c "from dfbench.problems import UIFOProblem; from dfbench import Objective; p=UIFOProblem(size=3,topology_seed=419164888,n_frequencies=50); o=Objective(p,max_evals=1); print(o.problem_spec)"
```
You should see `params.topology` as a 21-character string. If missing, `build_starts` falls back to prior/random and the warm-start advantage disappears.

### Tier 4 — sanity checks that already passed

Kept for reference; you can re-run any time.

```bash
# name-gen matches dfbench's own reconstruct_uifo_setup on 15 topologies
python -c "
import sys; sys.path[:0]=['dataset/examples','l2d']
from dataset_utils import reconstruct_uifo_setup
from name_gen import pairs_for_topology
import h5py, numpy as np, random
random.seed(0)
h=h5py.File('dataset/dataset.h5','r'); e=h['entries'][:]
s3=list(np.where(e['size']==3)[0]); s4=list(np.where(e['size']==4)[0])
for i in random.sample(s3,12)+random.sample(s4,3):
  t,s=e[i]['topology_string'].decode(),int(e[i]['size'])
  _,tp=reconstruct_uifo_setup(t,s); gp=pairs_for_topology(t,s)
  ok=all(list(a)==list(b) for a,b in zip(tp,gp)) and len(tp)==len(gp)
  assert ok, (t,s)
print('OK 15/15')
"

# donor round-trip: reconstruct donor 0's vector, eval via UIFOProblem, delta should be ~1e-6
python -c "
import sys; sys.path.insert(0,'l2d')
import numpy as np, jax.numpy as jnp
from warm import Prior; from name_gen import pairs_for_topology, canonical_key
from dfbench.problems import UIFOProblem; from dfbench import Objective
p=Prior('l2d/prior.npz'); d=0
topo=p.donor_topo[d]; sz=int(p.donor_size[d]); stored=float(p.donor_loss[d])
kids=p.donor_key_id[p.donor_offsets[d]:p.donor_offsets[d+1]]
vals=p.donor_value[p.donor_offsets[d]:p.donor_offsets[d+1]]
lut=dict(zip(kids.tolist(),vals.tolist()))
kti={k:i for i,k in enumerate(p.keys_arr)}
vec=np.array([lut[kti[f'{k[0]}/{k[1]}']] for k in [canonical_key(pp) for pp in pairs_for_topology(topo,sz)]])
o=Objective(UIFOProblem(size=sz,topology=topo,n_frequencies=50),max_evals=1,verbose=0,display_mode='log')
o.warmup_value(); o.start_logging()
got=float(o.value(jnp.asarray(vec)))
print(f'stored={stored:+.6f}  eval={got:+.6f}  delta={got-stored:+.2e}')
assert abs(got-stored)<1e-4, 'donor round-trip diverges'
print('OK donor round-trip within 1e-4')
"
```

---

## What each entry does (short)

- **m1_warmpack**: 12 unbounded starts (8 NN-donor + 3 prior + 1 uniform). Row 0 = best donor, never noised. Clipped-Adam lr 0.1 with Gaussian noise capped at 0.25·‖update‖, annealed to 0 by 50 % of budget. Stall-triggered reseed of worst half from best-feasible. `insure` every 200 iters.
- **m2_probe**: same primitives, explicit time phases: 36-wide value screen (0–5 %) → 19-wide gradient batch (5–25 %) → probe → 3 basins × 6 replicas (25–87.5 %) → noise=0 refinement.
- **m3_barrier**: custom-JIT loop on `value_function_aux(unbounded=True)`, optimizing `aux.sensitivity_loss + β·softplus_barrier(power_values)`. Logs the true objective via `log_evaluation`. Last 12 % is L-BFGS on the best-feasible.
- **m4_priorgen**: 18 M1 donors + 18 samples from a kernel-mixture generative model (Nadaraya-Watson in topology-feature space, per-key sampling in quantile space, inverse-CDF via `prior.npz`). Value-only screen → top 12 into the M1 loop.
- **m5_surro**: ReSTIR kNN surrogate over 400-point reference set, 10k candidates/iter, importance-weighted reservoir sampling, 12-wide Adam refine, running elite pool.
- **champion**: composed schedule — value screen → 19-wide capped-noise → 3-basin replication → L-BFGS polish → noise-free descent to the wire. Row 0 protected everywhere. Every phase catches exceptions and falls through to plain descent so the container never returns early.

---

## Scoring hazards handled

- **`log_evaluation(aux=None)` hides feasibility** → every custom-JIT logger passes the aux pytree.
- **Batched-feasibility scoring hazard** → `insure_incumbent` re-evals the incumbent as a single-point `value_aux` every 200 iters so its feasibility is logged as its own history entry.
- **`set_penalty_fn` custom callable breaks `to_spec()`** → `core.set_penalty` only permits registered presets.
- **Early return forfeits budget** (`oneday` in round 1: 98 %) → every entry uses `run_to_wire` and catches exceptions inside phases.
- **Tied length groups in `optimization_pairs`** (nested lists) → `name_gen.canonical_key` reduces to the first pair's name.
