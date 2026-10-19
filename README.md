# gwopt

optimizer for gravitational-wave detector design (NeurIPS 2026 Learn2Design challenge).

uses warm-starting from ~30k precomputed detector designs + population-based gradient descent with noise annealing, basin tracking, and L-BFGS polishing. there's also an SQP-based local refinement variant for tighter convergence.

## setup

```bash
pip install -e .
```

needs `dfbench>=0.3.3` (the competition simulator) and `optax`.

## quick start

the `submission/` directory is a self-contained competition entry you can zip and upload directly. or use the library:

```python
from gwopt import Prior, build_starts, find_prior, Tracker

prior = Prior(find_prior())
starts = build_starts(objective, k=36, prior=prior, rng_seed=42)
```

the main algorithms live in:
- `gwopt/relay.py` - multi-track relay optimizer with rescue/polish cycles
- `gwopt/local_refine.py` - SQP trust-region local search (extends relay)  
- `gwopt/portfolio.py` - portfolio optimizer with basin formation

## structure

```
gwopt/              core library
  topology.py       UIFO topology parsing
  warmstart.py      prior data loading + donor warm-starts
  optimizer.py      tracker, adam step, utilities
  relay.py          relay optimizer
  local_refine.py   SQP local refinement
  portfolio.py      portfolio optimizer
  box.py            normalized-box projection helpers
  batch.py          auto batch size detection
  prior.npz         precomputed prior (~30k designs)

submission/         self-contained competition entry
tests/              basic tests
```

## how it works

1. **warm-start**: loads prior designs from a 360k GPU-hour exploration campaign. matches topology signatures via hamming distance to find nearest donors.

2. **screening**: evaluates ~36 candidates (donors + prior draws + sobol points), picks the top ~19.

3. **gradient descent**: runs batched adam with decaying noise injection. protects the best candidate from noise (anchor).

4. **basin formation**: clusters around the 3-6 best solutions, spawns local copies with small perturbations.

5. **rescue**: periodically replaces stagnant tracks with fresh donors or mutated archive solutions.

6. **polish**: L-BFGS refinement on the best feasible solutions in bounded unit space.

7. **local SQP** (optional): trust-region SQP with BFGS hessian updates, constraint jacobians, and feasibility restoration.

## license

MIT
