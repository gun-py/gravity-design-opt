# n1–n4: second-generation methods

## The finding that motivated all of them

Measured over the 200 lowest-loss designs in `dataset.h5` (size-3):

```
fraction of params within 1e-3 of a bound:  mean 0.725   median 0.766
|x| needed in sigmoid space:                median 10.23   p90 27.63

  |x|= 5.0 -> sigmoid' = 6.6e-03    79.7% of elite params are beyond this
  |x|=10.0 -> sigmoid' = 4.5e-05    51.3%
  |x|=15.0 -> sigmoid' = 3.1e-07    36.9%
  |x|=20.0 -> sigmoid' = 2.1e-09    26.5%
  |x|=27.6 -> sigmoid' = 1.0e-12    12.3%   (the logit clip)
```

`AdamGD`, `NAAdamGD`, and m1–m5 all call `prepare(obj, unbounded=True)`, so the Objective
maps `x -> lb + (ub-lb)*sigmoid(x)`. The chain rule multiplies every gradient by
`sigmoid'(x)`. **For over half of the parameters that a good design needs, that factor is
below 5e-5, and for a quarter of them below 2e-9.** Gradient descent in sigmoid space
structurally cannot drive a parameter onto its bound — but that is exactly where ~72% of
elite parameters sit. The optimiser spends the whole 4 h fighting its own parameterisation.

That is the gap n3/n4 close.

---

## n3 — BoxProj  (the fix; highest expected value)

`prepare(obj, unbounded=False)`, then optimise in normalised box coords
`u in [0,1]^n`, with `x = lb + u*(ub-lb)`:

```
gu = gx * (ub - lb)              # chain rule, linear -- no vanishing factor
u  = clip(u + adam_update + noise, 0, 1)
```

A parameter can now land **exactly** on its bound and stay there. Normalising by
`(ub-lb)` also fixes the scale mismatch (reflectivity ∈ [0,1] vs length ∈ [0.1, 4000]),
which a single learning rate otherwise cannot serve.

16 warm starts, row 0 protected, noise annealed to 0 by 55% of budget, stall-triggered
reseed from the best-feasible incumbent.

## n1 — Halving  (budget allocation)

Successive halving. Every baseline splits the budget uniformly across its trajectories,
so ~90% of compute goes to trajectories that were already losing at 10% of budget. The
round-1 data says a 60-min probe predicts final per-run rank at rho = 0.82 — i.e. you can
prune early and safely.

```
value-only screen of 64 -> 32
 6% budget: 32 traj -> keep 16
15%:        16       -> keep 8
32%:         8       -> keep 4
58%:         4       -> keep 2
100%:        2 finalists, no noise
```

Ranking at each rung uses each trajectory's own best feasible loss (infeasible points get
+5 so a feasible-but-worse trajectory outranks an infeasible-but-lower one). Uses the n3
box parameterisation.

## n2 — ALM  (exact constrained optimisation)

The score is `sensitivity_loss` **subject to** `power <= threshold`, and
`is_feasible <=> penalty == 0`. So optimise that constrained problem directly with a
Powell–Hestenes–Rockafellar augmented Lagrangian rather than a fixed penalty:

```
g_i(x) = power_i / threshold_i - 1                       # <=0 means feasible
L      = sensitivity_loss + (1/2rho) * sum_i [ max(0, lam_i + rho*g_i)^2 - lam_i^2 ]
lam_i <- max(0, lam_i + rho*g_i)                          # dual ascent, every 40 steps
rho   <- 2*rho    if max violation did not drop by 25%
```

Why this beats a fixed penalty: the default `squashed_relu_penalty` is bounded in [0,1),
so once you are well outside the feasible set its gradient is ~0 — it cannot pull you
back. And any *fixed* penalty converges to a biased point, not the constrained optimum.
Multipliers let iterates settle **on** the active constraint boundary, which is where the
best designs live (elite designs have power constraints active).

Falls back to plain projected descent on problems with no aux objective.

## n4 — Fusion  (the submission candidate)

n3 parameterisation + n1 schedule + n2 tail:

```
screen 48 -> 24
 8% budget: 24 -> 12
20%:        12 -> 6
40%:         6 -> 3
65%:         3 finalists
80%->end:   augmented Lagrangian on the finalists
tail:       plain projected descent, no noise, to the wire
```

Every phase is wrapped so a failure degrades to plain projected descent; the container
never returns early.

---

## What to measure

The one experiment that decides everything (section 4 of the Colab notebook):

```bash
python tools/bench.py m1_warmpack --fast --seeds 3 --minutes 20   # sigmoid
python tools/bench.py n3_boxproj  --fast --seeds 3 --minutes 20   # box-projected
```

Same warm starts, same noise schedule, same budget — only the parameterisation differs.
If n3 wins, ship n4. If it does not, the sigmoid-saturation argument is wrong in practice
and you should fall back to `champion.zip`.

Then the sweep:

```bash
for e in m1_warmpack n1_halving n2_alm n3_boxproj n4_fusion; do
  python tools/bench.py $e --fast --seeds 3 --minutes 20
done
```

| bar | round-1 mean |
|---|---:|
| winner | **0.0197** |
| rank 3 | 0.0916 |
| rank 10 | 0.365 |
| NAAdamGD | 0.504 |
| RandomSearch (fallback) | 4.749 |
