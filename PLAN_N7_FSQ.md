# Implement N7-FSQ as a competition submission

## Summary

Build `n7_fsq`, a new algorithm that retains n6’s broad search and adds a refiner that uses the gradients of both sensitivity loss and physical power constraints. Its local steps use a quadratic model and are accepted only after the official Objective confirms feasibility and a lower loss.

The mathematical guarantees concern accepted local steps: the stored feasible incumbent cannot worsen, and each accepted refinement improves its evaluated loss. No algorithm can guarantee a lower four-hour result than Champion on every unseen nonconvex topology. That claim requires benchmarking.

## Implementation

**Submission entry.** Add `entries/n7_fsq/submission.py` with exactly one `OptimizationAlgorithm` subclass and the standard `optimize(objective, init_params=None, random_seed=None, **kw)` interface. Write zero comments and zero docstrings in this file. Keep it limited to imports, the class, and construction of the optimizer. Put the algorithm in `l2d/fsq.py`; use only occasional short comments there when the equations would otherwise be unclear.

**Objective adapter.** Optimize live parameters as normalized coordinates \(u\in[0,1]^n\), mapping to physical parameters by \(x=l+(h-l)\odot u\). Define sensitivity loss \(f(u)\) and one normalized constraint per power reading, \(c_i(u)=P_i(x)/T_i-1\). Obtain thresholds and power readings through the public Objective API. Use `value_aux` for trials. After logging starts, use `value_function_aux(unbounded=False)` and JAX VJPs for derivatives of sensitivity and selected constraints. Process constraint derivatives in fixed chunks of eight rows to limit memory. Synchronize and log each completed raw evaluation with its actual official score and auxiliary data; never log a model or restoration value as the score. The ordinary `value_and_grad_aux` gradient belongs to the penalized loss and must not be used as \(\nabla f\).

**Local solver.** For each feasible basin, maintain a positive-definite curvature matrix \(B\), multipliers, working set, and trust radius. Begin with \(B=I\) and radius `0.05`. Include constraints within \(\max(10^{-4},\min(0.02,\Delta^2))\) of their limits, constraints with multiplier above `1e-8`, and constraints violated by the previous trial. Process all selected rows; never drop an active row to meet a fixed cap.

Solve the convex quadratic proposal

\[
\min_d\;g^\top d+\tfrac12d^\top Bd
\quad\text{subject to}\quad
c_W+J_Wd\le0,\qquad
\max(-u,-\Delta)\le d\le\min(1-u,\Delta).
\]

Use OSQP with float64, warm starts, `eps_abs=eps_rel=1e-8`, `max_iter=4000`, polishing enabled, and a `0.5`-second solve limit. Independently reject inaccurate, nonfinite, or non-descent solutions. Try step lengths `1`, `1/2`, and `1/4`.

If the full step improves the model but violates nonlinear constraints, allow up to two bounded second-order correction attempts using the measured violation. A correction’s total infinity norm may not exceed \(\min(\Delta/2,2\|d\|_\infty^2)\).

Every proposed point must pass the official feasibility check and Armijo decrease check

\[
f(u+s)\le f(u)+10^{-4}g^\top s,\qquad g^\top s<0.
\]

Also require its official score to improve by more than \(10^{-10}\max(1,|\text{current score}|)\).

**Feasible-direction safeguard.** If the SQP step fails, solve

\[
\begin{aligned}
\min_{d,t}\quad&t+\tfrac12d^\top\widehat B d\\
\text{subject to}\quad&g^\top d/a\le t,\quad c_W+J_Wd\le t,\quad t\le0,\\
&\max(-u,-\Delta)\le d\le\min(1-u,\Delta),
\end{aligned}
\]

where \(a=\max(1,\|g\|_\infty)\) and \(\widehat B\) is \(B/a\) with eigenvalues clipped to `[1e-4, 1e4]`. The zero step is feasible. An exact solution with negative objective value gives strict objective descent and moves inward from active constraints; sufficiently small steps are feasible under smoothness assumptions. Check the actual point with the official Objective, backtracking at most 12 times. A practical run may still stall because backtracking and time are capped.

After an accepted SQP step, update \(B\) with a Powell-damped BFGS update to the Lagrangian gradient, using the same QP multipliers at both endpoints. Skip invalid updates, symmetrize, and clip eigenvalues to `[1e-6, 1e6]`. Skip curvature updates after safeguard steps. Double the trust radius, up to `0.25`, after a strong accepted step that reaches its edge; halve it for a poor model ratio; multiply it by `0.25` after an unsuccessful iteration. Stop a basin after three failures at radius `1e-6`.

**Infeasible starts.** Use projected gradient descent on \(H(u)=\tfrac12\sum_i\max(c_i(u),0)^2\), with Armijo backtracking. Stop when officially feasible, after 25 accepted steps, or after 60 seconds per point. Cap all restoration at 5% of the four-hour budget.

**Population and time.** Keep n6’s 36 structured starts, 19 screened trajectories, six distinct basins with 18 trajectories, and rescues at 40%, 60%, and 80%. Use the least-violating candidates if screening finds no feasible point. Spend the first 25% on exploration; from 25% to 85%, alternate 120-second population and local slices, visiting basins in round-robin order; spend the final 15% refining the best three distinct feasible basins. Feed accepted local points back into their population anchors with fresh Adam state. Return unused local time to population descent. Reserve at least ten seconds, or twice recent 95th-percentile evaluation latency, for final confirmation.

On an invalid QP result, try the safeguard. On repeated nonfinite derivatives, reset that basin’s curvature; after three such failures, suspend its local solver for the slice. Release derivative tapes between basins. Reduce population microbatch size on recognized memory failures, at most twice. Surface other errors.

## Validation and selection

Add focused tests for derivative agreement with finite differences, constraint ordering, fixed coordinates, curvature positivity, QP failure, infeasible restoration, logging admission, budget termination, partial batches, and incumbent preservation. Include a curved-boundary case such as \(x^2+y^2\le1\), where a tangent straight-line step is infeasible but correction or the feasible-direction safeguard can progress. Verify that every accepted local iterate is officially feasible and improves its evaluated score.

Validate a cleanly extracted ZIP with the official static validator and a short execution check. A reduced-frequency smoke test checks code execution; performance checks must use 50 frequencies.

Prepare paired comparisons against the preserved Champion baseline and n6, with fresh processes and identical seed pairs. Within the previously chosen 12 A100-hour cap, use two 20-minute screening seeds, three untouched 45-minute confirmation seeds for baseline versus the frozen challenger, and one four-hour endurance run. Keep failures in the results. Promote n7 only if all confirmation runs are feasible, paired mean loss improves by at least `0.02`, it wins at least two of three pairs, no pair regresses by more than `0.10`, and endurance succeeds.

Deliver the saved Markdown plan, one validated n7 ZIP and checksum, test results, per-seed losses, and a candid comparison. “Mathematically safeguarded” describes its accepted steps; “better submission” is a measured outcome.

## Assumptions

- The competition remains 10 hidden size-three topologies, 50 frequencies, four hours per topology on one H100.
- The public dataset prior already packaged with n6 remains available; no model training is required.
- Start with population microbatch two and derivative chunks of eight.
- Package `osqp>=1.0,<2`; use the evaluator’s float64 JAX and SciPy environment.
- Preserve existing Champion and n6 archives and avoid seed-specific submission behavior.
