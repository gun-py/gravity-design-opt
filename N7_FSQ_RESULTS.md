# N7-FSQ submission status

The uploadable archive is `dist/n7_fsq.zip` (4.70 MB). Its SHA-256 is `85754506d9adcd67a126ef21819149ad30ced8cc1965ed0d1e55e4897fa0aebb`. The archive contains one algorithm entry, its required source and prior, and `requirements.txt` with `osqp>=1.0,<2`.

The official static validator accepted the ZIP with zero warnings. Twenty-seven focused and existing tests passed with OSQP 1.1.3. A clean extraction completed a 25-second size-three UIFO run at **two frequencies**: 136 logged evaluations, 113 feasible candidates, and best feasible loss `4.2702681963551825`. This is a code execution check, not a 50-frequency performance result. The machine used for this check is not an A100 or H100.

The official Round 2 `gun-py` result was ninth of 60, with mean feasible loss `0.1678753990`. The exact portal archive is still unverified locally; `artifacts/baselines/local_champion.zip` has SHA-256 `9b55bb44832d2ee4bfcd7d75b291ff2bc56a891fc30d5ac16f21f55f475041e5`. Do not label a replay of that file an exact Round 2 replay unless its checksum matches the portal upload.

| Round 2 seed index | Official `gun-py` loss | N7, 50 frequencies |
|---:|---:|---:|
| 0 | 0.918266 | Pending |
| 1 | -0.246005 | Pending |
| 2 | 0.352875 | Pending |
| 3 | 0.323975 | Pending |
| 4 | 0.423686 | Pending |
| 5 | -0.381481 | Pending |
| 6 | 0.414034 | Pending |
| 7 | -0.249689 | Pending |
| 8 | -0.074437 | Pending |
| 9 | 0.197529 | Pending |

Run the paired A100 comparison in an environment with the competition packages and OSQP installed:

```bash
python -m pip install -r entries/n7_fsq/requirements.txt
python tools/compare_n7.py --baseline /path/to/portal_champion_250.zip --baseline-verified --out results/n7_comparison
```

If the portal archive is unavailable, use `--baseline artifacts/baselines/local_champion.zip` and omit `--baseline-verified`. The runner starts each topology in a fresh process, uses identical seed pairs, enforces a 12-hour cap, keeps failed runs, and writes per-run logs plus `decision.json`. It screens Champion, n6, and n7 on indices 0 and 5 for 20 minutes; confirms Champion versus the lower-mean challenger on indices 2, 7, and 9 for 45 minutes; then runs a four-hour endurance check on the hardest confirmation topology if the challenger qualifies. Promotion requires all three confirmation runs feasible, mean improvement at least `0.02`, two paired wins, regression at most `0.10` on any seed, and endurance success. If those checks fail or remain incomplete, the runner selects the baseline.

The improvement guarantee applies only to accepted local refinement steps: each has an officially logged feasible score lower than its predecessor, and the global feasible incumbent is retained. N7 has **not** yet been measured against Champion or n6 at 50 frequencies and is not established as the better competition submission.
