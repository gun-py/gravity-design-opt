# Champion-based final candidate

Status on 7 October 2026: `dist/champion_v2.zip` is a **validated candidate**, not a selected final submission. Preserve `artifacts/baselines/local_champion.zip` unchanged. The former is built from the latter's `submission.py`, `warm.py`, and `core.py`; `prior.npz` and the remaining support files are byte-identical to the baseline. The only algorithm changes are controlled by four feature names:

| Feature | Change |
|---|---|
| `screen` | Evaluate the initial 36 candidates once instead of twice. |
| `basin` | Carry each trajectory's best feasible point forward and prefer distinct basins at the 25% transition. |
| `state` | Give each trajectory its own Adam count and reset its moments when it is replaced. |
| `reinject` | If local polishing improves the feasible incumbent, resume the final search from that point. |

All four features are on by default in the packaged candidate. `L2D_CHAMPION_FEATURES=screen` enables only the named feature for **local ablation**; an empty value disables all four. The comparison runner records the feature setting. It cannot promote a package tested with an ablation setting, because the ZIP would run with all four features when submitted without that environment variable.

The preserved local Champion SHA-256 is `9b55bb44832d2ee4bfcd7d75b291ff2bc56a891fc30d5ac16f21f55f475041e5`. Its identity relative to portal submission #250 remains unverified. The actual Phase 3 uploaded ZIP remains unidentified from the public results. Download both exact uploads and record their checksums before attributing Phase 2 versus Phase 3 to Champion versus N7.

## Build and checks

```bash
python tools/pack.py champion_v2
python tools/official/validate_submission.py dist/champion_v2.zip
python -m unittest discover -s tests
python tools/compare_final.py --dry-run
```

The static validator accepted the ZIP with no warnings. A clean extracted archive completed a **48-second, two-frequency** CPU UIFO run on a published Round 3 seed: 254 logged evaluations, 161 feasible candidates, best feasible loss `2.849015040122746`. The objective exceeded the nominal 48-second budget by finishing an in-flight evaluation; the local runner marked the run complete. This checks the entry point and search phases. It says nothing about four-hour, 50-frequency H100 performance.

## Comparison workflow

The fresh [holdout seed file](analysis/final_holdout_seeds.csv) contains ten deterministic topology/optimizer pairs absent from the published Round 2 and Round 3 pairs. They are reproducible local holdouts, not organizer hidden seeds. The default runner compares three of these at four hours per archive. It checks both ZIPs, records hashes, settings, GPU, environment, per-run results and logs, resumes only matching completed runs, and keeps earlier attempts. Its 30-hour cap includes 30 minutes of startup allowance per run. The official setting remains 50 frequencies.

First compare the exact Phase 3 upload with the exact earlier Champion upload on the published Phase 3 seed pairs. That establishes whether the apparent cross-round regression is real on those tasks. Then compare the strongest of those with `champion_v2.zip`; freeze the candidate before the full-budget holdout test. Give each comparison a different output directory.

```bash
python tools/compare_final.py --baseline /path/to/exact_champion_upload.zip \
  --candidate /path/to/exact_phase3_upload.zip \
  --seed-file competition_data/round3/seeds.csv \
  --baseline-verified --out results/phase3_vs_champion

python tools/compare_final.py --baseline /path/to/exact_champion_upload.zip \
  --candidate dist/champion_v2.zip \
  --baseline-verified --out results/final_comparison
```

If the exact Phase 3 upload wins the first comparison, use that ZIP as the second comparison's baseline and omit `--baseline-verified` unless its provenance has also been checked. Keep the run settings identical when comparing results.

The runner requires H100 unless `--allow-other-gpu` is explicitly supplied. If only an A100 is available and the original Champion's wide batch cannot fit, `--adapt-baseline --batch 2` uses a benchmark-only microbatch wrapper and labels the outcome adapted. The original archive is unaffected. A100 results should be treated separately from the official H100 environment.

To inspect a public Round 3 weakness or test one switch quickly, point to the public seed file and lower the run time. These experiments can diagnose behavior; promotion remains disabled below the full four-hour budget and for ablation feature settings.

```bash
python tools/compare_final.py --seed-file competition_data/round3/seeds.csv \
  --indices 7,8 --minutes 30 --max-hours 4 \
  --candidate-features basin --allow-other-gpu \
  --baseline artifacts/baselines/local_champion.zip
```

Each run writes `run.json`, `driver.log`, `progress.jsonl`, and a configuration `meta.json`. The paired output includes `decision.json`. If at least three full-budget pairs all complete, the runner writes `selected.zip`: the candidate only if the paired mean improves by at least `0.02`, it wins a majority of pairs, no regression exceeds `0.10`, and the candidate was tested with its packaged default features. Otherwise it retains the baseline. A three-pair decision has substantial uncertainty; ten paired runs are preferable if GPU time permits. The tool never uploads to the competition portal.

The older `tools/choose_submission.py` still has a 30 September cutoff and `tools/compare_n7.py` still uses Round 2 seeds. Use `tools/compare_final.py` for the final comparison. The last submission received by **15 October 2026 Anywhere on Earth** is used for the final evaluation.
