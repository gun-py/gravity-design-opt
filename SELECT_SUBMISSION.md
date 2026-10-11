# Round 3 selection

This document describes the September Round 3 selection experiment. For the
October final candidate and current paired runner, see [FINAL_IMPLEMENTATION.md](FINAL_IMPLEMENTATION.md).

Open `notebooks/colab_select.ipynb` in a **separate** Colab A100 40 GB runtime and run all cells. Give the first cell a path to `dist/l2d_selection_kit.zip` in Drive, or leave `KIT_PATH` empty and select the kit in Colab's upload dialog. Set `OUTPUT_DIR` to a persistent Drive folder. The notebook does not submit anything to Codabench.

The kit contains the current local Champion, two challengers, the public Round 2 seeds, the official validator, and the runner. The local Champion is **not verified as submission #250**. If you download #250 from the portal, set `PORTAL_ZIP` to its Drive path and `PORTAL_SHA256` to the downloaded file's checksum. Set `PORTAL_SOURCE_VERIFIED = True` only when its origin is confirmed. Keep that original archive unchanged.

The notebook checks the A100 model, available memory, float64 JAX, and all archive hashes. It probes the original baseline; if its fixed batch calls run out of 40 GB memory, the comparison uses a benchmark-only microbatch wrapper and labels the results adapted. This does not alter the original ZIP. Every timed topology run starts in a new process and writes `run.json`, logs, a progress trace, and the archive hash to `OUTPUT_DIR`. Only completed runs with matching configuration and hashes are reused.

Screening uses Round 2 indices 0 and 5 for 20 minutes per archive. Confirmation keeps the baseline and compares the frozen challenger on indices 2, 7, and 9 for 45 minutes each. The challenger must lower the paired mean by at least 0.02, win two seeds, avoid a regression over 0.10, and complete a four-hour endurance run on the hardest confirmation topology. Otherwise `selected.zip` is the preserved baseline. The runner skips a stage if it cannot finish before 30 September 2026, 12:00 IST or within 12 GPU-hours from diagnostics.

The Drive output includes `selected.zip`, `decision.md`, `decision.json`, `per_seed.csv`, `comparison.png`, `validation.json`, `archives.json`, and detailed run folders. Upload **only** `selected.zip` after checking the report and SHA-256. A100 results guide selection but do not reproduce the official H100 leaderboard.

Local checks:

```bash
python -m unittest discover -s tests -p 'test_selection.py'
python tools/pack.py champion_repair
python tools/pack.py n5_rescue
python tools/make_selection_kit.py
python tools/choose_submission.py --baseline artifacts/baselines/local_champion.zip --repair dist/champion_repair.zip --rescue dist/n5_rescue.zip --out /tmp/l2d-select-check --dry-run
```
