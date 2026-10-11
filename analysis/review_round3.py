import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "round3_review"
OUT.mkdir(parents=True, exist_ok=True)


def rows(round_number, name):
    with (ROOT / "competition_data" / f"round{round_number}" / name).open() as f:
        return list(csv.DictReader(f))


def participant(data, name="gun-py"):
    return [r for r in data if r["participant_id"] == name]


source = json.loads((ROOT / "competition_data/round3/source.json").read_text())
manifest = json.loads((ROOT / "competition_data/round3/manifest.json").read_text())
verified = []
for entry in manifest["files"]:
    path = ROOT / "competition_data/round3" / entry["path"]
    if path.exists():
        data = path.read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        verified.append(entry["path"])

boards = {r: rows(r, "leaderboard.csv") for r in (2, 3)}
summaries = {r: participant(boards[r])[0] for r in (2, 3)}
runs = rows(3, "runs.csv")
own_runs = sorted(participant(runs), key=lambda r: int(r["run_number"]))
checkpoints = rows(3, "checkpoint_summary.csv")
own_checkpoints = sorted(participant(checkpoints), key=lambda r: int(r["elapsed_minutes"]))
for r in (2, 3):
    scores = [float(x["score_loss"]) for x in participant(rows(r, "runs.csv"))]
    assert len(scores) == 10
    assert abs(statistics.mean(scores) - float(summaries[r]["mean_feasible_loss"])) < 1e-12
    assert abs(statistics.stdev(scores) - float(summaries[r]["score_std"])) < 1e-12

leader, tenth = boards[3][0], next(r for r in boards[3] if r["rank"] == "10")
reference_runs = {
    name: {int(r["run_number"]): r for r in participant(runs, name)}
    for name in (leader["participant_id"], tenth["participant_id"])
}
per_run = []
for r in own_runs:
    n, score = int(r["run_number"]), float(r["score_loss"])
    per_run.append({
        "run_number": n,
        "seed_index": n - 1,
        "gun_py_loss": score,
        "rank_on_topology": 1 + sum(float(x["score_loss"]) < score for x in runs if int(x["run_number"]) == n),
        "leader_loss": float(reference_runs[leader["participant_id"]][n]["score_loss"]),
        "tenth_place_loss": float(reference_runs[tenth["participant_id"]][n]["score_loss"]),
        "time_to_best_minutes": float(r["time_to_best_minutes"]),
        "evaluation_count": int(r["evaluation_count"]),
    })

metrics = {
    "source_revision": source["revision"],
    "retrieved_at_utc": source["retrieved_at_utc"],
    "manifest_verified_files": verified,
    "participant_id": "gun-py",
    "round2": summaries[2],
    "round3": summaries[3],
    "round3_scored_archive_identity": "unverified; not supplied in public result release",
    "round2_exact_portal_archive_verified": False,
    "cross_round_mean_delta_not_paired": float(summaries[3]["mean_feasible_loss"]) - float(summaries[2]["mean_feasible_loss"]),
    "reported_evaluation_count_change_pct": 100 * (float(summaries[3]["mean_evaluation_count"]) / float(summaries[2]["mean_evaluation_count"]) - 1),
    "round3_gap_to_first": float(summaries[3]["mean_feasible_loss"]) - float(leader["mean_feasible_loss"]),
    "round3_gap_to_tenth": float(summaries[3]["mean_feasible_loss"]) - float(tenth["mean_feasible_loss"]),
    "runs_best_after_180_minutes": sum(r["time_to_best_minutes"] > 180 for r in per_run),
    "runs_best_after_237_minutes": sum(r["time_to_best_minutes"] > 237 for r in per_run),
    "round3_per_run": per_run,
    "round3_checkpoints": own_checkpoints,
}
(OUT / "statistics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
with (OUT / "per_run_comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_run[0]))
    writer.writeheader()
    writer.writerows(per_run)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
colors = {"gun-py": "#b45309", tenth["participant_id"]: "#0f766e", leader["participant_id"]: "#475569"}
for name, label in [("gun-py", "gun-py · 18th"), (tenth["participant_id"], f"{tenth['participant_id']} · 10th"), (leader["participant_id"], f"{leader['participant_id']} · 1st")]:
    data = sorted(participant(checkpoints, name), key=lambda r: int(r["elapsed_minutes"]))
    axes[0].plot([int(r["elapsed_minutes"]) for r in data], [float(r["mean_best_feasible_loss"]) for r in data], marker="o", markersize=4, color=colors[name], label=label)
axes[0].set(title="Round 3 mean loss over time", xlabel="Elapsed minutes", ylabel="Best feasible loss (lower is better)", xticks=[15, 60, 120, 180, 240])
axes[0].legend(frameon=False, fontsize=9)
for key, name, label in [("gun_py_loss", "gun-py", "gun-py"), ("tenth_place_loss", tenth["participant_id"], "10th-place participant")]:
    axes[1].plot([r["run_number"] for r in per_run], [r[key] for r in per_run], marker="o", color=colors[name], label=label)
axes[1].set(title="Same-topology final comparison", xlabel="Official run number (seed index + 1)", ylabel="Final feasible loss (lower is better)", xticks=range(1, 11))
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.axhline(0, color="#94a3b8", linewidth=.8, linestyle=":")
    ax.grid(axis="y", alpha=.15)
fig.suptitle("Phase 3: reliable runs, but a substantial solution-quality gap", fontsize=15, fontweight="bold", x=.06, ha="left")
fig.text(.06, .02, "Official Round 3 data · 10 shared topologies · 4 hours each\nIntermediate checkpoints are conservative upper bounds; final scores are exact. Lines connect published checkpoints.", fontsize=9, color="#475569")
fig.tight_layout(rect=[.02, .10, 1, .90])
fig.savefig(OUT / "round3_comparison.png", dpi=180)
plt.close(fig)
print(json.dumps({k: v for k, v in metrics.items() if k not in {"round2", "round3", "round3_per_run", "round3_checkpoints", "manifest_verified_files"}}, indent=2))
