#!/usr/bin/env python3
"""Exploratory data analysis of the Differometor-30k UIFO design corpus.

Reads dataset/dataset.h5 and writes figures to analysis/figures/.
Run from the repo root:  python analysis/eda_dataset.py
"""
from __future__ import annotations

import os
from collections import Counter

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(ROOT, "dataset", "dataset.h5")

R1_BEST = 0.504      # round-1 winning score (NAAdamGD)
ACCENT = "#2b6cb0"
ACCENT2 = "#dd6b20"
GOOD = "#2f855a"
BAD = "#c53030"

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
print("loading", DATA)
f = h5py.File(DATA, "r")
e = f["entries"][:]
def col(name):
    c = e[name]
    return np.array([x.decode("utf-8", "replace") for x in c]) if c.dtype.kind == "S" else c

loss = col("loss").astype(float)
size = col("size").astype(int)
complexity = col("complexity").astype(int)
topo = col("topology_string")
uhash = col("unique_hash")
parent = col("initialized_from")
freqs = f["frequency_values"][:]
sens = f["sensitivities"][:]              # (29650, 50)
N = len(loss)
print(f"{N} designs, sensitivities {sens.shape}, freqs {freqs.shape}")

# topology parts
topoA = np.array([t.split("-")[0] for t in topo])
topoB = np.array([t.split("-")[1] if "-" in t else "" for t in topo])

# ----------------------------------------------------------------------------
# Figure 1 — Loss landscape
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.22)
fig.suptitle("Differometor-30k — Loss Landscape (29,650 optimized UIFO designs)",
             fontsize=14, fontweight="bold")

# (a) histogram
ax = fig.add_subplot(gs[0, 0])
ax.hist(loss, bins=120, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.2)
for x, c, lbl in [(0.0, GOOD, "loss = 0"), (R1_BEST, BAD, f"round-1 best = {R1_BEST}"),
                  (np.median(loss), "#666", f"median = {np.median(loss):.2f}")]:
    ax.axvline(x, color=c, ls="--", lw=1.5, label=lbl)
ax.set(xlabel="loss (lower is better)", ylabel="# designs", title="(a) Loss distribution")
ax.legend(fontsize=8)

# (b) loss by size — violin
ax = fig.add_subplot(gs[0, 1])
data = [loss[size == 3], loss[size == 4]]
parts = ax.violinplot(data, showmedians=True, showextrema=False)
for pc, c in zip(parts["bodies"], [ACCENT, ACCENT2]):
    pc.set_facecolor(c); pc.set_alpha(0.6)
parts["cmedians"].set_color("black")
ax.set_xticks([1, 2], [f"3×3\n(n={ (size==3).sum() })", f"4×4\n(n={ (size==4).sum() })"])
ax.axhline(R1_BEST, color=BAD, ls="--", lw=1.2)
ax.axhline(0, color=GOOD, ls="--", lw=1.2)
ax.set(ylabel="loss", title="(b) Loss by UIFO grid size")

# (c) CDF
ax = fig.add_subplot(gs[1, 0])
xs = np.sort(loss)
ys = np.arange(1, N + 1) / N
ax.plot(xs, ys, color=ACCENT, lw=2)
p_neg = (loss < 0).mean(); p_r1 = (loss < R1_BEST).mean()
ax.axvline(0, color=GOOD, ls="--", lw=1.3)
ax.axvline(R1_BEST, color=BAD, ls="--", lw=1.3)
ax.annotate(f"{p_neg*100:.1f}% < 0\n({(loss<0).sum()} designs)", (0, p_neg),
            xytext=(0.6, 0.18), fontsize=8.5, color=GOOD,
            arrowprops=dict(arrowstyle="->", color=GOOD))
ax.annotate(f"{p_r1*100:.1f}% beat\nround-1 best\n({(loss<R1_BEST).sum()} designs)",
            (R1_BEST, p_r1), xytext=(1.6, 0.42), fontsize=8.5, color=BAD,
            arrowprops=dict(arrowstyle="->", color=BAD))
ax.set(xlabel="loss threshold", ylabel="fraction of designs ≤ threshold",
       title="(c) Cumulative distribution")

# (d) loss vs complexity (hexbin)
ax = fig.add_subplot(gs[1, 1])
hb = ax.hexbin(complexity, loss, gridsize=45, cmap="viridis", bins="log", mincnt=1)
fig.colorbar(hb, ax=ax, label="log10(count)")
# trend: median loss per complexity bin
bins = np.linspace(complexity.min(), complexity.max(), 25)
idx = np.digitize(complexity, bins)
mx, my = [], []
for b in range(1, len(bins)):
    m = idx == b
    if m.sum() > 20:
        mx.append(complexity[m].mean()); my.append(np.median(loss[m]))
ax.plot(mx, my, color="white", lw=2.5); ax.plot(mx, my, color=BAD, lw=1.5, label="median loss")
ax.set(xlabel="complexity (# parameters)", ylabel="loss",
       title="(d) Loss vs. design complexity")
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "01_loss_landscape.png"), bbox_inches="tight")
plt.close(fig)
print("wrote 01_loss_landscape.png")

# ----------------------------------------------------------------------------
# Figure 2 — Topology structure
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)
fig.suptitle("Topology Structure & Reuse", fontsize=14, fontweight="bold")

# per-topology aggregation
uniq_topo, inv = np.unique(topo, return_inverse=True)
n_per = np.bincount(inv)
best_per = np.full(len(uniq_topo), np.inf)
np.minimum.at(best_per, inv, loss)

# (a) designs per topology
ax = fig.add_subplot(gs[0, 0])
vals, counts = np.unique(n_per, return_counts=True)
ax.bar(vals[:20], counts[:20], color=ACCENT, alpha=0.85)
ax.set(xlabel="# designs sharing a topology", ylabel="# topologies (log)",
       title=f"(a) Reuse: {len(uniq_topo):,} unique topologies", yscale="log")
ax.text(0.97, 0.9, f"max reuse = {n_per.max():,}×\nmedian = 1", transform=ax.transAxes,
        ha="right", fontsize=8.5, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# (b) best loss per topology
ax = fig.add_subplot(gs[0, 1])
ax.hist(best_per, bins=100, color=ACCENT2, alpha=0.85, edgecolor="white", linewidth=0.2)
ax.axvline(0, color=GOOD, ls="--", lw=1.3, label="loss = 0")
ax.axvline(R1_BEST, color=BAD, ls="--", lw=1.3, label=f"round-1 best")
ax.set(xlabel="best loss achieved for that topology", ylabel="# topologies",
       title="(b) Best-in-topology loss")
ax.legend(fontsize=8)

# (c) node-component code frequency (A part)
ax = fig.add_subplot(gs[1, 0])
cA = Counter("".join(topoA.tolist()))
codes = sorted(cA)
ax.bar(codes, [cA[c] for c in codes], color=ACCENT, alpha=0.85)
ax.set(xlabel="node-component code", ylabel="total occurrences",
       title="(c) Node-component usage (A-part, codes A–H)")

# (d) connection code freq (B part) + mean loss by dominant code
ax = fig.add_subplot(gs[1, 1])
cB = Counter("".join([b for b in topoB.tolist() if b]))
codes = sorted(cB)
ax.bar(codes, [cB[c] for c in codes], color=ACCENT2, alpha=0.85)
ax.set(xlabel="connection code", ylabel="total occurrences",
       title="(d) Connection usage (B-part, codes S/L/D/H)")
fig.savefig(os.path.join(FIG, "02_topology_structure.png"), bbox_inches="tight")
plt.close(fig)
print("wrote 02_topology_structure.png")

# ----------------------------------------------------------------------------
# Figure 3 — Sensitivity spectra (the physics payload)
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)
fig.suptitle("Sensitivity Spectra (50 frequency points per design)", fontsize=14, fontweight="bold")

order = np.argsort(loss)
best_idx = order[:50]
worst_idx = order[-50:]

# (a) best vs worst overlaid
ax = fig.add_subplot(gs[0, 0])
for i in best_idx:
    ax.plot(freqs, sens[i], color=GOOD, alpha=0.15, lw=0.8)
for i in worst_idx:
    ax.plot(freqs, sens[i], color=BAD, alpha=0.12, lw=0.8)
ax.plot(freqs, np.median(sens[best_idx], axis=0), color=GOOD, lw=2.5, label="best-50 median")
ax.plot(freqs, np.median(sens[worst_idx], axis=0), color=BAD, lw=2.5, label="worst-50 median")
ax.set(xscale="log", yscale="log", xlabel="frequency", ylabel="sensitivity",
       title="(a) Best-50 vs worst-50 designs")
ax.legend(fontsize=8)

# (b) heatmap sorted by loss
ax = fig.add_subplot(gs[0, 1])
logs = np.log10(np.clip(sens[order], 1e-30, None))
im = ax.imshow(logs, aspect="auto", cmap="magma", origin="lower",
               extent=[freqs.min(), freqs.max(), 0, N])
fig.colorbar(im, ax=ax, label="log10(sensitivity)")
ax.set(xscale="log", xlabel="frequency", ylabel="designs, sorted by loss (best→worst ↑)",
       title="(b) Spectrum heatmap")

# (c) median spectrum by loss quartile
ax = fig.add_subplot(gs[1, 0])
q = np.quantile(loss, [0, .25, .5, .75, 1.0])
cmap = plt.cm.viridis(np.linspace(0, 0.9, 4))
for k in range(4):
    m = (loss >= q[k]) & (loss <= q[k+1])
    ax.plot(freqs, np.median(sens[m], axis=0), color=cmap[k], lw=2,
            label=f"Q{k+1} (loss {q[k]:.2f}–{q[k+1]:.2f})")
ax.set(xscale="log", yscale="log", xlabel="frequency", ylabel="median sensitivity",
       title="(c) Median spectrum by loss quartile")
ax.legend(fontsize=8)

# (d) spectrum of the single best design
ax = fig.add_subplot(gs[1, 1])
b = order[0]
ax.plot(freqs, sens[b], color=ACCENT, lw=2, marker="o", ms=3)
ax.set(xscale="log", yscale="log", xlabel="frequency", ylabel="sensitivity",
       title=f"(d) Global best design  (loss={loss[b]:.4f}, size={size[b]})")
ax.text(0.05, 0.05, f"topology:\n{topo[b]}", transform=ax.transAxes, fontsize=8,
        va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.85))
fig.savefig(os.path.join(FIG, "03_sensitivity_spectra.png"), bbox_inches="tight")
plt.close(fig)
print("wrote 03_sensitivity_spectra.png")

# ----------------------------------------------------------------------------
# Figure 4 — Reoptimization / lineage
# ----------------------------------------------------------------------------
hash2loss = {h: l for h, l in zip(uhash, loss)}
has_parent = np.array([p != "" and p in hash2loss for p in parent])
child_loss = loss[has_parent]
par_loss = np.array([hash2loss[p] for p in parent[has_parent]])
delta = child_loss - par_loss   # negative = improved

fig = plt.figure(figsize=(13, 5))
gs = GridSpec(1, 2, figure=fig, wspace=0.25)
fig.suptitle(f"Reoptimization / Warm-Start Lineage  ({has_parent.sum():,} child designs)",
             fontsize=13, fontweight="bold")

ax = fig.add_subplot(gs[0, 0])
lo = min(par_loss.min(), child_loss.min()); hi = max(par_loss.max(), child_loss.max())
ax.scatter(par_loss, child_loss, s=8, c=np.where(delta < 0, GOOD, BAD), alpha=0.4)
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change")
ax.set(xlabel="parent loss", ylabel="child (reoptimized) loss",
       title=f"(a) Child vs parent — {(delta<0).mean()*100:.0f}% improved")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 1])
ax.hist(delta, bins=60, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.2)
ax.axvline(0, color="k", ls="--", lw=1.2)
ax.axvline(np.median(delta), color=BAD, ls="-", lw=1.5, label=f"median Δ={np.median(delta):.3f}")
ax.set(xlabel="Δloss (child − parent);  negative = better", ylabel="# designs",
       title="(b) Improvement from reoptimization")
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "04_reoptimization.png"), bbox_inches="tight")
plt.close(fig)
print("wrote 04_reoptimization.png")

# ----------------------------------------------------------------------------
# Figure 5 — Parameter structure (bounded_params are sigmoid-mapped to [0,1])
# ----------------------------------------------------------------------------
# use the single most-reused topology so all vectors share length & meaning
top_topo = uniq_topo[np.argmax(n_per)]
sel = np.where(topo == top_topo)[0]
poff = e["param_offset"]; plen = e["param_length"]
bp = f["bounded_params"]
P = np.stack([bp[poff[i]:poff[i]+plen[i]] for i in sel])   # (n_sel, n_params)  PHYSICAL units
sel_loss = loss[sel]
print(f"most-reused topology {top_topo}: {len(sel)} designs, {P.shape[1]} params")

# normalize each parameter column to its observed [0,1] range (params are in
# heterogeneous physical units: powers ~1e3 W, reflectivities ~[0,1], distances, ...)
pmin = P.min(0); pmax = P.max(0)
rng = np.where(pmax > pmin, pmax - pmin, 1.0)
Pn = (P - pmin) / rng
const_mask = pmax <= pmin   # parameters fixed across all designs of this topology

fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)
fig.suptitle(f"Parameter Structure  (physical bounded_params, most-reused topology, "
             f"n={len(sel):,}, dim={P.shape[1]})", fontsize=13, fontweight="bold")

# (a) distribution of each param normalized to its observed range
ax = fig.add_subplot(gs[0, 0])
ax.hist(Pn[:, ~const_mask].ravel(), bins=80, color=ACCENT, alpha=0.85,
        edgecolor="white", linewidth=0.2)
ax.set(xlabel="value, normalized to each parameter's observed range", ylabel="count",
       title="(a) Position within observed per-parameter range")
frac_edge = ((Pn[:, ~const_mask] < 0.02) | (Pn[:, ~const_mask] > 0.98)).mean()
ax.text(0.5, 0.9, f"{frac_edge*100:.0f}% at an extreme of their range\n"
        f"{const_mask.sum()} of {P.shape[1]} params are fixed",
        transform=ax.transAxes, ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# (b) per-parameter normalized mean ± std across designs
ax = fig.add_subplot(gs[0, 1])
mu = Pn.mean(0); sd = Pn.std(0)
xi = np.arange(P.shape[1])
ax.plot(xi, mu, color=ACCENT, lw=1)
ax.fill_between(xi, mu-sd, mu+sd, color=ACCENT, alpha=0.25)
ax.set(xlabel="parameter index", ylabel="normalized value (mean ± std)", ylim=(-0.05, 1.05),
       title="(b) Per-parameter spread (normalized)")

# (c) |correlation| of each parameter with loss
ax = fig.add_subplot(gs[1, 0])
Pc = P - P.mean(0); lc = sel_loss - sel_loss.mean()
denom = (P.std(0) * sel_loss.std() + 1e-12)
corr = (Pc * lc[:, None]).mean(0) / denom
ax.bar(xi, np.abs(corr), color=np.where(corr < 0, GOOD, BAD), alpha=0.8)
ax.set(xlabel="parameter index", ylabel="|corr(param, loss)|",
       title="(c) Which parameters drive loss? (red=+, green=−)")
top3 = np.argsort(-np.abs(corr))[:3]
ax.text(0.97, 0.9, "top: " + ", ".join(f"#{i}({corr[i]:+.2f})" for i in top3),
        transform=ax.transAxes, ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# (d) the most loss-correlated parameter vs loss
ax = fig.add_subplot(gs[1, 1])
j = top3[0]
ax.scatter(P[:, j], sel_loss, s=10, c=sel_loss, cmap="viridis", alpha=0.6)
ax.set(xlabel=f"parameter #{j} value", ylabel="loss",
       title=f"(d) Most loss-correlated parameter (#{j}, r={corr[j]:+.2f})")
fig.savefig(os.path.join(FIG, "05_parameter_structure.png"), bbox_inches="tight")
plt.close(fig)
print("wrote 05_parameter_structure.png")

# ----------------------------------------------------------------------------
# Text summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"designs:                 {N:,}")
print(f"unique topologies:       {len(uniq_topo):,}")
print(f"loss  min/med/mean/max:  {loss.min():.3f} / {np.median(loss):.3f} / {loss.mean():.3f} / {loss.max():.3f}")
print(f"designs with loss<0:     {(loss<0).sum():,} ({(loss<0).mean()*100:.1f}%)")
print(f"designs beating {R1_BEST}:   {(loss<R1_BEST).sum():,} ({(loss<R1_BEST).mean()*100:.1f}%)")
print(f"reoptimized children:    {has_parent.sum():,}  ({(delta<0).mean()*100:.0f}% improved, median Δ={np.median(delta):.3f})")
print(f"param count 3x3 / 4x4:   {int(np.median(complexity[size==3]))} / {int(np.median(complexity[size==4]))}")
print(f"global best:             loss={loss[order[0]]:.4f}  topo={topo[order[0]]}  size={size[order[0]]}")
f.close()
