"""Figures for the Milestone 2 report.

  python3 scripts/plots.py baseline:plain baseline:scalenorm improved:scalenorm
Each argument is <run>:<eval tag>; the first is treated as the reference.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = "results/figures"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def dots(c):
    v = c + 1
    return "".join(str(i + 1) for i in range(6) if v >> i & 1)


def read_log(run):
    with open(f"results/{run}/log.csv") as f:
        rows = list(csv.DictReader(f))
    return rows


def curves(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, run in enumerate(runs):
        rows = read_log(run)
        ep = [int(r["epoch"]) for r in rows]
        tot = [float(r["box"]) + float(r["cls"]) + float(r["dfl"]) for r in rows]
        ax[0].plot(ep, tot, color=SERIES[k], lw=2, label=run)
        ev = [(int(r["epoch"]), float(r["val_map50"])) for r in rows if r["val_map50"]]
        if ev:
            ax[1].plot(*zip(*ev), color=SERIES[k], lw=2, marker="o", ms=4, label=run)
    ax[0].set(title="Training loss (box + cls + dfl)", xlabel="epoch", yscale="log")
    ax[1].set(title="Validation mAP@0.5 (EMA weights)", xlabel="epoch")
    for a in ax:
        a.legend()
    fig.savefig(f"{OUT}/training_curves.png")


def per_class(specs, split="test"):
    ref_run, ref_tag = specs[0]
    gt_counts = np.zeros(63)
    idx = json.load(open("data/index.json"))
    for it in idx[split]:
        for l in it["labels"]:
            gt_counts[l] += 1
    present = np.where(gt_counts > 0)[0]
    order = present[np.argsort(-gt_counts[present])]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    w = 0.8 / len(specs)
    for k, (run, tag) in enumerate(specs):
        f1 = np.load(f"results/{run}/eval_{tag}/{split}_f1_per_class.npy")
        ax.bar(np.arange(len(order)) + (k - (len(specs) - 1) / 2) * w, np.nan_to_num(f1[order]), w * 0.9,
               color=SERIES[k], label=f"{run} ({tag})")
    ax.set_xticks(np.arange(len(order)), [f"{dots(c)}\n{int(gt_counts[c])}" for c in order], fontsize=5.5)
    ax.set(title=f"Per-class F1 on {split} (classes sorted by frequency; label = dots / count)", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    fig.savefig(f"{OUT}/per_class_f1_{split}.png")


def top_confusions(run, tag, split="test", k=12):
    cm = np.load(f"results/{run}/eval_{tag}/{split}_confusion.npy")
    nc = cm.shape[0] - 1
    pairs = []
    for i in range(nc + 1):
        for j in range(nc + 1):
            if i != j and cm[i, j] > 0:
                gi = "bg" if i == nc else dots(i)
                pj = "bg" if j == nc else dots(j)
                pairs.append((cm[i, j], f"{gi} → {pj}"))
    pairs.sort(reverse=True)
    pairs = pairs[:k][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(pairs)), [p[0] for p in pairs], color=SERIES[0], height=0.6)
    ax.set_yticks(range(len(pairs)), [p[1] for p in pairs])
    for y, (v, _) in enumerate(pairs):
        ax.text(v, y, f" {v}", va="center", color=INK2, fontsize=8)
    ax.set(title=f"Most frequent errors, {run} on {split}\n(true dots → predicted; bg = background)", xlabel="count")
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/confusions_{run}_{tag}_{split}.png")


def summary_table(specs):
    rows = []
    for run, tag in specs:
        m = json.load(open(f"results/{run}/eval_{tag}/metrics.json"))
        for split in ("val", "test"):
            s = m[split]
            rows.append([run, tag, split] + [round(s[k], 4) for k in
                        ("map50", "map50_95", "char_P", "char_R", "char_F1", "ctr_F1", "det_F1", "macro_F1")] +
                        [round(s["ms_per_image"])])
    hdr = ["run", "inference", "split", "mAP50", "mAP50-95", "charP", "charR", "charF1", "ctrF1", "detF1", "macroF1", "ms/img"]
    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)
    for r in [hdr] + rows:
        print(" | ".join(str(x) for x in r))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    specs = [tuple(s.split(":")) for s in sys.argv[1:]]
    curves(sorted({r for r, _ in specs}, key=[r for r, _ in specs].index))
    per_class(specs)
    for run, tag in specs:
        top_confusions(run, tag)
    summary_table(specs)
