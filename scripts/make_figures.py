#!/usr/bin/env python3
"""Draw the paper's three result figures from comparison-metrics.json.

Figures: exact disclosure by model and memory system, membership-inference AUC by
query budget, and preservation of unrelated memory after cleanup. Style follows
figures4papers (github.com/ChenLiu-1996/figures4papers).

Usage:
  python scripts/make_figures.py --metrics <comparison-dir>/comparison-metrics.json --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- house style ---------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 20
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 3
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["savefig.bbox"] = "tight"

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E",
    "highlight": "#FFD700",
}
BACKEND_COLOR = {"mem0": PALETTE["blue_main"], "amem": PALETTE["red_strong"]}
BACKEND_LABEL = {"mem0": "Mem0", "amem": "A-MEM"}
MODEL_ORDER = ["llama", "gpt-oss", "gemma"]
MODEL_LABEL = {"llama": "Llama 3.1\n8B Instruct", "gpt-oss": "GPT-OSS\n20B",
               "gemma": "Gemma 4\n31B IT"}
MODEL_SHORT = {"llama": "Llama", "gpt-oss": "GPT-OSS", "gemma": "Gemma"}
MODEL_COLOR = {"llama": PALETTE["blue_main"], "gpt-oss": PALETTE["teal"],
               "gemma": PALETTE["violet"]}


def style_ax(ax):
    ax.tick_params(width=3, length=7)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(3)


def load(metrics_path):
    rows = json.loads(Path(metrics_path).read_text())
    return rows


def pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return out[0] if out else None


def val(r, field):
    return (r.get(field, 0.0) or 0.0) * 100.0


def err(r, field):
    s = r.get(field + "_std")
    return (s or 0.0) * 100.0 if s is not None else None


# ---- Figure 1: exact disclosure, grouped bars by model x backend ---------
def fig_disclosure(rows, out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    for i, backend in enumerate(["mem0", "amem"]):
        heights, errs = [], []
        for m in MODEL_ORDER:
            r = pick(rows, model=m, backend=backend, kind="synthetic",
                     state="normally_deleted", method="api", wording="direct",
                     defense="none", budget=5)
            heights.append(val(r, "exact_disclosure_rate") if r else 0)
            errs.append(err(r, "exact_disclosure_rate") if r else None)
        yerr = errs if any(e for e in errs) else None
        bars = ax.bar(x + (i - 0.5) * w, heights, w, yerr=yerr, capsize=5,
                      color=BACKEND_COLOR[backend], label=BACKEND_LABEL[backend],
                      edgecolor="black", linewidth=1.5, error_kw={"elinewidth": 2.5})
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_ylabel("Exact disclosure (%)", fontsize=24)
    ax.set_ylim(0, 108)
    ax.set_title("Deleted value revealed within five questions\n(API deletion, no cleanup)",
                 fontsize=22)
    ax.legend(frameon=False, fontsize=18, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"exact-disclosure.{ext}", dpi=600)
    plt.close(fig)


# ---- Figure 3: unrelated preference kept after natural-language deletion + cascade ----
def fig_useful(rows, out):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    cols = [(m, b) for m in MODEL_ORDER for b in ("mem0", "amem")]
    x = np.arange(len(cols))
    w = 0.38
    specs = [("normally_deleted", PALETTE["green_3"], "Ordinary test"),
             ("partially_deleted", PALETTE["red_strong"], "Combined-summary test")]
    for i, (state, color, label) in enumerate(specs):
        heights = []
        for m, b in cols:
            r = pick(rows, model=m, backend=b, kind="synthetic", state=state,
                     method="natural_language", wording="broad",
                     defense="cascade", budget=5)
            heights.append(val(r, "retained_useful_memory") if r else 0)
        ax.bar(x + (i - 0.5) * w, heights, w, color=color, label=label,
               edgecolor="black", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_SHORT[m]}\n{BACKEND_LABEL[b]}" for m, b in cols], fontsize=15)
    ax.set_ylabel("Preference kept (%)", fontsize=24)
    ax.set_ylim(0, 108)
    ax.set_title("Unrelated preference kept after deletion + cascade defense",
                 fontsize=22, pad=44)
    # Bars span the full height on most columns, so place the legend outside the
    # axes (above the plot, below the title) where it cannot overlap any bar.
    ax.legend(frameon=False, fontsize=17, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=1.4, columnspacing=1.6)
    style_ax(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"useful-memory.{ext}", dpi=600)
    plt.close(fig)


# ---- Figure 2: AUC vs query budget, palette line plot (backend x defense) -
def fig_auc(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    budgets = [1, 2, 3, 5]
    for bi, backend in enumerate(["mem0", "amem"]):
        for di, defense in enumerate(["none", "cascade"]):
            ax = axes[bi, di]
            for m in MODEL_ORDER:
                xs, ys, lo, hi = [], [], [], []
                for bud in budgets:
                    r = pick(rows, model=m, backend=backend, kind="synthetic",
                             state="normally_deleted", method="api",
                             wording="direct", defense=defense, budget=bud)
                    if not r:
                        continue
                    xs.append(bud); ys.append(r["roc_auc"])
                    lo.append(r.get("roc_auc_ci95_low", r["roc_auc"]))
                    hi.append(r.get("roc_auc_ci95_high", r["roc_auc"]))
                ax.plot(xs, ys, marker="o", markersize=9, linewidth=3.5,
                        color=MODEL_COLOR[m], label=MODEL_LABEL[m].replace("\n", " "))
                ax.fill_between(xs, lo, hi, color=MODEL_COLOR[m], alpha=0.15)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=2)
            ax.set_title(f"{BACKEND_LABEL[backend]} / "
                         f"{'without cascade defense' if defense == 'none' else 'with cascade defense'}", fontsize=20)
            ax.set_xlabel("Questions per fact", fontsize=20)
            ax.set_ylabel("ROC-AUC", fontsize=20)
            ax.set_ylim(-0.02, 1.04)
            ax.set_xticks(budgets)
            style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=18)
    fig.suptitle("Membership-inference AUC after API deletion (0.5 = indistinguishable)\n"
                 "shading: 95% paired-fact bootstrap intervals", fontsize=22)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ordinary-api-auc.{ext}", dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default="images")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.metrics)
    fig_disclosure(rows, out)
    fig_useful(rows, out)
    fig_auc(rows, out)
    print(f"Wrote exact-disclosure, useful-memory, ordinary-api-auc (png+pdf) to {out}/")


if __name__ == "__main__":
    main()
