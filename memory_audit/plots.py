"""Export Matplotlib figures from already-computed results."""
import json
from pathlib import Path


def plot_report(output, kind="synthetic", method="api", wording="direct"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(output)
    report = json.loads((root / "summary.json").read_text())
    rows = [r for r in report["conditions"] if r["kind"] == kind and r["method"] == method
            and r["wording"] == wording and r["metrics"]["status"] == "ok"]
    if not rows:
        raise ValueError("No completed result rows match the plotting filters")
    backends = sorted({r["backend"] for r in rows})
    states = ["normally_deleted", "partially_deleted"]
    fig, axes = plt.subplots(2, len(backends), squeeze=False, figsize=(6 * len(backends), 9))
    for si, state in enumerate(states):
        for bi, backend in enumerate(backends):
            ax = axes[si][bi]
            group = [r for r in rows if r["state"] == state and r["backend"] == backend]
            for defense in sorted({r["defense"] for r in group}):
                series = sorted([r for r in group if r["defense"] == defense], key=lambda r: r["budget"])
                x = [r["budget"] for r in series]
                y = [r["metrics"]["roc_auc"] for r in series]
                intervals = [r["metrics"]["roc_auc_ci95"] for r in series]
                ax.plot(x, y, marker="o", label=defense)
                ax.fill_between(x, [v[0] for v in intervals], [v[1] for v in intervals], alpha=0.15)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set(title=f"{backend}: {state.replace('_', ' ')}", xlabel="Query budget", ylabel="Prior-adjusted ROC AUC",
                   ylim=(-0.02, 1.02))
            ax.legend(title="Defense")
            ax.grid(alpha=0.2)
    fixture = any(r["fixture"] for r in rows)
    fig.suptitle(("FIXTURE VALIDATION: NOT LLM RESULTS\n" if fixture else "") +
                 f"{kind} | {method} | {wording}\nShading: 95% paired-fact bootstrap intervals")
    fig.tight_layout()
    destination = root / f"auc-{kind}-{method}-{wording}"
    for extension in ("png", "pdf"):
        fig.savefig(str(destination) + "." + extension, dpi=180)
    plt.close(fig)
    return str(destination)
