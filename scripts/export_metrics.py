"""Flatten a model comparison into one metrics table.

Reads the per-model reports written by `python -m memory_audit compare` and writes,
into the comparison directory:

  comparison-metrics.json / .csv   one row per model x condition x query budget:
                                   exact disclosure, retention, AUC with 95% CI, ...
  control-disclosure.json          exact disclosure in the control states

Usage:
  python scripts/export_metrics.py --comparison <comparison-dir>
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

CONDITION_KEYS = ("backend", "kind", "state", "method", "wording", "defense")


def flatten(model, rows, report):
    """One flat record per reported condition, plus exact-disclosure counts from transcripts."""
    groups = defaultdict(list)
    for row in rows:
        if row["condition"]:
            condition = row["condition"]
            key = (row["backend"], row["kind"], row["state"],
                   condition["method"], condition["wording"], condition["defense"])
            groups[key].append(row)
    records = []
    for condition in report["conditions"]:
        key = tuple(condition[name] for name in CONDITION_KEYS)
        test = [row for row in groups[key] if row["split"] == "test"]
        budget = condition["budget"]
        # Disclosed = the exact value appears in any reply within the first `budget` questions.
        disclosed = sum(any(turn["features"]["disclosure"] for turn in row["transcript"][:budget]) for row in test)
        metrics = condition["metrics"]
        interval = metrics.get("roc_auc_ci95", [None, None])
        gain = condition.get("multi_minus_single") or {}
        records.append(dict(
            model=model, **dict(zip(CONDITION_KEYS, key)), budget=budget, test_n=len(test),
            exact_disclosed=disclosed, exact_disclosure_rate=disclosed / len(test) if test else None,
            active_artifact_survival=condition["active_artifact_survival"],
            confirmed_active_residual_rate=condition["confirmed_active_residual_rate"],
            retained_useful_memory=condition["retained_useful_memory"],
            utility_recall=condition["utility_recall"],
            correct_target_selection=condition["correct_target_selection"],
            roc_auc=metrics.get("roc_auc"), roc_auc_ci95_low=interval[0], roc_auc_ci95_high=interval[1],
            test_roc_tpr_at_1pct_fpr=metrics.get("tpr_at_1pct_fpr"),
            calibrated_test_fpr=metrics.get("calibrated", {}).get("fpr"),
            calibrated_test_recall=metrics.get("calibrated", {}).get("recall"),
            delta_auc_vs_one_query=gain.get("delta_auc"),
            delta_auc_ci95_low=gain.get("ci95", [None, None])[0],
            delta_auc_ci95_high=gain.get("ci95", [None, None])[1]))
    return records


def control_disclosure(model, rows):
    """Exact disclosure (any reply) in the control states, per backend and fact kind."""
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        if not row["condition"] and row["split"] == "test":
            counts = groups[(model, row["backend"], row["kind"], row["state"])]
            counts[0] += any(turn["features"]["disclosure"] for turn in row["transcript"])
            counts[1] += 1
    return [dict(zip(("model", "backend", "kind", "state"), key), exact_disclosed=hits, test_n=total)
            for key, (hits, total) in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--comparison", required=True, help="directory written by `memory_audit compare`")
    root = Path(parser.parse_args().comparison)
    models = list(json.loads((root / "comparison-manifest.json").read_text())["runs"])
    records, controls = [], []
    for model in models:
        rows = json.loads((root / model / "observations.json").read_text())
        report = json.loads((root / model / "summary.json").read_text())
        records.extend(flatten(model, rows, report))
        controls.extend(control_disclosure(model, rows))
    (root / "comparison-metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    with (root / "comparison-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (root / "control-disclosure.json").write_text(json.dumps(controls, indent=2) + "\n")
    print(f"Wrote {len(records)} condition rows for {', '.join(models)} to {root}")


if __name__ == "__main__":
    main()
