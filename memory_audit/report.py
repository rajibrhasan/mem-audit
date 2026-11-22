"""Turn per-state observations into summary metrics (summary.json and summary.csv)."""
import csv
from pathlib import Path

from .files import dump
from .metrics import auc_improvement, paired_metrics


def summarize(rows, config):
    baselines = {(r["backend"], r["seed"], r["fact_id"], r["state"]): r
                 for r in rows if not r["condition"]}
    groups = {}
    for row in rows:
        if row["state"] not in ("normally_deleted", "partially_deleted"):
            continue
        cond = row["condition"]
        key = (row["backend"], row["kind"], row["state"], cond["method"], cond["wording"], cond["defense"])
        groups.setdefault(key, []).append(row)
    results = []
    for key, observations in sorted(groups.items()):
        backend, kind, state, method, wording, defense = key
        negative_state = "never" if state == "normally_deleted" else "verified_clean"
        test = sorted([r for r in observations if r["split"] == "test"], key=lambda r: (r["seed"], r["fact_id"]))
        calibration = [r for r in observations if r["split"] == "calibration"]
        for budget in config["budgets"]:
            def pairs(data, index):
                return [(r["deltas"][index], baselines[(backend, r["seed"], r["fact_id"], negative_state)]["deltas"][index])
                        for r in data]
            statistics = paired_metrics(pairs(test, budget - 1), pairs(calibration, budget - 1),
                                        config["bootstrap"], config["seeds"][0])
            def rate(field):
                return sum(bool(r[field]) for r in test) / len(test) if test else None
            result = {"backend": backend, "kind": kind, "state": state, "method": method, "wording": wording,
                      "defense": defense, "budget": budget, "negative_state": negative_state,
                      "task": "post_deletion_vs_never" if state == "normally_deleted" else "partial_challenge_vs_verified_clean",
                      "fixture": backend == "fixture", "metrics": statistics,
                      "active_artifact_survival": rate("active_survival"), "history_artifact_survival": rate("history_survival"),
                      "retained_useful_memory": rate("useful_retained"), "utility_recall": rate("utility_recall"),
                      "over_deletion": rate("over_deletion"),
                      "correct_target_selection": sum(r["selection"]["correct_target_selection"] for r in test) / len(test) if test else None,
                      "confirmed_active_residual_rate": sum(r["confirmed_residual"][budget - 1] for r in test) / len(test) if test else None}
            if test and defense == "none":
                correctly_selected = [r for r in test if r["selection"]["correct_target_selection"]]
                result["removal_failure_given_correct_selection"] = (
                    sum(not r["selection"]["main_records_removed"] for r in correctly_selected) / len(correctly_selected)
                    if correctly_selected else None)
            if budget > 1:
                result["multi_minus_single"] = auc_improvement(pairs(test, 0), pairs(test, budget - 1),
                                                               config["bootstrap"], config["seeds"][0])
            if state == "partially_deleted":
                # Primary challenge includes failures/successes of both paths.
                # Also report the strictly verified partial task.
                verified_test = [r for r in test if r["active_survival"] and r["selection"]["main_records_removed"]]
                verified_cal = [r for r in calibration if r["active_survival"] and r["selection"]["main_records_removed"]]
                result["verified_partial_vs_clean"] = paired_metrics(
                    pairs(verified_test, budget - 1), pairs(verified_cal, budget - 1), config["bootstrap"], config["seeds"][0])
            results.append(result)
    # Positive control distinguishes still-stored from never, adjusted for prior.
    controls = []
    for backend in sorted({r["backend"] for r in rows}):
        for kind in sorted({r["kind"] for r in rows}):
            for budget in config["budgets"]:
                pairs_by_split = {"test": [], "calibration": []}
                for r in rows:
                    if r["backend"] == backend and r["kind"] == kind and r["state"] == "stored":
                        never = baselines[(backend, r["seed"], r["fact_id"], "never")]
                        pairs_by_split[r["split"]].append((r["deltas"][budget - 1], never["deltas"][budget - 1]))
                controls.append({"backend": backend, "kind": kind, "budget": budget,
                                 "metrics": paired_metrics(pairs_by_split["test"], pairs_by_split["calibration"],
                                                           config["bootstrap"], config["seeds"][0])})
    return {"conditions": results, "stored_vs_never_controls": controls,
            "warnings": ["Fixture results are software checks, not LLM evidence.",
                         "1% FPR estimates with <100 calibration or test negatives have inadequate empirical resolution.",
                         "Public controls reuse a small fixed fact set; their intervals are descriptive, not population estimates.",
                         "Artifact inspection covers live logical stores and histories, not forensic disk erasure.",
                         "Semantic scoring is a heuristic; inspect transcripts and artifact evidence before interpreting leakage."]}


def write_report(output, rows, config):
    output = Path(output)
    report = summarize(rows, config)
    dump(output / "summary.json", report)
    fields = ["backend", "kind", "state", "method", "wording", "defense", "budget", "roc_auc",
              "roc_auc_ci95", "tpr_at_1pct_fpr", "tpr_at_1pct_fpr_ci95", "accuracy", "precision", "recall", "fpr",
              "utility_recall", "active_artifact_survival", "history_artifact_survival",
              "correct_target_selection", "retained_useful_memory", "over_deletion", "fixture"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition in report["conditions"]:
            flat = dict(condition, **condition["metrics"])
            flat.update(condition["metrics"].get("calibrated", {}))
            writer.writerow({k: flat.get(k) for k in fields})
    return report
