"""Tie-aware metrics and paired fact-cluster bootstrap, standard library only."""
import math
import random


def roc(labels, scores):
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        raise ValueError("ROC needs equally sized finite scores and both binary classes")
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("Non-finite score")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    groups = {}
    for label, score in zip(labels, scores):
        groups.setdefault(score, []).append(label)
    points = [{"threshold": None, "fpr": 0.0, "tpr": 0.0}]
    tp = fp = 0
    for threshold, group in sorted(groups.items(), reverse=True):
        tp += sum(group)
        fp += len(group) - sum(group)
        points.append({"threshold": threshold, "fpr": fp / negatives, "tpr": tp / positives})
    return points


def auc(labels, scores):
    points = roc(labels, scores)
    return sum((b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2
               for a, b in zip(points, points[1:]))


def empirical_tpr(labels, scores, fpr=0.01):
    # Do not interpolate across a tied group: a deployable threshold must
    # accept or reject the entire group, especially at low FPR.
    return max(p["tpr"] for p in roc(labels, scores) if p["fpr"] <= fpr)


def threshold_at_fpr(labels, scores, fpr=0.01):
    eligible = [p for p in roc(labels, scores) if p["fpr"] <= fpr]
    best = max(eligible, key=lambda p: (p["tpr"], -p["fpr"], p["threshold"] or 0))
    # None denotes reject-all, JSON-compliant (no Infinity).
    return best["threshold"]


def classification(labels, scores, threshold):
    predictions = [threshold is not None and s >= threshold for s in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p for y, p in zip(labels, predictions))
    fn = sum(y == 1 and not p for y, p in zip(labels, predictions))
    tn = sum(y == 0 and not p for y, p in zip(labels, predictions))
    return {"accuracy": (tp + tn) / len(labels), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "fpr": fp / (fp + tn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def paired_metrics(pairs, calibration, bootstrap=1000, seed=0, fpr=0.01):
    """Each pair is (positive, negative); one independent candidate per pair."""
    if not pairs or not calibration:
        return {"status": "insufficient_data", "test_pairs": len(pairs), "calibration_pairs": len(calibration)}
    labels = [y for _ in pairs for y in (1, 0)]
    scores = [x for pair in pairs for x in pair]
    cal_labels = [y for _ in calibration for y in (1, 0)]
    cal_scores = [x for pair in calibration for x in pair]
    threshold = threshold_at_fpr(cal_labels, cal_scores, fpr)
    result = {"status": "ok", "test_pairs": len(pairs), "calibration_pairs": len(calibration),
              "roc_auc": auc(labels, scores), "tpr_at_1pct_fpr": empirical_tpr(labels, scores, fpr),
              "roc": roc(labels, scores), "threshold_from_calibration": threshold,
              "calibrated": classification(labels, scores, threshold),
              "fpr_resolution": 1 / len(pairs),
              "low_fpr_underpowered": len(pairs) < 100 or len(calibration) < 100,
              "ci_method": "95% percentile paired-fact bootstrap; fixed calibrated threshold"}
    rng, auc_samples, tpr_samples = random.Random(seed), [], []
    for _ in range(bootstrap):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        sample_scores = [x for pair in sample for x in pair]
        auc_samples.append(auc(labels, sample_scores))
        tpr_samples.append(empirical_tpr(labels, sample_scores, fpr))
    result["roc_auc_ci95"] = [percentile(auc_samples, q) for q in (0.025, 0.975)]
    result["tpr_at_1pct_fpr_ci95"] = [percentile(tpr_samples, q) for q in (0.025, 0.975)]
    return result


def auc_improvement(pairs_one, pairs_multi, bootstrap=1000, seed=0):
    if not pairs_one or len(pairs_one) != len(pairs_multi):
        return None
    labels = [y for _ in pairs_one for y in (1, 0)]
    def delta(indices):
        return (auc(labels, [s for i in indices for s in pairs_multi[i]]) -
                auc(labels, [s for i in indices for s in pairs_one[i]]))
    rng = random.Random(seed)
    samples = [delta([rng.randrange(len(pairs_one)) for _ in pairs_one]) for _ in range(bootstrap)]
    return {"delta_auc": delta(range(len(pairs_one))),
            "ci95": [percentile(samples, q) for q in (0.025, 0.975)]}
