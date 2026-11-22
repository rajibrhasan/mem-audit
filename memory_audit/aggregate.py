"""Combine finished shards into reports.

* ``collect``          merge the shards of one model into a single run with a report
* ``compare``          compare models on the facts valid for every model and memory system
* ``collect_ablation`` compare reasoning modes on the cases valid in every mode

A sharded run lives in ``<run>/shards/<name>/``; each shard covers one memory
system and one seed (see runner.run).
"""
import json
from pathlib import Path

from .ablation import variants
from .config import expected_rows
from .data import population
from .files import dump
from .plots import plot_report
from .report import write_report
from .runner import read_case

EXCLUSION_NOTE = ("Deletion metrics are conditional on valid controls and complete responses. "
                  "Report every excluded case and population denominator separately; "
                  "never score these outcomes as successful deletion.")
COHORT_NOTE = ("Eligibility can depend on the model and attribute. These conditional metrics must be "
               "read with full-population failure counts; omitted/corrupted controls are not deletion successes.")


def find_shards(root, config):
    """Map each (backend, seed) requested by `config` to its shard directory."""
    requested = {(backend, seed) for backend in config["backends"] for seed in config["seeds"]}
    found = {}
    for path in sorted((Path(root) / "shards").glob("*/manifest.json")):
        shard_config = json.loads(path.read_text())["config"]
        if len(shard_config["backends"]) != 1 or len(shard_config["seeds"]) != 1:
            raise ValueError(f"Shard {path.parent.name} must hold one backend and one seed")
        identity = (shard_config["backends"][0], shard_config["seeds"][0])
        if identity not in requested:
            continue
        if identity in found:
            raise ValueError(f"Duplicate shard for backend/seed {identity}")
        if shard_config != dict(config, backends=[identity[0]], seeds=[identity[1]]):
            raise ValueError(f"Shard {path.parent.name} was run with a different configuration")
        found[identity] = path.parent
    missing = requested - found.keys()
    if missing:
        raise ValueError(f"Missing shards for backend/seed: {sorted(missing)}")
    return found


def load_run(root, config):
    """Read and validate every case of a sharded run. Every case must be finished."""
    rows, outcomes = [], []
    per_case = expected_rows(config)
    shards = find_shards(root, config)
    for backend in config["backends"]:
        for seed in config["seeds"]:
            shard = shards[(backend, seed)]
            for fact in population(config["facts"], seed, config["public_controls"]):
                result = read_case(shard / "cases" / backend / str(seed) / fact.id, backend, seed, fact, per_case)
                if result is None:
                    raise ValueError(f"Shard {shard.name} has an unfinished case: {backend}/{seed}/{fact.id}")
                rows.extend(result[0])
                outcomes.append(result[1])
    return rows, outcomes, shards


def eligibility(outcomes):
    """Population counts by backend, fact kind, split and outcome."""
    counts = {}
    for outcome in outcomes:
        key = (outcome["backend"], outcome["kind"], outcome["split"], outcome["status"], outcome.get("reason"))
        counts[key] = counts.get(key, 0) + 1
    return {
        "population_cases": len(outcomes),
        "complete_cases": sum(o["status"] == "complete" for o in outcomes),
        "excluded_cases": sum(o["status"] != "complete" for o in outcomes),
        "interpretation": EXCLUSION_NOTE,
        "counts": [dict(zip(("backend", "kind", "split", "status", "reason"), key), count=count)
                   for key, count in counts.items()],
    }


def _report(destination, rows, config, eligibility_info, note, **extra):
    """Write summary.json/.csv and the per-run AUC plot for one set of observations."""
    report = write_report(destination, rows, config)
    report["eligibility"] = eligibility_info
    report.update(extra)
    report["warnings"].append(note)
    dump(Path(destination) / "summary.json", report)
    plot_report(destination)
    return report


def collect(config, output):
    """Merge one model's shards into `output` and write its report."""
    root = Path(output).resolve()
    rows, outcomes, shards = load_run(root, config)
    # Link each shard's cases into the combined run only after every case validated.
    for (backend, seed), shard in shards.items():
        link, target = root / "cases" / backend / str(seed), shard / "cases" / backend / str(seed)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError("Combined case link points to a different shard")
        elif link.exists():
            raise ValueError("Combined case destination already exists")
        else:
            link.symlink_to(target, target_is_directory=True)
    for seed in config["seeds"]:
        dump(root / f"population-{seed}.json",
             [fact.to_dict() for fact in population(config["facts"], seed, config["public_controls"])])
    dump(root / "manifest.json", {"config": config,
                                  "shards": {f"{backend}/{seed}": shard.name for (backend, seed), shard in shards.items()}})
    dump(root / "observations.json", rows)
    dump(root / "case-outcomes.json", outcomes)
    info = eligibility(outcomes)
    dump(root / "eligibility.json", info)
    _report(root, rows, config, info, EXCLUSION_NOTE)
    print(f"Collected {len(rows)} observations from {len(outcomes)} cases: {root}", flush=True)
    return rows


def common_subjects(outcomes_by_model, backends):
    """(seed, fact) pairs completed by every model under every backend."""
    cohorts = [{(o["seed"], o["id"]) for o in outcomes if o["backend"] == backend and o["status"] == "complete"}
               for outcomes in outcomes_by_model.values() for backend in backends]
    return set.intersection(*cohorts) if cohorts else set()


def compare(runs, output):
    """Compare collected model runs ({name: run dir}) on their common eligible facts."""
    root = Path(output).resolve()
    loaded, reference = {}, None
    for name, path in runs.items():
        source = Path(path).resolve()
        config = json.loads((source / "manifest.json").read_text())["config"]
        protocol = {key: value for key, value in config.items() if key != "llm"}
        protocol["generation"] = {key: config["llm"].get(key)
                                  for key in ("temperature", "max_tokens", "memory_max_tokens", "seed")}
        if reference is not None and protocol != reference:
            raise ValueError("Model runs differ in population, protocol or generation budgets")
        reference = protocol
        rows, outcomes, _ = load_run(source, config)
        loaded[name] = (source, config, rows, outcomes)
    common = common_subjects({name: item[3] for name, item in loaded.items()}, reference["backends"])
    cohort = {
        "scope": "same eligible subjects across every backbone and memory backend",
        "runs": {name: str(item[0]) for name, item in loaded.items()},
        "common_subjects": [{"seed": seed, "fact_id": fact} for seed, fact in sorted(common)],
        "common_subject_count": len(common),
        "eligibility_by_model": {name: eligibility(item[3]) for name, item in loaded.items()},
        "warning": COHORT_NOTE,
    }
    dump(root / "comparison-manifest.json", cohort)
    conditions = []
    for name, (source, config, rows, outcomes) in loaded.items():
        matched = [row for row in rows if (row["seed"], row["fact_id"]) in common]
        destination = root / name
        dump(destination / "observations.json", matched)
        dump(destination / "manifest.json", {"config": config, "comparison_cohort": cohort})
        report = _report(destination, matched, config, cohort["eligibility_by_model"][name], COHORT_NOTE,
                         common_subject_count=len(common))
        conditions.extend(dict(condition, backbone=name) for condition in report["conditions"])
    dump(root / "comparison.json", {"cohort": cohort, "conditions": conditions})
    print(f"Compared {len(runs)} models on {len(common)} common facts: {root}", flush=True)


def collect_ablation(config, baseline, output):
    """Compare reasoning modes on the cases completed in every mode.

    `baseline` is the model's sharded main run; `output` holds the ablation
    shards written by runner.ablate under the same shard names.
    """
    root, baseline = Path(output).resolve(), Path(baseline).resolve()
    modes = variants(config)
    default = next(iter(modes))
    per_case = expected_rows(config)
    matched = {mode: [] for mode in modes}
    outcomes = []
    order = lambda item: (config["backends"].index(item[0][0]), config["seeds"].index(item[0][1]))
    for (backend, seed), shard in sorted(find_shards(baseline, config).items(), key=order):
        replayed_root = root / "shards" / shard.name
        for fact in population(config["facts"], seed, config["public_controls"]):
            result = read_case(shard / "cases" / backend / str(seed) / fact.id, backend, seed, fact, per_case)
            if result is None:
                raise ValueError(f"Baseline run is unfinished: {backend}/{seed}/{fact.id}")
            status, case_rows = {default: result[1]}, {}
            if result[1]["status"] == "complete":
                case_rows[default] = result[0]
                for mode in modes:
                    if mode == default:
                        continue
                    replay = replayed_root / mode / "cases" / backend / str(seed) / fact.id
                    replayed = read_case(replay, backend, seed, fact, per_case)
                    if replayed is None:
                        raise ValueError(f"Ablation is unfinished: {replay}")
                    case_rows[mode], status[mode] = replayed
            common = len(status) == len(modes) and all(item["status"] == "complete" for item in status.values())
            outcomes.append({"backend": backend, "seed": seed, "fact": fact.to_dict(), "modes": status,
                             "common_eligible": common})
            if common:
                for mode in modes:
                    matched[mode].extend(case_rows[mode])
    root.mkdir(parents=True, exist_ok=True)
    dump(root / "case-outcomes.json", outcomes)
    counts = {}
    for case in outcomes:
        for mode in modes:
            outcome = case["modes"].get(mode, {"status": "baseline_ineligible"})
            key = (mode, case["backend"], case["fact"]["kind"], case["fact"]["split"],
                   outcome["status"], outcome.get("reason"))
            counts[key] = counts.get(key, 0) + 1
    info = {"population_cases": len(outcomes),
            "common_eligible_cases": sum(case["common_eligible"] for case in outcomes),
            "interpretation": "Effort comparisons use the same cases successfully completed in every mode. "
                              "All baseline and variant failures are reported separately.",
            "counts": [dict(zip(("mode", "backend", "kind", "split", "status", "reason"), key), count=count)
                       for key, count in counts.items()]}
    dump(root / "eligibility.json", info)
    conditions = []
    for mode, rows in matched.items():
        destination = root / mode
        dump(destination / "manifest.json", {"config": modes[mode]})
        dump(destination / "observations.json", rows)
        report = _report(destination, rows, modes[mode], info, info["interpretation"])
        conditions.extend(dict(condition, reasoning_mode=mode) for condition in report["conditions"])
    dump(root / "comparison.json", {"scope": "answering_only", "eligibility": info, "conditions": conditions})
    print(f"Ablation compared on {info['common_eligible_cases']} common cases: {root}", flush=True)
    return matched
