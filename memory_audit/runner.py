"""Run audit cases in worker processes, with resume and failure accounting.

A run covers every (backend, seed, fact) case of a config. Each case runs in its
own process and writes its evidence to

    <output>/cases/<backend>/<seed>/<fact>/attempt-<id>/

When the process ends, the parent writes one terminal marker beside the attempt:
``complete.json`` if the case produced ``result.json``, or ``excluded.json`` if it
stopped on a known failure recorded in ``failure.json``. Known failures are
invalid controls (the memory system never retained the planted fact, so deletion
cannot be tested) and invalid model output (truncated or empty replies). They are
excluded from the deletion metrics and counted separately; any other error stops
the run. Rerunning the same command resumes from the markers.
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import fcntl
from functools import partial
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import traceback
from typing import Callable, Optional
from urllib.parse import urlsplit
import uuid

from .ablation import replay_case, variants
from .config import expected_rows
from .data import Fact, population
from .doctor import check
from .experiment import run_fact
from .files import dump, sha256

CONTROL_FAILURES = {
    "Ingestion did not retain the unrelated utility control": "utility_not_retained",
    "Ingestion did not retain the target value; no valid stored-state control": "target_not_retained",
    "Target disappeared during ingestion/evolution": "target_lost_during_evolution",
    "Could not construct a distinct derived-memory positive control": "derived_control_missing",
    "Clean control failed artifact verification": "clean_control_invalid",
}
OUTPUT_FAILURES = {
    "Model output truncated; increase max_tokens": "output_truncated",
    "Endpoint returned empty or non-text content": "empty_output",
    "Endpoint response failed required JSON schema validation": "invalid_schema",
}
# Every request gets one long attempt and no automatic retry, so a slow or
# failed reply is classified rather than silently re-sampled.
TRANSPORT = {"timeout": 900, "retries": 0}


def classify(error):
    """Return (status, reason) for a known failure, or None for anything else."""
    message = str(error)
    if type(error) is RuntimeError and message in CONTROL_FAILURES:
        return "invalid_control", CONTROL_FAILURES[message]
    if type(error) in (RuntimeError, ValueError) and message in OUTPUT_FAILURES:
        return "invalid_model_output", OUTPUT_FAILURES[message]
    return None


def effective_config(config):
    """The config actually used: transport settings plus the job's server endpoint."""
    effective = copy.deepcopy(config)
    effective["llm"].update(TRANSPORT)
    endpoint = os.environ.get("MEMAUDIT_BASE_URL")
    if endpoint:
        url = urlsplit(endpoint)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.port is None or url.path != "/v1"
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("MEMAUDIT_BASE_URL must look like http://127.0.0.1:<port>/v1")
        effective["llm"]["base_url"] = endpoint
    return effective


# ---- worker processes ------------------------------------------------------

def _guarded(task, allowed, destination, backend, seed, fact_data):
    """Run a case; record known failures of the allowed kinds, re-raise the rest."""
    try:
        task()
    except Exception as error:
        category = classify(error)
        if category is None or category[0] not in allowed:
            raise
        dump(Path(destination) / "failure.json", {
            "status": category[0], "reason": category[1], "message": str(error),
            "exception_type": type(error).__name__, "traceback": traceback.format_exc(),
            "backend": backend, "seed": seed, "fact": fact_data,
        })


def run_case(config, backend, fact_data, seed, destination):
    """Worker: one experiment case (plant, delete, audit)."""
    effective = effective_config(config)
    dump(Path(destination) / "execution.json", {"effective_llm": effective["llm"]})
    _guarded(lambda: run_fact(effective, backend, Fact(**fact_data), seed, destination),
             ("invalid_control", "invalid_model_output"), destination, backend, seed, fact_data)


def replay_ablation_case(config, backend, fact_data, seed, source, destination):
    """Worker: re-answer one finished case under another reasoning mode."""
    effective = effective_config(config)
    dump(Path(destination) / "execution.json", {"effective_llm": effective["llm"], "baseline_case": str(source)})
    _guarded(lambda: replay_case(effective, backend, fact_data, seed, source, destination),
             ("invalid_model_output",), destination, backend, seed, fact_data)


# ---- case markers ------------------------------------------------------------

def _attempt(case, attempt):
    if not isinstance(attempt, str) or not attempt.startswith("attempt-") or Path(attempt).name != attempt:
        raise ValueError("Invalid attempt path")
    return case / attempt


def read_case(case, backend, seed, fact, rows_per_case):
    """Return (rows, outcome) for a case with a terminal marker, or None if unfinished."""
    complete, excluded = case / "complete.json", case / "excluded.json"
    if complete.exists() and excluded.exists():
        raise ValueError(f"Conflicting case outcomes: {case}")
    identity = {"backend": backend, "seed": seed, **fact.to_dict()}
    if complete.exists():
        marker = json.loads(complete.read_text())
        path = _attempt(case, marker["attempt"]) / "result.json"
        if sha256(path) != marker["sha256"]:
            raise ValueError(f"Completed result checksum mismatch: {case}")
        rows = json.loads(path.read_text())["rows"]
        if len(rows) != rows_per_case or any((r["backend"], r["seed"], r["fact_id"]) != (backend, seed, fact.id)
                                             for r in rows):
            raise ValueError(f"Incomplete or incorrect case rows: {case}")
        return rows, dict(identity, status="complete", result_sha256=marker["sha256"])
    if excluded.exists():
        marker = json.loads(excluded.read_text())
        path = _attempt(case, marker["attempt"]) / "failure.json"
        if sha256(path) != marker["sha256"]:
            raise ValueError(f"Failure record checksum mismatch: {case}")
        failure = json.loads(path.read_text())
        error_type = {"RuntimeError": RuntimeError, "ValueError": ValueError}.get(failure["exception_type"])
        category = classify(error_type(failure["message"])) if error_type else None
        if category is None or category != (failure["status"], failure["reason"]):
            raise ValueError(f"Unrecognized exclusion: {case}")
        if (failure["backend"], failure["seed"], failure["fact"]) != (backend, seed, fact.to_dict()):
            raise ValueError(f"Failure identity mismatch: {case}")
        return [], dict(identity, status=failure["status"], reason=failure["reason"],
                        failure_sha256=marker["sha256"])
    return None


# ---- scheduling ----------------------------------------------------------------

@dataclass
class _Job:
    label: str
    case: Path
    target: Optional[Callable]  # worker function; None for outcomes known in advance
    args: tuple
    read: Callable              # returns (rows, outcome) or None


@contextmanager
def _locked(root):
    """Refuse to let two runners write to the same output directory."""
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another runner is using {root}") from None
        yield


def _claim(path, content):
    """Record what an output directory holds; refuse to mix in different work."""
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise ValueError(f"{path.parent} was started with different settings; use a new output directory")
    else:
        dump(path, content)


def _execute(jobs, workers, outcomes_path):
    """Run unfinished jobs with at most `workers` processes; return all outcomes in job order."""
    if not 1 <= workers <= 64:
        raise ValueError("workers must be between 1 and 64")
    outcomes, pending = {}, []
    for index, job in enumerate(jobs):
        result = job.read()
        if result is None:
            pending.append(index)
        else:
            outcomes[index] = result[1]

    def save():
        dump(outcomes_path, [outcomes[i] for i in sorted(outcomes)])

    save()
    print(f"Reusing {len(outcomes)}/{len(jobs)} finished cases; running {len(pending)} "
          f"with {workers} worker(s)", flush=True)
    context = multiprocessing.get_context("spawn")
    queue, active, errors = iter(pending), {}, []
    try:
        while True:
            while not errors and len(active) < workers:
                index = next(queue, None)
                if index is None:
                    break
                job = jobs[index]
                attempt = "attempt-" + uuid.uuid4().hex[:12]
                process = context.Process(target=job.target, args=(*job.args, str(job.case / attempt)))
                process.start()
                active[process.sentinel] = (process, index, attempt)
            if not active:
                break
            for sentinel in wait(list(active)):
                process, index, attempt = active.pop(sentinel)
                process.join()
                job = jobs[index]
                directory = job.case / attempt
                if process.exitcode != 0:
                    errors.append(f"Unexpected failure in {job.label}; evidence kept in {directory}")
                    continue
                failure = directory / "failure.json"
                if failure.exists():
                    dump(job.case / "excluded.json", {"attempt": attempt, "sha256": sha256(failure)})
                else:
                    dump(job.case / "complete.json", {"attempt": attempt, "sha256": sha256(directory / "result.json")})
                outcomes[index] = job.read()[1]
                save()
                print(f"{job.label}: {outcomes[index]['status']} ({len(outcomes)}/{len(jobs)})", flush=True)
    finally:
        # Never leave orphaned workers writing after the parent stops.
        for process, _, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
    if errors:
        raise RuntimeError("\n".join(errors))
    return [outcomes[i] for i in sorted(outcomes)]


# ---- entry points ------------------------------------------------------------

def run(config, output, workers=1):
    """Run (or resume) every case of `config` into `output`."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root):
        check(effective_config(config))
        _claim(root / "manifest.json", {"config": config})
        rows = expected_rows(config)
        jobs = []
        for seed in config["seeds"]:
            facts = population(config["facts"], seed, config["public_controls"])
            dump(root / f"population-{seed}.json", [fact.to_dict() for fact in facts])
            for backend in config["backends"]:
                for fact in facts:
                    case = root / "cases" / backend / str(seed) / fact.id
                    jobs.append(_Job(f"{backend} seed={seed} {fact.id}", case, run_case,
                                     (config, backend, fact.to_dict(), seed),
                                     partial(read_case, case, backend, seed, fact, rows)))
        outcomes = _execute(jobs, workers, root / "case-outcomes.json")
    print(f"All {len(outcomes)} cases accounted for: {root}", flush=True)


def ablate(baseline, output, workers=1):
    """Re-answer every finished case of one run under each other reasoning mode.

    Memory snapshots, deletion selections and the population stay fixed at the
    baseline; only the answering step changes (see ablation.replay_case).
    """
    source = Path(baseline).resolve()
    config = json.loads((source / "manifest.json").read_text())["config"]
    rows = expected_rows(config)
    modes = variants(config)
    default = next(iter(modes))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root):
        _claim(root / "ablation-manifest.json", {"baseline": str(source), "modes": list(modes)})
        cases = []
        for backend in config["backends"]:
            for seed in config["seeds"]:
                for fact in population(config["facts"], seed, config["public_controls"]):
                    case = source / "cases" / backend / str(seed) / fact.id
                    result = read_case(case, backend, seed, fact, rows)
                    if result is None:
                        raise ValueError(f"Baseline run is unfinished: {case}")
                    cases.append((backend, seed, fact, case, result[1]))
        for mode, mode_config in modes.items():
            if mode == default:
                continue
            check(effective_config(mode_config))
            jobs = []
            for backend, seed, fact, case, outcome in cases:
                destination = root / mode / "cases" / backend / str(seed) / fact.id
                label = f"{mode} {backend} seed={seed} {fact.id}"
                if outcome["status"] != "complete":
                    known = ([], dict(outcome, status="baseline_ineligible"))
                    jobs.append(_Job(label, destination, None, (), lambda known=known: known))
                    continue
                attempt = json.loads((case / "complete.json").read_text())["attempt"]
                jobs.append(_Job(label, destination, replay_ablation_case,
                                 (mode_config, backend, fact.to_dict(), seed, str(case / attempt)),
                                 partial(read_case, destination, backend, seed, fact, rows)))
            _execute(jobs, workers, root / mode / "case-outcomes.json")
    print(f"Ablation finished: {root}", flush=True)
