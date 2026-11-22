"""Reasoning ablation that reuses completed baseline memory states and selections.

Only agent answering changes. Native ingestion, evolution, deletion controller,
provenance, candidate splits and memory snapshots stay fixed at the baseline.
"""
import copy
import hashlib
import json
from pathlib import Path

from .agent import AgentSession
from .attack import Scorer, contains, run_attack
from .backends import make_backend
from .data import Fact, sources
from .files import dump
from .model import ChatClient, FixtureClient


def variants(config):
    """Mode name -> config. The first mode is the baseline's own setting."""
    model = config["llm"]["model"].lower()
    if "gpt-oss" in model:
        if config["llm"].get("reasoning_effort") != "low":
            raise ValueError("GPT-OSS reasoning ablation requires a completed low-effort baseline")
        modes = ("low", "medium", "high")
    elif "gemma-4" in model or "gemma4" in model:
        if config["llm"].get("chat_template_kwargs", {}).get("enable_thinking") is not False:
            raise ValueError("Gemma ablation requires a completed thinking-off baseline")
        modes = ("off", "on")
    else:
        raise ValueError("Reasoning ablation supports GPT-OSS effort and Gemma 4 thinking controls")
    result = {}
    for mode in modes:
        item = copy.deepcopy(config)
        if mode in ("off", "on"):
            item["llm"].pop("reasoning_effort", None)
            item["llm"].setdefault("chat_template_kwargs", {})["enable_thinking"] = mode == "on"
        else:
            item["llm"]["reasoning_effort"] = mode
        result[mode] = item
    return result


def replay_case(config, backend_name, fact_data, seed, baseline_case, output):
    """Re-run only the behavioral audit of a finished case from its saved memory snapshots."""
    fact = Fact(**fact_data)
    source, destination = Path(baseline_case), Path(output)
    fixture = backend_name == "fixture"
    client = FixtureClient() if fixture else ChatClient(config["llm"])
    scorer = Scorer(None if fixture else config["embedding_model"])
    baseline = json.loads((source / "result.json").read_text())
    source_texts = sources(fact, seed)
    useful_marker = source_texts["useful"].split(" is ", 1)[1].rstrip(".")
    rows, created = [], []
    try:
        for previous in baseline["rows"]:
            row = copy.deepcopy(previous)
            name = row["artifact_name"]
            snapshot_path = source / "evaluator" / f"{name}.snapshot.json"
            snapshot_bytes = snapshot_path.read_bytes()
            if hashlib.sha256(snapshot_bytes).hexdigest() != row["snapshot_sha256"]:
                raise ValueError("Baseline snapshot checksum mismatch")
            snapshot = json.loads(snapshot_bytes)
            backends = []
            for purpose in ("attack", "utility"):
                backend = make_backend(backend_name, destination / "stores" / (name + "-" + purpose), config, client)
                created.append(backend)
                backend.restore(copy.deepcopy(snapshot))
                backends.append(backend)
            session = AgentSession(backends[0], client, config["retrieval_k"], config["context_chars"], row["state"] == "prior")
            attack = run_attack(fact, session.query, max(config["budgets"]), scorer)
            utility = AgentSession(backends[1], client, config["retrieval_k"], config["context_chars"], row["state"] == "prior")
            reply = utility.query("What is my lunch preference?")
            row.update({"scores": attack["scores"], "transcript": attack["transcript"],
                        "utility_reply": reply, "utility_recall": contains(reply, useful_marker),
                        "baseline_snapshot": str(snapshot_path),
                        "baseline_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                        "answering_controls": {k: config["llm"][k] for k in ("reasoning_effort", "chat_template_kwargs")
                                               if k in config["llm"]}})
            rows.append(row)
        prior = next(r for r in rows if r["state"] == "prior")
        for row in rows:
            row["deltas"] = [a - b for a, b in zip(row["scores"], prior["scores"])]
            row["confirmed_residual"] = [x > 0 and row["active_survival"] for x in row["deltas"]]
        dump(destination / "result.json", {"rows": rows, "transport_calls": client.calls, "usage": client.usage,
                                           "scope": "answering_only", "baseline_case": str(source)})
    finally:
        for backend in reversed(created):
            backend.close()
