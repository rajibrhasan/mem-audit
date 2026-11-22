"""Fail-fast setup checks. Never generate a reply or alter memory."""
import importlib.metadata
import json
import os
import urllib.error
import urllib.request


def check(config, offline=False):
    if config["backends"] == ["fixture"]:
        return {"fixture": True, "ready": True, "model_check": "not applicable"}
    required = ["sentence-transformers", "chromadb", "qdrant-client"]
    if "mem0" in config["backends"]:
        required.append("mem0ai")
    if "amem" in config["backends"]:
        required.append("agentic-memory")
    versions = {}
    for name in required:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"Missing {name}; run bash slurm/setup.sh and use MEMAUDIT_PYTHON") from None
    if "mem0" in config["backends"] and versions["mem0ai"] != "2.2.1":
        raise RuntimeError("Mem0 adapter requires mem0ai==2.2.1")
    result = {"dependencies": versions, "model_check": "skipped" if offline else "pending",
              "ready": False, "embedding_model": config["embedding_model"]}
    if offline:
        return result
    llm = config["llm"]
    request = urllib.request.Request(llm["base_url"].rstrip("/") + "/models", headers={
        "Authorization": "Bearer " + os.environ.get(llm.get("api_key_env", "LOCAL_LLM_API_KEY"), "local")})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            models = json.load(response)
        found = [item for item in models["data"] if item["id"] == llm["model"]]
        if not found:
            raise RuntimeError(f"Configured model {llm['model']} is not listed by the local server")
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Cannot reach the local model server /models endpoint; start it or update llm.base_url") from None
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Local model server returned an invalid /models response") from None
    result.update({"ready": True, "model_check": "listed", "served_model": found[0]})
    return result
