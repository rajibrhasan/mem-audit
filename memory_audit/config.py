"""Load and validate experiment configuration files."""
import json
from pathlib import Path

REQUIRED_KEYS = {
    "backends", "llm", "embedding_model", "embedding_dims", "facts", "seeds", "budgets",
    "deletion_methods", "wordings", "defenses", "retrieval_k", "context_chars", "bootstrap",
    "public_controls",
}
CHOICES = {
    "backends": {"fixture", "mem0", "amem"},
    "deletion_methods": {"api", "natural_language"},
    "wordings": {"direct", "paraphrase", "broad"},
    "defenses": {"none", "cascade"},
}
POSITIVE_INTEGERS = ("facts", "retrieval_k", "context_chars", "bootstrap", "embedding_dims")
LLM_KEYS = {
    "base_url", "model", "api_key_env", "temperature", "max_tokens", "memory_max_tokens",
    "timeout", "retries", "seed", "reasoning_effort", "chat_template_kwargs",
}
# Project policy: no models from Chinese developers.
EXCLUDED_MODEL_NAMES = ("qwen", "deepseek", "glm", "baichuan", "internlm", "minimax", "hunyuan")


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate(config)
    return config


def validate(config):
    if set(config) != REQUIRED_KEYS:
        raise ValueError(f"Config keys mismatch: missing={REQUIRED_KEYS - set(config)}, "
                         f"extra={set(config) - REQUIRED_KEYS}")
    for field, allowed in CHOICES.items():
        values = config[field]
        if (not isinstance(values, list) or not values or len(values) != len(set(values))
                or not set(values) <= allowed):
            raise ValueError(f"Invalid {field}")
    if "fixture" in config["backends"] and config["backends"] != ["fixture"]:
        raise ValueError("Fixture and real backends must run in separate output directories")
    for key in POSITIVE_INTEGERS:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["facts"] < 4:
        raise ValueError("Need at least four synthetic facts")
    for key in ("seeds", "budgets"):
        values = config[key]
        if not values or len(values) != len(set(values)) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError(f"Invalid {key}")
    budgets = config["budgets"]
    if 1 not in budgets or min(budgets) < 1:
        raise ValueError("Budgets must include single-query baseline 1")
    if budgets != sorted(budgets):
        raise ValueError("Budgets must be sorted")
    _validate_llm(config)
    if type(config["public_controls"]) is not bool:
        raise ValueError("public_controls must be boolean")
    if config["backends"] != ["fixture"]:
        llm = config["llm"]
        if not llm.get("model") or not llm.get("base_url", "").startswith(("http://", "https://")):
            raise ValueError("Specify local model and base_url")
        if any(name in llm["model"].lower() for name in EXCLUDED_MODEL_NAMES):
            raise ValueError("This project excludes Chinese-developed models")


def _validate_llm(config):
    llm = config["llm"]
    if set(llm) - LLM_KEYS:
        raise ValueError(f"Unknown LLM options: {set(llm) - LLM_KEYS}")
    effort = llm.get("reasoning_effort")
    if effort not in (None, "none", "low", "medium", "high"):
        raise ValueError("reasoning_effort must be none, low, medium or high")
    if "gpt-oss" in llm.get("model", "").lower() and effort == "none":
        raise ValueError("GPT-OSS supports low/medium/high effort, not a no-thinking mode")


def expected_rows(config):
    """Evaluated memory states per case.

    Four controls (prior, never stored, stored, verified clean) plus, for both the
    ordinary and the combined-summary test, one state per deletion path and defense.
    """
    paths = sum(len(config["wordings"]) if method == "natural_language" else 1
                for method in config["deletion_methods"])
    return 4 + 2 * paths * len(config["defenses"])
