"""Local chat transport. No credentials or full prompts in exception messages."""
import json
import os
import time
import urllib.error
import urllib.request


class ChatClient:
    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, messages, json_mode=False, max_tokens=None, response_format=None):
        body = {"model": self.config["model"], "messages": messages,
                "temperature": self.config.get("temperature", 0),
                "max_tokens": max_tokens or self.config.get("max_tokens", 512)}
        if response_format is not None:
            body["response_format"] = response_format
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.config.get("seed") is not None:
            body["seed"] = self.config["seed"]
        if self.config.get("reasoning_effort") is not None:
            body["reasoning_effort"] = self.config["reasoning_effort"]
        if self.config.get("chat_template_kwargs") is not None:
            body["chat_template_kwargs"] = self.config["chat_template_kwargs"]
        request = urllib.request.Request(
            self.config["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " +
                     os.environ.get(self.config.get("api_key_env", "LOCAL_LLM_API_KEY"), "local")})
        retries = self.config.get("retries", 2)
        for attempt in range(retries + 1):
            try:
                self.calls += 1
                with urllib.request.urlopen(request, timeout=self.config.get("timeout", 120)) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise RuntimeError("Model output truncated; increase max_tokens")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Endpoint returned empty or non-text content")
                if any(marker in content for marker in ("<think>", "<|channel|>analysis", "<|channel>thought")):
                    raise ValueError("Endpoint returned unparsed reasoning in content; configure its reasoning parser")
                if response_format and response_format.get("type") == "json_schema":
                    from jsonschema import ValidationError, validate
                    try:
                        validate(json.loads(content), response_format["json_schema"]["schema"])
                    except (json.JSONDecodeError, ValidationError):
                        raise ValueError("Endpoint response failed required JSON schema validation") from None
                for key in self.usage:
                    self.usage[key] += result.get("usage", {}).get(key, 0)
                return content
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == retries:
                    raise RuntimeError(f"Chat endpoint HTTP {error.code}; check local server configuration") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == retries:
                    raise RuntimeError("Local chat endpoint unreachable or timed out") from None
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")


class FixtureClient:
    """Deterministic transport fixture, NEVER evidence about an LLM."""
    calls = 0
    usage = {}

    def complete(self, messages, json_mode=False, max_tokens=None):
        self.calls += 1
        if json_mode:
            payload = json.loads(messages[-1]["content"])
            request = payload["request"].lower()
            selected = [r["id"] for r in payload["records"]
                        if any(w in r["text"].lower() for w in request.split()
                               if len(w) > 5)]
            return json.dumps({"ids": selected})
        system = messages[0]["content"]
        memories = json.loads(system.split("MEMORY_JSON:\n", 1)[1])
        question = messages[-1]["content"].lower()
        if "how do you know" in question or "still" in question:
            question += " " + messages[1]["content"].lower()
        ranked = sorted(memories, key=lambda m: len(set(question.split()) & set(m.lower().split())), reverse=True)
        if ranked and len(set(question.split()) & set(ranked[0].lower().split())) >= 2:
            return ranked[0]
        return "I don't know that information."
