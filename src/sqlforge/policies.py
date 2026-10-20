from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .rollout import Generation


class ScriptedPolicy:
    """Deterministic action replay for local integration checks; never a learned baseline."""

    def __init__(self, actions: list[dict]):
        self.actions = iter(actions)

    def generate(self, messages):
        try:
            action = next(self.actions)
        except StopIteration:
            action = {"name": "schema", "arguments": {}}
        return Generation(json.dumps(action))


class IssuePolicy:
    """Submit the broken input unchanged: checks whether the judge rejects the issue."""

    def generate(self, messages):
        public = json.loads(messages[1]["content"])
        return Generation(
            json.dumps({"name": "submit_solution", "arguments": {"sql_list": public["issue_sql"]}})
        )


class APIChatPolicy:
    """OpenAI-compatible HTTP transport; no OpenAI-specific SDK dependency.

    Use a local inference endpoint or an explicitly configured remote endpoint.
    API log probabilities are not used for RL training.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key_env: str = "SQLFORGE_API_KEY",
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.api_key = os.environ.get(api_key_env, "")

    def generate(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Inference HTTP {exc.code}") from exc
        choice = result["choices"][0]
        text = choice["message"].get("content")
        if not isinstance(text, str):
            raise RuntimeError("Endpoint must return text JSON actions in message.content")
        usage = result.get("usage", {})
        return Generation(
            text,
            truncated=choice.get("finish_reason") == "length",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
