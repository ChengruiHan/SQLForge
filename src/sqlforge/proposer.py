"""Bounded external proposal transport; reads credentials only from the caller."""

import json
import urllib.request

from .rollout import Generation
from .tasks import write_json


class ProposalClient:
    def __init__(self, config, output, key):
        self.config, self.output, self.key = config, output, key
        self.calls = 0
        if not key:
            raise ValueError("Set SQLFORGE_API_KEY for the explicitly requested proposal run")
        if config["base_url"] != "https://api.deepseek.com":
            raise ValueError("Expected the official DeepSeek API endpoint")

    def generate(self, messages):
        c = self.config
        if self.calls >= 4 or len(json.dumps(messages, ensure_ascii=False).encode()) + 1024 > 8192:
            raise ValueError("Proposal call or input limit exceeded")
        self.calls += 1
        payload = {
            k: c[k]
            for k in ["model", "thinking", "reasoning_effort", "max_tokens", "response_format"]
        }
        payload["messages"] = messages
        req = urllib.request.Request(
            c["base_url"] + "/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        choice = result["choices"][0]
        write_json(
            self.output / "response.json",
            {
                "model": result.get("model"),
                "usage": result.get("usage"),
                "content": choice["message"].get("content"),
                "finish_reason": choice.get("finish_reason"),
            },
        )
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("Missing proposal text")
        return Generation(content, truncated=choice.get("finish_reason") == "length")
