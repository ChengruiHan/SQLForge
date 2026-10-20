"""Evaluation-time, public-message-only harness modules for frozen H0 models.

Candidates are declarative JSON, not arbitrary executable code. Rendering cannot
query a database, grade an answer, change actions, or add model calls.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class PromptGuidance:
    additional_instruction: str = ""

    def __post_init__(self):
        if (
            not isinstance(self.additional_instruction, str)
            or len(self.additional_instruction) > 2048
        ):
            raise ValueError("additional_instruction must be text of at most 2048 characters")


@dataclass(frozen=True)
class FeedbackPresentation:
    style: str = "raw"
    row_preview: int = 5

    def __post_init__(self):
        if not isinstance(self.style, str) or self.style not in {"raw", "status_first"}:
            raise ValueError("Feedback style must be raw or status_first")
        if type(self.row_preview) is not int or not 0 <= self.row_preview <= 5:
            raise ValueError("row_preview must be an integer between 0 and 5")


@dataclass(frozen=True)
class HistoryState:
    mode: str = "full"
    keep_recent_observations: int = 1
    include_state: bool = False

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {"full", "compact_observations"}:
            raise ValueError("History mode must be full or compact_observations")
        if (
            type(self.keep_recent_observations) is not int
            or not 1 <= self.keep_recent_observations <= 3
        ):
            raise ValueError("keep_recent_observations must be between 1 and 3")
        if type(self.include_state) is not bool:
            raise ValueError("include_state must be boolean")


@dataclass(frozen=True)
class HarnessConfig:
    name: str = "H0"
    schema_version: int = 1
    prompt: PromptGuidance = PromptGuidance()
    feedback: FeedbackPresentation = FeedbackPresentation()
    history: HistoryState = HistoryState()

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.name):
            raise ValueError(
                "Harness name must contain 1–80 letters, digits, underscores or hyphens"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported harness schema_version")
        for key, cls in [
            ("prompt", PromptGuidance),
            ("feedback", FeedbackPresentation),
            ("history", HistoryState),
        ]:
            if not isinstance(getattr(self, key), cls):
                raise ValueError(f"{key} must be {cls.__name__}; use HarnessConfig.from_dict")

    @classmethod
    def from_dict(cls, value):
        def checked(data, target):
            if not isinstance(data, dict) or set(data) - {f.name for f in fields(target)}:
                raise ValueError(f"Unknown or invalid {target.__name__} fields")
            return dict(data)

        value = checked(value, cls)
        for key, target in [
            ("prompt", PromptGuidance),
            ("feedback", FeedbackPresentation),
            ("history", HistoryState),
        ]:
            if key in value:
                value[key] = target(**checked(value[key], target))
        return cls(**value)


def load_harness(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate harness JSON key: {key}")
            result[key] = value
        return result

    return HarnessConfig.from_dict(
        json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def harness_manifest(config):
    if config is None:
        return None
    if not isinstance(config, HarnessConfig):
        raise ValueError("Expected HarnessConfig")
    from .tasks import file_sha256

    source = Path(__file__).parent
    value = {
        "candidate": asdict(config),
        "candidate_sha256": _digest(asdict(config)),
        "source_sha256": {
            name: file_sha256(source / name)
            for name in ["harness.py", "environment.py", "rollout.py"]
        },
        "contract": "H0-public-messages-v1; fixed actions; no additional tool or model calls",
    }
    return {**value, "identity_sha256": _digest(value)}


def validate_harness_env(config):
    if config.harness is None:
        return
    if not isinstance(config.harness, HarnessConfig):
        raise ValueError("EnvConfig.harness requires HarnessConfig")
    if (
        config.prompt_version != "one_action_v2"
        or config.submit_only
        or config.hide_execution_feedback
    ):
        raise ValueError("Modular harness requires unmasked H0 feedback mode")
    if config.max_turns != 4 or config.row_limit != 5:
        raise ValueError("Modular harness keeps the H0 four-action and five-row budgets")
    if config.sql_timeout != 3.0:
        raise ValueError("Modular harness keeps the H0 three-second SQL timeout")
    if config.observation_token_limit != 384 or not config.observation_tokenizer:
        raise ValueError("Modular harness requires the pinned tokenizer and 384-token observations")


def _public_state(messages):
    """Summarize observable events; execution validity is never semantic validity."""
    from .environment import parse_action

    state = {"turns_used": 0, "execute_calls": 0, "execution_errors": 0, "invalid_actions": 0}
    for i in range(2, len(messages), 2):
        state["turns_used"] += 1
        observed = json.loads(messages[i + 1]["content"])
        state["remaining_turns"] = observed["remaining_turns"]
        try:
            action = parse_action(messages[i]["content"])
        except (ValueError, TypeError):
            state["invalid_actions"] += 1
            state["last_action"] = "invalid_action"
            continue
        state["last_action"] = action["name"]
        if action["name"] == "execute_sql":
            state["execute_calls"] += 1
            state["execution_errors"] += int(observed.get("ok") is False)
            state["last_execute_sql"] = action["arguments"]["sql"]
            state["last_execution_ok"] = observed.get("ok")
    return state


def render_messages(messages, config, *, tokenizer=None):
    """Render a fresh view of canonical env messages, never mutate env history.

    Assistant messages remain byte-for-byte intact, including invalid actions.
    This preserves turn-based inference seeds and the record of model decisions.
    """
    candidate = config.harness
    if candidate is None:
        return copy.deepcopy(messages)
    validate_harness_env(config)
    from .environment import (
        ONE_ACTION_V2_PROMPT,
        budget_observation,
        diagnostic_observation,
        observation_tokenizer,
        parse_action,
    )

    if (
        len(messages) < 2
        or len(messages) % 2
        or messages[0] != {"role": "system", "content": ONE_ACTION_V2_PROMPT}
    ):
        raise ValueError("Harness input must be canonical H0 messages before an action")
    # No-op H0 deliberately preserves JSON whitespace and message bytes.
    identity = (
        candidate.prompt == PromptGuidance()
        and candidate.feedback == FeedbackPresentation()
        and candidate.history.mode == "full"
        and not candidate.history.include_state
    )
    if identity:
        return copy.deepcopy(messages)
    view = copy.deepcopy(messages)
    if candidate.prompt.additional_instruction:
        view[0]["content"] += "\nAdditional guidance:\n" + candidate.prompt.additional_instruction
    observation_indices = list(range(3, len(messages), 2))
    state = (
        _public_state(messages) if candidate.history.include_state and observation_indices else None
    )
    for number, index in enumerate(observation_indices):
        visible = json.loads(messages[index]["content"])
        try:
            action = parse_action(messages[index - 1]["content"])["name"]
            invalid = False
        except (ValueError, TypeError):
            action, invalid = None, True
        if "rows" in visible and len(visible["rows"]) > candidate.feedback.row_preview:
            visible["rows"] = visible["rows"][: candidate.feedback.row_preview]
            visible["observation_truncated"] = True
        if (
            candidate.history.mode == "compact_observations"
            and number < len(observation_indices) - candidate.history.keep_recent_observations
        ):
            # Keep truncated error previews as evidence. Only successful or
            # schema-result previews may be omitted from older observations.
            removable = ["rows", "schema"]
            if not invalid and visible.get("ok") is not False:
                removable.append("observation_preview")
            for key in removable:
                if key in visible:
                    del visible[key]
                    visible["harness_history_compacted"] = True
        if candidate.feedback.style == "status_first":
            if (invalid or visible.get("ok") is False) and "error" not in visible:
                # The canonical environment may already have truncated a long
                # error into observation_preview. Never invent missing details.
                remaining = visible["remaining_turns"]
                visible = {
                    "action": action,
                    "status": "invalid_action" if invalid else "error",
                    "result": {k: v for k, v in visible.items() if k != "remaining_turns"},
                    "remaining_turns": remaining,
                }
            else:
                visible = diagnostic_observation(visible, action, invalid)
        if state is not None and index == observation_indices[-1]:
            visible["agent_state"] = state
        if tokenizer is None:
            tokenizer = observation_tokenizer(config.observation_tokenizer)
        # State summaries share the observation budget. They get no extra tokens.
        view[index]["content"] = budget_observation(
            visible, tokenizer, config.observation_token_limit
        )
    return view
