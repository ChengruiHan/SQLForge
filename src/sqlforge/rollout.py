from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .environment import EnvConfig, SQLRepairEnv
from .harness import harness_manifest, render_messages, validate_harness_env
from .tasks import Task


@dataclass
class Generation:
    text: str
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Policy(Protocol):
    def generate(self, messages: list[dict]) -> Generation: ...


class ContextOverflow(RuntimeError):
    pass


@dataclass
class Trajectory:
    task_id: str
    steps: list[dict]
    submission: list[str] | None
    turns: int
    execute_calls: int
    execution_errors: int
    invalid_actions: int
    generated_tokens: int
    token_usage_known: bool
    truncated: bool
    termination: str
    elapsed_s: float
    harness: dict | None = None

    def asdict(self) -> dict:
        result = asdict(self)
        if self.harness is None:
            result.pop("harness")
        return result


def _episode(task: Task, config: EnvConfig | None = None):
    started = time.monotonic()
    config = config or EnvConfig()
    validate_harness_env(config)
    harness = harness_manifest(config.harness)
    env = SQLRepairEnv(task, config)
    steps, termination = [], "turn_limit"
    try:
        while not env.done:
            messages = render_messages(env.messages, config)
            try:
                generation = yield messages
            except ContextOverflow:
                termination = "context_limit"
                break
            if generation.truncated:
                # Never silently repair a cut-off JSON action or reward a partial submission.
                termination = "generation_limit"
                steps.append(
                    {"messages": messages, "generation": asdict(generation), "observation": None}
                )
                break
            observation = env.step(generation.text)
            steps.append(
                {"messages": messages, "generation": asdict(generation), "observation": observation}
            )
        if env.submission is not None:
            termination = "submitted"
        return Trajectory(
            task.task_id,
            steps,
            env.submission,
            env.turns,
            env.execute_calls,
            env.execution_errors,
            env.invalid_actions,
            sum(s["generation"]["completion_tokens"] for s in steps),
            all(s["generation"]["completion_tokens"] > 0 for s in steps),
            termination in {"generation_limit", "context_limit"},
            termination,
            time.monotonic() - started,
            harness,
        )
    finally:
        env.close()


def run_episode(task: Task, policy: Policy, config: EnvConfig | None = None) -> Trajectory:
    episode = _episode(task, config)
    try:
        messages = next(episode)
        while True:
            try:
                generation = policy.generate(messages)
            except ContextOverflow as exc:
                messages = episode.throw(exc)
            else:
                messages = episode.send(generation)
    except StopIteration as completed:
        return completed.value
    finally:
        episode.close()


def run_group(task: Task, policy: Policy, size: int, config: EnvConfig | None = None):
    """Batch active model turns while preserving one isolated DB per trajectory."""
    if not callable(getattr(policy, "generate_batch", None)):
        return [run_episode(task, policy, config) for _ in range(size)]
    return run_episodes([task] * size, policy, config)


def run_episodes(tasks, policy: Policy, config: EnvConfig | None = None):
    """Batch independent tasks, retaining isolated SQL state and complete per-task histories."""
    if not callable(getattr(policy, "generate_batch", None)):
        return [run_episode(task, policy, config) for task in tasks]
    size = len(tasks)
    episodes = [_episode(task, config) for task in tasks]
    results, active = [None] * size, {}
    try:
        for index, episode in enumerate(episodes):
            active[index] = next(episode)
        while active:
            indices = list(active)
            generations = policy.generate_batch([active[index] for index in indices])
            if len(generations) != len(indices):
                raise ValueError("Batched policy returned an invalid number of generations")
            for index, generation in zip(indices, generations):
                try:
                    if isinstance(generation, ContextOverflow):
                        active[index] = episodes[index].throw(generation)
                    else:
                        active[index] = episodes[index].send(generation)
                except StopIteration as completed:
                    results[index] = completed.value
                    del active[index]
        return results
    finally:
        for episode in episodes:
            episode.close()
