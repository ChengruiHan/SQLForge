from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from .database import Database
from .environment import EnvConfig, parse_action
from .harness import harness_manifest, validate_harness_env
from .judges import Judge, JudgeError
from .rollout import Policy, Trajectory, run_episode
from .tasks import Task, write_json
from .workflow import verify_audit


def choose_without_gold(task: Task, trajectories) -> int:
    """Select by public execution validity, then modal observed result, then earliest.

    Never call the hidden judge on candidates to select the best one. For management SQL,
    the schema and the final result are only a weak public selector; report that limitation.
    """
    signatures = []
    for trajectory in trajectories:
        if trajectory.submission is None:
            signatures.append(None)
            continue
        db = Database(task.db_path, task.setup_sql)
        try:
            result = db.execute(";\n".join(trajectory.submission))
            signature = None
            if result["ok"]:
                signature = json.dumps(
                    [result["columns"], result["rows"], db.schema()],
                    sort_keys=True,
                    ensure_ascii=False,
                )
            signatures.append(signature)
        finally:
            db.close()
    counts = Counter(s for s in signatures if s is not None)
    if not counts:
        return 0
    return max(range(len(trajectories)), key=lambda i: counts.get(signatures[i], 0))


def bootstrap_mean(values: list[float], seed=42, repetitions=2000) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return {
        "mean": statistics.mean(values),
        "ci95": [means[int(0.025 * repetitions)], means[int(0.975 * repetitions)]],
        "n": len(values),
    }


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[f"category:{row['category']}"].append(float(row["grade"]["success"]))
        groups[f"db:{row['db_id']}"].append(float(row["grade"]["success"]))
    return {
        "success": bootstrap_mean([float(r["grade"]["success"]) for r in records]),
        "mean_test_pass_fraction": statistics.mean(r["grade"]["score"] for r in records),
        "mean_calls_all_attempts": statistics.mean(r["total_calls"] for r in records),
        "mean_execute_calls_including_selector": statistics.mean(
            r["total_execute_calls"] for r in records
        ),
        "mean_generated_tokens_all_attempts": statistics.mean(
            r["total_generated_tokens"] for r in records
        )
        if all(r["token_usage_known"] for r in records)
        else None,
        "mean_elapsed_s": statistics.mean(r["elapsed_s"] for r in records),
        "truncated_attempts": sum(r["truncated_attempts"] for r in records),
        "diagnostics": diagnose_records(records),
        "mean_sampling_elapsed_s": statistics.mean(r["sampling_elapsed_s"] for r in records)
        if all("sampling_elapsed_s" in r for r in records)
        else None,
        "mean_selector_elapsed_s": statistics.mean(r["selector_elapsed_s"] for r in records)
        if all("selector_elapsed_s" in r for r in records)
        else None,
        "breakdown": {key: bootstrap_mean(values) for key, values in sorted(groups.items())},
        "selection": "public_execution_validity_then_modal_result; no hidden oracle",
        "ci_unit": "task; database breakdown also reported; not independent across same database",
    }


def diagnose_records(records: list[dict]) -> dict:
    """Separate task failures from per-attempt output/protocol diagnostics."""
    failures, attempts = Counter(), Counter()
    for row in records:
        selected = row["trajectories"][row["selected_attempt"]]
        if not row["grade"]["success"]:
            if selected["submission"] is None:
                failures["no_submission"] += 1
            elif row["grade"]["execution_success"]:
                failures["executable_but_incorrect"] += 1
            else:
                failures["submitted_sql_error"] += 1
        for trajectory in row["trajectories"]:
            attempts["total"] += 1
            attempts["no_submission"] += int(trajectory["submission"] is None)
            attempts["generation_or_context_limit"] += int(trajectory["truncated"])
            malformed = False
            for step in trajectory["steps"]:
                if step["generation"]["truncated"]:
                    continue
                try:
                    parse_action(step["generation"]["text"])
                except (ValueError, TypeError):
                    malformed = True
            attempts["format_failure"] += int(malformed)
            attempts["invalid_action"] += int(trajectory["invalid_actions"] > 0)
    return {
        "task_failures": dict(failures),
        "attempts": dict(attempts),
        "attempt_rates": {
            key: value / attempts["total"] for key, value in attempts.items() if key != "total"
        },
        "note": "Attempt counters overlap; truncation is reported separately from format failures",
    }


def evaluate(
    tasks: list[Task],
    policy: Policy,
    judge: Judge,
    output: str,
    *,
    method: str = "feedback",
    attempts: int = 1,
    max_turns: int = 6,
    metadata: dict | None = None,
    audit_report: str | None = None,
    cached_records: dict | None = None,
    hide_execution_feedback: bool = False,
    task_seeds: dict[str, int] | None = None,
    env_config: EnvConfig | None = None,
) -> dict:
    if not tasks:
        raise ValueError("No tasks selected")
    if attempts < 1 or method not in {"single", "feedback", "independent", "issue"}:
        raise ValueError("Invalid method/attempts")
    if method in {"single", "issue"} and attempts != 1:
        raise ValueError("single/issue must use exactly one attempt")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    audit = verify_audit(tasks, audit_report, judge)
    if task_seeds is not None and set(task_seeds) != {t.task_id for t in tasks}:
        raise ValueError("Seeds must cover exactly the selected tasks")
    if hide_execution_feedback and method != "feedback":
        raise ValueError("Feedback masking requires the feedback protocol")
    harness = harness_manifest(env_config.harness if env_config else None)
    if harness is not None:
        validate_harness_env(env_config)
        if method != "feedback" or hide_execution_feedback or max_turns != 4:
            raise ValueError(
                "Modular harness supports the fixed four-action feedback protocol only"
            )
    judge_identity = judge.identity()
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "manifest.json",
        {
            **(metadata or {}),
            "method": method,
            "attempts": attempts,
            "max_turns": max_turns,
            "direct_submission_only": method in {"single", "independent"},
            "task_ids": [t.task_id for t in tasks],
            "fingerprints": [t.fingerprint() for t in tasks],
            "judge_identity": judge_identity,
            "audit": audit,
            "hide_execution_feedback": hide_execution_feedback,
            "task_seeds": task_seeds,
            "env_config": __import__("dataclasses").asdict(env_config) if env_config else None,
            **({"harness": harness} if harness is not None else {}),
        },
    )
    cache = cached_records or {}
    if set(cache) - {t.task_id for t in tasks}:
        raise ValueError("Recovery contains tasks outside the evaluation split")
    for task in tasks:
        if task.task_id in cache and cache[task.task_id]["fingerprint"] != task.fingerprint():
            raise ValueError("Recovery task fingerprint changed")
        if task.task_id in cache and any(
            t.get("harness") != harness for t in cache[task.task_id]["trajectories"]
        ):
            raise ValueError("Recovery harness identity changed")
    records = []
    try:
        with (out / "episodes.jsonl").open("w") as handle:
            for task in tasks:
                started = time.monotonic()
                config = EnvConfig(
                    max_turns=max_turns if method == "feedback" else 1,
                    submit_only=method in {"single", "independent"},
                    hide_execution_feedback=hide_execution_feedback,
                )
                if env_config is not None:
                    from dataclasses import replace

                    config = replace(
                        env_config,
                        max_turns=max_turns if method == "feedback" else 1,
                        submit_only=method in {"single", "independent"},
                        hide_execution_feedback=hide_execution_feedback,
                    )
                if task_seeds is not None:
                    from .model import seed_everything

                    seed_everything(task_seeds[task.task_id])
                cached = cache.get(task.task_id)
                trajectories = (
                    [Trajectory(**t) for t in cached["trajectories"]]
                    if cached
                    else [run_episode(task, policy, config) for _ in range(attempts)]
                )
                if len(trajectories) != attempts:
                    raise ValueError("Recovery attempt count changed")
                sampled_at = time.monotonic()
                selected = choose_without_gold(task, trajectories)
                selected_at = time.monotonic()
                if (
                    cached
                    and "selected_attempt" in cached
                    and selected != cached["selected_attempt"]
                ):
                    raise ValueError("Recovery public selection changed")
                # Persist the exact selected answer BEFORE calling the hidden scorer.
                # Infrastructure retries must never resample a more favorable answer.
                write_json(
                    out / "pending.json",
                    {
                        "task_id": task.task_id,
                        "fingerprint": task.fingerprint(),
                        "selected_attempt": selected,
                        "trajectories": [t.asdict() for t in trajectories],
                    },
                )
                judge_retries = 0
                try:
                    grade = judge.grade(task, trajectories[selected].submission)
                except JudgeError:
                    judge_retries = 1
                    grade = judge.grade(task, trajectories[selected].submission)
                record = {
                    "task_id": task.task_id,
                    "fingerprint": task.fingerprint(),
                    "db_id": task.db_id,
                    "category": task.category,
                    "source": task.source,
                    "grade": grade.asdict(),
                    "judge_identity": judge_identity,
                    "judge_retries": judge_retries,
                    "selected_attempt": selected,
                    "total_calls": sum(len(t.steps) for t in trajectories),
                    "selector_execute_calls": sum(t.submission is not None for t in trajectories),
                    "total_execute_calls": sum(t.execute_calls for t in trajectories)
                    + sum(t.submission is not None for t in trajectories),
                    "total_generated_tokens": sum(t.generated_tokens for t in trajectories),
                    "token_usage_known": all(t.token_usage_known for t in trajectories),
                    "elapsed_s": time.monotonic() - started,
                    "agent_and_selector_elapsed_s": selected_at - started,
                    "sampling_elapsed_s": sampled_at - started,
                    "selector_elapsed_s": selected_at - sampled_at,
                    "judge_elapsed_s": time.monotonic() - selected_at,
                    "truncated_attempts": sum(t.truncated for t in trajectories),
                    "trajectories": [t.asdict() for t in trajectories],
                }
                if cached:
                    record["reused_generation"] = True
                    record["recovery_judge_elapsed_s"] = record["judge_elapsed_s"]
                    record["generation_cost_source"] = (
                        "saved" if "elapsed_s" in cached else "trajectory_only"
                    )
                    for field in [
                        "elapsed_s",
                        "agent_and_selector_elapsed_s",
                        "sampling_elapsed_s",
                        "selector_elapsed_s",
                    ]:
                        if field in cached:
                            record[field] = cached[field]
                    if "elapsed_s" not in cached:
                        record["sampling_elapsed_s"] = cached.get(
                            "sampling_elapsed_s", sum(t.elapsed_s for t in trajectories)
                        )
                        record["agent_and_selector_elapsed_s"] = (
                            record["sampling_elapsed_s"] + record["selector_elapsed_s"]
                        )
                        record["elapsed_s"] = (
                            record["agent_and_selector_elapsed_s"] + record["judge_elapsed_s"]
                        )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                (out / "pending.json").unlink()
    except Exception as exc:
        write_json(out / "FAILED.json", {"error": str(exc), "completed_tasks": len(records)})
        raise
    summary = summarize(records)
    summary["records_sha256"] = hashlib.sha256((out / "episodes.jsonl").read_bytes()).hexdigest()
    write_json(out / "summary.json", summary)
    return summary


def compare_runs(first: str, second: str) -> dict:
    def read(path):
        rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        mapping = {r["task_id"]: r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate task IDs in result file")
        return mapping

    a, b = read(first), read(second)
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires exactly matching task IDs")
    deltas, wins, losses = [], 0, 0
    for key in sorted(a):
        if a[key]["fingerprint"] != b[key]["fingerprint"]:
            raise ValueError(f"Different task content: {key}")
        if a[key].get("judge_identity") != b[key].get("judge_identity"):
            raise ValueError(f"Different evaluator: {key}")
        delta = int(b[key]["grade"]["success"]) - int(a[key]["grade"]["success"])
        deltas.append(delta)
        wins += int(delta > 0)
        losses += int(delta < 0)
    return {
        "difference_second_minus_first": bootstrap_mean(deltas),
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
    }
