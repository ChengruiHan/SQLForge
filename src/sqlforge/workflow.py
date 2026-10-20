"""Data audit is a reusable input artifact, not an unrelated reporting command."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .judges import Judge, JudgeError
from .tasks import Task, audit_splits, file_sha256


def audit_tasks(tasks: list[Task], judge: Judge, *, check_gold: bool) -> tuple[dict, list[Task]]:
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs across input manifests")
    report = {
        **audit_splits(tasks),
        "schema_version": 1,
        "gold_checked": check_gold,
        "judge_identity": judge.identity(),
    }
    accepted, rejected, checked, errors = [], [], [], []
    for task in tasks:
        valid = False
        if check_gold:
            gold = issue = None
            try:
                gold = judge.grade(task, task.reference_sql)
                issue = judge.grade(task, task.issue_sql)
            except (JudgeError, sqlite3.Error) as exc:
                # A failed audit is indeterminate, never a model reward of zero.
                # Keep the exclusion evidence without losing the rest of a data audit.
                errors.append({"task_id": task.task_id, "error": str(exc)})
                rejected.append(
                    {
                        "task_id": task.task_id,
                        "gold": gold.asdict() if gold is not None else None,
                        "issue": issue.asdict() if issue is not None else None,
                        "audit_error": str(exc),
                    }
                )
            else:
                valid = gold.success and not issue.success and gold.task_valid is not False
                if valid:
                    accepted.append(task)
                else:
                    rejected.append(
                        {"task_id": task.task_id, "gold": gold.asdict(), "issue": issue.asdict()}
                    )
        checked.append(
            {
                "task_id": task.task_id,
                "split": task.split,
                "fingerprint": task.fingerprint(),
                "valid": valid,
            }
        )
    report.update(
        validated=len(accepted), rejected=rejected, checked_tasks=checked, audit_errors=errors
    )
    return report, accepted


def verify_audit(tasks: list[Task], report_path: str | None, judge: Judge) -> dict | None:
    if report_path is None:
        if any(task.judge == "bird" for task in tasks):
            raise ValueError("BIRD runs require --audit-report from audit --check-gold")
        return None  # Synthetic fixtures may exercise the infrastructure directly.
    report = json.loads(Path(report_path).read_text())
    if not report.get("gold_checked") or report.get("schema_version") != 1:
        raise ValueError("Audit must contain completed gold/issue validation")
    if report.get("duplicate_tasks"):
        raise ValueError("Resolve duplicate tasks before using this audit")
    if report.get("judge_identity") != judge.identity():
        raise ValueError("Audit evaluator changed; rerun the audit")
    checked = {row["task_id"]: row for row in report["checked_tasks"]}
    for task in tasks:
        row = checked.get(task.task_id)
        if not row or not row["valid"] or row["split"] != task.split:
            raise ValueError(f"Task absent, rejected or reassigned in audit: {task.task_id}")
        if row["fingerprint"] != task.fingerprint():
            raise ValueError(f"Task changed after audit: {task.task_id}")
    return {"report_sha256": file_sha256(report_path), "tasks_verified": len(tasks)}
