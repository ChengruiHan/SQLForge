"""Normalize downloaded benchmark records; does not generate demonstrations."""

import hashlib
import json
import random
from pathlib import Path

from .tasks import Task, audit_splits, write_json, write_tasks


def read_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array or a JSONL file, not an unspecified wrapper")
    return data


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
        raise ValueError("Expected SQL/test code as string or list[str]")
    return value


def import_bird(
    input_path: str,
    db_root: str,
    output: str,
    *,
    split: str,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Normalize public SQLite exports without executing dataset test code.

    Train/dev are disjoint by database. Official evaluation exports remain test-only.
    No guessed SQL translation or silent test fallback.
    """
    path, root = Path(input_path).resolve(), Path(db_root).resolve()
    records = read_records(path)
    tasks, excluded = [], []
    for row in records:
        # The first experiment is SELECT/query debugging. Do not silently mix in
        # management/DDL tasks with different persistence and scoring requirements.
        if str(row.get("category", "")).lower() != "query":
            excluded.append({"task_id": row.get("instance_id"), "category": row.get("category")})
            continue
        dialect = str(row.get("dialect", "SQLite")).lower()
        if "sqlite" not in dialect:
            raise ValueError(f"Non-SQLite task: {row.get('instance_id')}")
        db_id = str(row["db_id"])
        # Resolve paths under the explicit database root; reject traversal in external metadata.
        candidates = [
            root / db_id / f"{db_id}.sqlite",
            root / f"{db_id}.sqlite",
            root / db_id / f"{db_id}_template.sqlite",
        ]
        existing = [
            p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(root)
        ]
        if not existing:
            raise FileNotFoundError(f"No database for {db_id} under {root}")
        tests, reference = strings(row.get("test_cases", [])), strings(row.get("sol_sql", []))
        if not tests or not reference:
            raise ValueError(f"Missing private gold/tests: {row.get('instance_id')}")
        conditions = row.get("conditions", {})
        if isinstance(conditions, str):
            conditions = json.loads(conditions)
        tasks.append(
            Task(
                task_id=str(row["instance_id"]),
                db_id=db_id,
                db_path=str(existing[0]),
                question=row["query"],
                issue_sql=strings(row.get("issue_sql", [])),
                reference_sql=reference,
                setup_sql=strings(row.get("preprocess_sql", [])),
                test_cases=tests,
                conditions=conditions,
                category=row.get("category", "Query"),
                source=path.name,
                split=split,
                judge="bird",
            )
        )
    if split == "train":
        if not 0 < dev_fraction < 1:
            raise ValueError("dev_fraction must be between 0 and 1")
        dbs = sorted({t.db_id for t in tasks})
        if len(dbs) < 2:
            raise ValueError("At least two databases are required for database-disjoint train/dev")
        random.Random(seed).shuffle(dbs)
        n_dev = min(len(dbs) - 1, max(1, round(len(dbs) * dev_fraction)))
        dev = set(dbs[:n_dev])
        for task in tasks:
            task.split = "dev" if task.db_id in dev else "train"
    report = audit_splits(tasks)
    if report["duplicate_tasks"]:
        raise ValueError(
            f"Duplicate source tasks; inspect before importing: {report['duplicate_tasks'][:5]}"
        )
    write_tasks(output, tasks)
    report.update(
        {
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed,
            "scope": "Query only",
            "excluded_by_scope": excluded,
        }
    )
    write_json(str(output) + ".manifest.json", report)
    return report
