from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def database_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    wal = Path(str(path) + "-wal")
    return {"main": file_sha256(path), "wal": file_sha256(wal) if wal.exists() else None}


@dataclass
class Task:
    task_id: str
    db_id: str
    db_path: str
    question: str
    issue_sql: list[str]
    reference_sql: list[str]
    setup_sql: list[str] = field(default_factory=list)
    test_cases: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    category: str = "Query"
    source: str = "local"
    split: str = "train"
    judge: str = "bird"
    # Demo checks may use multiple hidden databases; not exposed to the agent.
    hidden_db_paths: list[str] = field(default_factory=list)
    ordered: bool = False

    def public(self) -> dict:
        return {
            "task_id": self.task_id,
            "db_id": self.db_id,
            "question": self.question,
            "issue_sql": self.issue_sql,
        }

    def input_fingerprint(self) -> str:
        # Private answer/test differences must not hide the same question across splits.
        value = {
            "db_id": self.db_id,
            "question": " ".join(self.question.split()),
            "issue_sql": self.issue_sql,
            "setup_sql": self.setup_sql,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def fingerprint(self) -> str:
        # Ignore source/split/path/identifier, but include everything defining behavior
        # and scoring. The same JSON path must not conceal changed DBs or private tests.
        content = {
            "db_id": self.db_id,
            "database": database_identity(self.db_path),
            "hidden_databases": [database_identity(p) for p in self.hidden_db_paths],
            "question": self.question.strip(),
            "issue_sql": self.issue_sql,
            "reference_sql": self.reference_sql,
            "setup_sql": self.setup_sql,
            "test_cases": self.test_cases,
            "conditions": self.conditions,
            "category": self.category,
            "judge": self.judge,
            "ordered": self.ordered,
        }
        return hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_tasks(path: str | Path, tasks: list[Task]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in tasks))


def load_tasks(path: str | Path, split: str | None = None) -> list[Task]:
    path = Path(path).resolve()
    tasks = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        t = Task(**json.loads(line))
        if t.task_id in seen:
            raise ValueError(f"Duplicate task_id at {path}:{line_number}: {t.task_id}")
        seen.add(t.task_id)
        if split is not None and t.split != split:
            continue
        t.db_path = str((path.parent / t.db_path).resolve())
        t.hidden_db_paths = [str((path.parent / p).resolve()) for p in t.hidden_db_paths]
        if not Path(t.db_path).is_file():
            raise FileNotFoundError(t.db_path)
        if not t.question or not t.reference_sql:
            raise ValueError(f"Missing question or private reference: {t.task_id}")
        if t.judge not in {"bird", "execution"}:
            raise ValueError(f"Unknown judge: {t.judge}")
        tasks.append(t)
    return tasks


def audit_splits(tasks: list[Task]) -> dict:
    seen: dict[str, str] = {}
    duplicates = []
    dbs: dict[str, set[str]] = {}
    for t in tasks:
        key = t.input_fingerprint()
        if key in seen:
            duplicates.append([seen[key], t.task_id])
        seen[key] = t.task_id
        dbs.setdefault(t.split, set()).add(t.db_id)
    overlaps = {}
    names = sorted(dbs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}/{b}"] = sorted(dbs[a] & dbs[b])
    return {
        "tasks": len(tasks),
        "duplicate_tasks": duplicates,
        "databases": {s: sorted(v) for s, v in dbs.items()},
        "database_overlap": overlaps,
    }
