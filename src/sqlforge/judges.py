from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .database import Database
from .tasks import Task, file_sha256

BIRD_REVISION = "b26c4285ef69dfb6c096b076a7499018c6b370ca"


class JudgeError(RuntimeError):
    """Infrastructure failure: abort the run rather than train on a bogus zero reward."""


@dataclass
class Grade:
    score: float
    success: bool
    execution_success: bool
    passed: int
    total: int
    judge: str
    error: str = ""
    task_valid: bool | None = None

    def asdict(self):
        return asdict(self)


class ExecutionJudge:
    """Read-only result equivalence for local fixtures, NOT the official BIRD metric.

    Preserves duplicate rows; tests every declared hidden database. Never strips DISTINCT,
    ROUND, ORDER BY, or columns. Explicit ordered=True compares output order as well.
    """

    def grade(self, task: Task, submission: list[str] | None) -> Grade:
        if task.judge != "execution":
            raise JudgeError("BIRD tasks require the BIRD judge; no fallback scoring")
        paths = [task.db_path, *task.hidden_db_paths]
        if not submission:
            return Grade(0.0, False, False, 0, len(paths), "execution")
        passed, execution_success = 0, True
        error = ""
        for path in paths:
            expected_db = Database(path, task.setup_sql, readonly=True)
            actual_db = Database(path, task.setup_sql, readonly=True)
            try:
                expected = expected_db.execute(";\n".join(task.reference_sql), full_results=True)
                if not expected["ok"]:
                    raise JudgeError(f"Invalid reference for {task.task_id}: {expected['error']}")
                actual = actual_db.execute(";\n".join(submission), full_results=True)
                execution_success &= actual["ok"]
                if not actual["ok"]:
                    error = actual["error"]
                    continue
                a, b = actual["rows"], expected["rows"]
                equal = a == b if task.ordered else Counter(map(tuple, a)) == Counter(map(tuple, b))
                equal &= len(actual["columns"]) == len(expected["columns"])
                passed += int(equal)
            finally:
                actual_db.close()
                expected_db.close()
        total = len(paths)
        return Grade(
            passed / total, passed == total, execution_success, passed, total, "execution", error
        )


class BirdJudge:
    """Invoke the pinned upstream evaluator in a subprocess against disposable DB copies.

    Trusted benchmark Python tests run in that process, not inside the agent. This wrapper
    supplies a wall-clock timeout, not an OS security sandbox for arbitrary test code.
    """

    def __init__(self, eval_dir: str, python: str = sys.executable, timeout: float = 30):
        self.eval_dir = Path(eval_dir).resolve()
        self.script = self.eval_dir / "single_instance_eval.py"
        self.python, self.timeout = python, timeout
        if not self.script.is_file():
            raise JudgeError(f"Missing upstream evaluator: {self.script}")
        self.identity()

    def identity(self) -> dict:
        provenance = self.eval_dir / "UPSTREAM.json"
        if not provenance.is_file():
            raise JudgeError("Missing UPSTREAM.json; fetch the pinned evaluator first")
        record = json.loads(provenance.read_text())
        expected = {
            "LICENSE",
            "__init__.py",
            "single_instance_eval.py",
            "db_utils.py",
            "test_utils.py",
            "logger.py",
            "utils.py",
        }
        if (
            record.get("revision") != BIRD_REVISION
            or set(record.get("files_sha256", {})) != expected
        ):
            raise JudgeError("Unexpected upstream evaluator revision/files")
        for name, sha in record["files_sha256"].items():
            if file_sha256(self.eval_dir / name) != sha:
                raise JudgeError(f"Upstream evaluator changed: {name}")
        return {
            "revision": BIRD_REVISION,
            "files_sha256": record["files_sha256"],
            "timeout": self.timeout,
            "python": str(Path(self.python).resolve()),
        }

    def grade(self, task: Task, submission: list[str] | None) -> Grade:
        if not task.test_cases:
            raise JudgeError(f"Missing hidden test_cases: {task.task_id}")
        if not submission:
            return Grade(0.0, False, False, 0, len(task.test_cases), "bird")
        # Apply the same SQL capability restrictions as the agent environment before handing
        # a candidate to the upstream runner (which itself has no SQL authorizer).
        preflight = Database(
            task.db_path, task.setup_sql, readonly=task.category.lower() == "query"
        )
        try:
            checked = preflight.execute(";\n".join(submission), drain_results=True)
        finally:
            preflight.close()
        if not checked["ok"]:
            return Grade(
                0.0,
                False,
                False,
                0,
                len(task.test_cases),
                "bird",
                (
                    "Candidate full-execution timeout (3s)"
                    if checked.get("timeout")
                    else checked["error"]
                ),
            )
        with tempfile.TemporaryDirectory(prefix="sqlforge-judge-") as temp:
            work = Path(temp)
            db_path = work / "candidate.sqlite"
            # backup() also includes a source WAL; plain copyfile() would miss pending pages.
            with sqlite3.connect(Path(task.db_path).as_uri() + "?mode=ro", uri=True) as source:
                for target in [db_path, work / "candidate_template.sqlite"]:
                    with sqlite3.connect(target) as dest:
                        source.backup(dest)
            payload = {
                "instance_id": task.task_id,
                "db_id": task.db_id,
                "issue_sql": task.issue_sql,
                "sol_sql": task.reference_sql,
                "pred_sqls": submission,
                "preprocess_sql": task.setup_sql,
                "clean_up_sql": [],
                "test_cases": task.test_cases,
                "conditions": task.conditions,
            }
            input_path, output_path = work / "input.jsonl", work / "output.json"
            input_path.write_text(json.dumps(payload) + "\n")
            env = {**os.environ, "EPHEMERAL_DB_PATH": str(db_path), "SQL_REWARD_DB_DIR": str(work)}
            cmd = [
                self.python,
                str(self.script),
                "--jsonl_file",
                str(input_path),
                "--output_file",
                str(output_path),
                "--mode",
                "pred",
                "--logging",
                "false",
            ]
            with (work / "process.log").open("w+") as log:
                proc = subprocess.Popen(
                    cmd, cwd=work, env=env, stdout=log, stderr=log, start_new_session=True
                )
                try:
                    proc.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired as exc:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    # Ambiguous tool/test infrastructure timeout is not a model failure reward.
                    raise JudgeError(f"BIRD evaluator timed out: {task.task_id}") from exc
                if proc.returncode or not output_path.is_file():
                    log.seek(0)
                    raise JudgeError(f"Upstream evaluator failed: {log.read(2000)}")
            result = json.loads(output_path.read_text())
        if result.get("issue_sql_passed_unexpectedly"):
            # Dataset defects are auditable rejected tasks, not infrastructure failures.
            return Grade(
                0.0,
                False,
                True,
                0,
                len(task.test_cases),
                "bird",
                "Issue SQL already passes hidden tests",
                task_valid=False,
            )
        total = result.get("total_test_cases", 0)
        passed = result.get("passed_test_cases", 0)
        if total <= 0:
            raise JudgeError(f"Evaluator returned no tests: {task.task_id}")
        success = result.get("status") == "success"
        exec_ok = not (
            result.get("evaluation_phase_execution_error")
            or result.get("evaluation_phase_timeout_error")
        )
        return Grade(
            float(passed / total) if exec_ok else 0.0,
            success,
            exec_ok,
            passed,
            total,
            "bird",
            result.get("error_message", "")[:1500],
            task_valid=True,
        )


class Judge:
    def __init__(self, bird_eval_dir: str | None = None, bird_python: str = sys.executable):
        self.execution = ExecutionJudge()
        self.bird = BirdJudge(bird_eval_dir, bird_python) if bird_eval_dir else None

    def identity(self) -> dict:
        from .database import __file__ as database_file

        return {
            "bird": self.bird.identity() if self.bird else None,
            "wrapper_sha256": file_sha256(__file__),
            "database_sha256": file_sha256(database_file),
            "execution_metric": "duplicate-preserving-readonly-v1",
        }

    def grade(self, task: Task, submission: list[str] | None) -> Grade:
        if task.judge == "execution":
            return self.execution.grade(task, submission)
        if self.bird is None:
            raise JudgeError("Pass --bird-eval-dir for official BIRD scoring")
        return self.bird.grade(task, submission)
