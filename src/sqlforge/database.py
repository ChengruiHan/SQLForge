from __future__ import annotations

import base64
import sqlite3
import time
from pathlib import Path


def split_sql(script: str) -> list[str]:
    """Respect quoted semicolons and trigger bodies; SQLite validates each complete statement."""
    statements, pending = [], ""
    for char in script:
        pending += char
        if char == ";" and sqlite3.complete_statement(pending):
            statements.append(pending.strip())
            pending = ""
    if pending.strip():
        statements.append(pending.strip())
    return statements


def json_cell(value):
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode()}
    if isinstance(value, str):
        return value[:2000]
    return value


class Database:
    """One isolated, persistent in-memory copy per trajectory; source is opened read-only.

    The authorizer restricts SQL to this copy. This is not an OS sandbox for Python.
    """

    def __init__(
        self,
        path: str,
        setup_sql=(),
        timeout: float = 3.0,
        row_limit: int = 30,
        readonly: bool = False,
    ):
        self.timeout = timeout
        self.row_limit = row_limit
        self.conn = sqlite3.connect(":memory:")
        self.conn.enable_load_extension(False)
        try:
            with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as source:
                source.backup(self.conn)
            self.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8_000_000)
            self.conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
            for script in setup_sql:
                self.conn.executescript(script)
            self.conn.commit()
            if readonly:
                self.conn.execute("PRAGMA query_only=ON")
            self.conn.set_authorizer(self._authorize)
        except Exception:
            self.conn.close()
            raise

    @staticmethod
    def _authorize(action, arg1, arg2, database, trigger):
        if action in {
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_DETACH,
            sqlite3.SQLITE_CREATE_VTABLE,
            sqlite3.SQLITE_DROP_VTABLE,
        }:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {
            "load_extension",
            "readfile",
            "writefile",
        }:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA and str(arg1).lower() not in {
            "table_info",
            "table_xinfo",
            "index_info",
            "index_list",
            "foreign_key_list",
        }:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def schema(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, type, sql, 'main' FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' UNION ALL "
            "SELECT name, type, sql, 'temp' FROM sqlite_temp_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name LIMIT 150"
        ).fetchall()
        return [
            {"name": n, "type": t, "sql": (s or "")[:2000], "database": d} for n, t, s, d in rows
        ]

    def execute(self, sql: str, *, full_results: bool = False, drain_results: bool = False) -> dict:
        started = time.monotonic()
        deadline = started + self.timeout
        self.conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        last_rows, columns = [], []
        try:
            statements = split_sql(sql)
            if not statements or len(statements) > 30:
                raise ValueError("SQL must contain between 1 and 30 statements")
            for statement in statements:
                cursor = self.conn.execute(statement)
                if cursor.description:
                    columns = [c[0] for c in cursor.description]
                    limit = 20_000 if full_results else self.row_limit
                    last_rows = cursor.fetchmany(limit + 1)
                    if drain_results and not full_results:
                        # Validate the entire query without retaining an unbounded result.
                        while cursor.fetchmany(256):
                            if time.monotonic() > deadline:
                                raise TimeoutError("SQL execution timed out")
                    if full_results and len(last_rows) > limit:
                        raise ValueError(
                            "Result exceeds the judging limit; refusing truncated scoring"
                        )
                else:
                    columns, last_rows = [], []
                if time.monotonic() > deadline:
                    raise TimeoutError("SQL execution timed out")
            self.conn.commit()
            truncated = not full_results and len(last_rows) > self.row_limit
            rows = last_rows if full_results else last_rows[: self.row_limit]
            if not full_results:
                rows = [[json_cell(v) for v in row] for row in rows]
            return {
                "ok": True,
                "columns": columns,
                "rows": rows,
                "truncated": truncated,
                "elapsed_s": time.monotonic() - started,
            }
        except (sqlite3.Error, ValueError, TimeoutError) as exc:
            self.conn.rollback()
            return {
                "ok": False,
                "error": str(exc)[:600],
                "timeout": time.monotonic() >= deadline,
                "elapsed_s": time.monotonic() - started,
            }
        finally:
            self.conn.set_progress_handler(None, 0)

    def close(self):
        self.conn.close()
