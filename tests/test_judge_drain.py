import tempfile
import unittest
from pathlib import Path

from fixtures import build_demo

from sqlforge.database import Database
from sqlforge.tasks import load_tasks


class DrainTests(unittest.TestCase):
    def test_preview_can_succeed_before_late_query_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = load_tasks(build_demo(str(Path(tmp) / "data")))[0]
            sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT x FROM n"
            db = Database(task.db_path, timeout=0.03)
            try:
                self.assertTrue(db.execute(sql)["ok"])
                result = db.execute(sql, drain_results=True)
                self.assertFalse(result["ok"])
                self.assertTrue(result["timeout"])
            finally:
                db.close()

    def test_drain_preserves_preview_and_does_not_apply_full_result_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = load_tasks(build_demo(str(Path(tmp) / "data")))[0]
            sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<25000) SELECT x FROM n"
            db = Database(task.db_path)
            try:
                result = db.execute(sql, drain_results=True)
                self.assertTrue(result["ok"])
                self.assertEqual(len(result["rows"]), 30)
                self.assertTrue(result["truncated"])
            finally:
                db.close()
