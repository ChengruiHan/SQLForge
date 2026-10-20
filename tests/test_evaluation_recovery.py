import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fixtures import build_demo

from sqlforge.evaluation import evaluate
from sqlforge.judges import Judge
from sqlforge.policies import ScriptedPolicy
from sqlforge.tasks import load_tasks


class RecoveryTests(unittest.TestCase):
    def test_saved_answer_regraded_without_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = load_tasks(build_demo(str(root / "data")))[0]
            policy = ScriptedPolicy(
                [{"name": "submit_solution", "arguments": {"sql_list": task.reference_sql}}]
            )
            evaluate([task], policy, Judge(), str(root / "first"), max_turns=4)
            record = json.loads((root / "first/episodes.jsonl").read_text())
            forbidden = Mock()
            forbidden.generate.side_effect = AssertionError("Resampled saved answer")
            result = evaluate(
                [task],
                forbidden,
                Judge(),
                str(root / "recovered"),
                max_turns=4,
                cached_records={task.task_id: record},
            )
            self.assertEqual(result["success"]["mean"], 1)
            forbidden.generate.assert_not_called()
            saved = json.loads((root / "recovered/episodes.jsonl").read_text())
            self.assertEqual(saved["trajectories"], record["trajectories"])
            self.assertEqual(saved["sampling_elapsed_s"], record["sampling_elapsed_s"])

    def test_recovery_refuses_changed_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = load_tasks(build_demo(str(Path(tmp) / "data")))[0]
            with self.assertRaises(ValueError):
                evaluate(
                    [task],
                    Mock(),
                    Judge(),
                    str(Path(tmp) / "bad"),
                    cached_records={task.task_id: {"fingerprint": "changed"}},
                )

    def test_generation_only_cache_selects_once_without_resampling(self):
        from unittest.mock import patch

        from sqlforge.evaluation import choose_without_gold

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = load_tasks(build_demo(str(root / "data")))[0]
            policy = ScriptedPolicy(
                [{"name": "submit_solution", "arguments": {"sql_list": task.reference_sql}}]
            )
            evaluate([task], policy, Judge(), str(root / "first"))
            row = json.loads((root / "first/episodes.jsonl").read_text())
            cached = {
                k: row[k] for k in ["task_id", "fingerprint", "trajectories", "sampling_elapsed_s"]
            }
            forbidden = Mock()
            forbidden.generate.side_effect = AssertionError("Resampled saved answer")
            with patch(
                "sqlforge.evaluation.choose_without_gold", wraps=choose_without_gold
            ) as select:
                result = evaluate(
                    [task],
                    forbidden,
                    Judge(),
                    str(root / "cached"),
                    cached_records={task.task_id: cached},
                )
            self.assertEqual(result["success"]["mean"], 1)
            self.assertEqual(select.call_count, 1)
            forbidden.generate.assert_not_called()
