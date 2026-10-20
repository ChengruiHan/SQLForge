import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fixtures import build_demo

from sqlforge.database import Database, split_sql
from sqlforge.environment import EnvConfig, SQLRepairEnv, parse_action
from sqlforge.evaluation import choose_without_gold, compare_runs, evaluate
from sqlforge.judges import BirdJudge, ExecutionJudge, Judge, JudgeError
from sqlforge.policies import IssuePolicy, ScriptedPolicy
from sqlforge.review import review_baselines
from sqlforge.rollout import ContextOverflow, Generation, run_episode, run_group
from sqlforge.tasks import load_tasks, write_tasks


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = build_demo(str(Path(self.temp.name) / "demo"))
        self.tasks = load_tasks(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_demo_gold_passes_and_issue_fails(self):
        judge = ExecutionJudge()
        for task in self.tasks:
            self.assertTrue(judge.grade(task, task.reference_sql).success)
            self.assertFalse(judge.grade(task, task.issue_sql).success)

    def test_batched_group_keeps_termination_and_turn_state_independent(self):
        task = self.tasks[0]

        class BatchedFixture:
            calls = 0

            def generate_batch(self, conversations):
                self.calls += 1

                def generation(action, truncated=False):
                    return Generation(json.dumps(action), [3], [4], truncated, 1, 1)

                submit = {"name": "submit_solution", "arguments": {"sql_list": task.reference_sql}}
                if self.calls == 1:
                    return [
                        generation(submit),
                        generation(
                            {"name": "execute_sql", "arguments": {"sql": task.issue_sql[0]}}
                        ),
                        ContextOverflow("too long"),
                        generation(submit, truncated=True),
                    ]
                self.assert_only_one_active = len(conversations) == 1
                return [generation(submit)]

        policy = BatchedFixture()
        trajectories = run_group(task, policy, 4, EnvConfig(max_turns=4))
        self.assertTrue(policy.assert_only_one_active)
        self.assertEqual(
            [t.termination for t in trajectories],
            ["submitted", "submitted", "context_limit", "generation_limit"],
        )
        self.assertEqual([t.turns for t in trajectories], [1, 2, 0, 0])
        self.assertEqual([t.execute_calls for t in trajectories], [0, 1, 0, 0])
        self.assertEqual([bool(t.submission) for t in trajectories], [True, True, False, False])

    def test_private_fields_do_not_reach_policy(self):
        task = self.tasks[0]
        task.reference_sql = ["SELECT 'HIDDEN_GOLD_SENTINEL'"]
        task.test_cases = ["HIDDEN_TEST_SENTINEL"]
        task.conditions = {"HIDDEN_CONDITION": True}
        env = SQLRepairEnv(task)
        try:
            visible = json.dumps(env.messages)
            self.assertNotIn("HIDDEN", visible)
            self.assertNotIn("db_path", visible)
        finally:
            env.close()

    def test_masked_execution_keeps_sql_state_and_hides_results_and_errors(self):
        normal = SQLRepairEnv(self.tasks[0], EnvConfig(max_turns=4))
        masked = SQLRepairEnv(self.tasks[0], EnvConfig(max_turns=4, hide_execution_feedback=True))
        try:
            self.assertEqual(normal.messages, masked.messages)
            for sql, expected_ok in [
                ("SELECT 246813579 AS secret_value", True),
                ("SELECT * FROM secret_missing_table", False),
            ]:
                result = masked.step(json.dumps({"name": "execute_sql", "arguments": {"sql": sql}}))
                self.assertEqual(result["ok"], expected_ok)
                visible = json.loads(masked.messages[-1]["content"])
                self.assertEqual(visible, {"accepted": True, "remaining_turns": 4 - masked.turns})
                self.assertNotIn("secret", json.dumps(visible))
            self.assertEqual(masked.execute_calls, 2)
            self.assertEqual(masked.execution_errors, 1)
            result = masked.step('{"name":"schema","arguments":{}}')
            self.assertIn("schema", json.loads(masked.messages[-1]["content"]))
            self.assertIn("schema", result)
            result = masked.step(
                json.dumps(
                    {
                        "name": "submit_solution",
                        "arguments": {"sql_list": self.tasks[0].reference_sql},
                    }
                )
            )
            self.assertTrue(result["accepted"])
            self.assertEqual(masked.submission, self.tasks[0].reference_sql)
        finally:
            normal.close()
            masked.close()

    def test_masked_feedback_never_hides_action_format_error(self):
        env = SQLRepairEnv(self.tasks[0], EnvConfig(hide_execution_feedback=True))
        try:
            result = env.step("not JSON")
            self.assertIn("Invalid action", result["error"])
            self.assertIn("Invalid action", env.messages[-1]["content"])
            self.assertEqual(env.execute_calls, 0)
        finally:
            env.close()

    def test_database_persistence_and_cross_episode_isolation(self):
        a = Database(self.tasks[0].db_path)
        b = Database(self.tasks[0].db_path)
        try:
            self.assertTrue(
                a.execute("CREATE TABLE scratch(x); INSERT INTO scratch VALUES(7)")["ok"]
            )
            self.assertEqual(a.execute("SELECT x FROM scratch")["rows"], [[7]])
            self.assertFalse(b.execute("SELECT x FROM scratch")["ok"])
            with sqlite3.connect(self.tasks[0].db_path) as source:
                self.assertEqual(
                    source.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE name='scratch'"
                    ).fetchone()[0],
                    0,
                )
        finally:
            a.close()
            b.close()

    def test_timeouts_and_sql_authorizer(self):
        db = Database(self.tasks[0].db_path, timeout=0.01)
        try:
            self.assertFalse(db.execute("ATTACH DATABASE ':memory:' AS other")["ok"])
            self.assertFalse(db.execute("PRAGMA writable_schema=ON")["ok"])
            result = db.execute(
                "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x"
            )
            self.assertFalse(result["ok"])
            self.assertTrue(result["timeout"])
        finally:
            db.close()

    def test_sql_semicolon_in_string(self):
        self.assertEqual(len(split_sql("SELECT ';'; SELECT 2;")), 2)

    def test_budget_does_not_fallback_to_last_execute(self):
        task = self.tasks[0]
        policy = ScriptedPolicy(
            [{"name": "execute_sql", "arguments": {"sql": task.reference_sql[0]}}]
        )
        trajectory = run_episode(task, policy, EnvConfig(max_turns=1))
        self.assertIsNone(trajectory.submission)
        self.assertEqual(trajectory.termination, "turn_limit")
        self.assertEqual(Judge().grade(task, trajectory.submission).score, 0)

    def test_direct_repair_disallows_tools_and_does_not_execute_them(self):
        env = SQLRepairEnv(self.tasks[0], EnvConfig(max_turns=1, submit_only=True))
        try:
            self.assertNotIn('"name":"execute_sql"', env.messages[0]["content"])
            self.assertIn('"name":"submit_solution"', env.messages[0]["content"])
            result = env.step(json.dumps({"name": "execute_sql", "arguments": {"sql": "SELECT 1"}}))
            self.assertIn("tools are disabled", result["error"])
            self.assertEqual(env.execute_calls, 0)
            self.assertEqual(env.invalid_actions, 1)
            self.assertIsNone(env.submission)
        finally:
            env.close()

    def test_independent_selection_grades_only_selected_answer_and_records_costs(self):
        from unittest.mock import patch

        task = self.tasks[0]
        policy = ScriptedPolicy(
            [
                {"name": "submit_solution", "arguments": {"sql_list": sql}}
                for sql in [
                    task.issue_sql,
                    task.reference_sql,
                    task.reference_sql,
                    task.reference_sql,
                ]
            ]
        )
        judge = Judge()
        with patch.object(judge, "grade", wraps=judge.grade) as grade:
            result = evaluate(
                [task],
                policy,
                judge,
                str(Path(self.temp.name) / "independent"),
                method="independent",
                attempts=4,
            )
        self.assertEqual(grade.call_count, 1)
        self.assertEqual(result["success"]["mean"], 1)
        self.assertEqual(result["mean_execute_calls_including_selector"], 4)
        self.assertEqual(result["diagnostics"]["attempts"]["total"], 4)
        self.assertGreaterEqual(result["mean_selector_elapsed_s"], 0)

    def test_scorer_retry_keeps_identical_answer_and_persistent_failure_keeps_pending(self):
        from unittest.mock import patch

        task = self.tasks[0]
        action = {"name": "submit_solution", "arguments": {"sql_list": task.reference_sql}}
        judge = Judge()
        correct = judge.grade(task, task.reference_sql)
        out = Path(self.temp.name) / "retry"
        policy = ScriptedPolicy([action])
        with (
            patch.object(policy, "generate", wraps=policy.generate) as generate,
            patch.object(
                judge, "grade", side_effect=[JudgeError("transient timeout"), correct]
            ) as grade,
        ):
            result = evaluate([task], policy, judge, str(out), method="single")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(grade.call_args_list[0], grade.call_args_list[1])
        self.assertEqual(result["success"]["mean"], 1)
        self.assertFalse((out / "pending.json").exists())
        failed = Path(self.temp.name) / "persistent"
        with patch.object(judge, "grade", side_effect=JudgeError("persistent timeout")):
            with self.assertRaises(JudgeError):
                evaluate([task], ScriptedPolicy([action]), judge, str(failed), method="single")
        pending = json.loads((failed / "pending.json").read_text())
        self.assertEqual(pending["trajectories"][0]["submission"], task.reference_sql)
        self.assertEqual((failed / "episodes.jsonl").read_text(), "")

    def test_baseline_review_distinguishes_cross_condition_and_trajectory_changes(self):
        tasks = self.tasks[:2]
        for task in tasks:
            task.split = "dev"
        manifest_path = Path(self.temp.name) / "dev.jsonl"
        write_tasks(manifest_path, tasks)

        def submit(sql):
            return {"name": "submit_solution", "arguments": {"sql_list": sql}}

        def execute(sql):
            return {"name": "execute_sql", "arguments": {"sql": ";".join(sql)}}

        a, b = tasks
        actions = {
            "single": [submit(a.reference_sql), submit(b.issue_sql)],
            "independent4": [submit(a.issue_sql)] * 4 + [submit(b.reference_sql)] * 4,
            "feedback": [
                execute(a.reference_sql),
                execute(a.issue_sql),
                submit(a.issue_sql),
                execute(b.issue_sql),
                execute(b.reference_sql),
                submit(b.reference_sql),
            ],
        }
        paths = {}
        for name, sequence in actions.items():
            paths[name] = Path(self.temp.name) / name
            evaluate(
                tasks,
                ScriptedPolicy(sequence),
                Judge(),
                str(paths[name]),
                method="independent" if name == "independent4" else name,
                attempts=4 if name == "independent4" else 1,
                max_turns=4,
                metadata={"seed": 42, "temperature": 0.7, "model": "fixture", "adapter": "same"},
            )
        result = review_baselines(manifest_path, paths, Path(self.temp.name) / "review", Judge())
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["transitions"]["feedback_pass_single_fail"], 1)
        self.assertEqual(result["transitions"]["single_pass_feedback_fail"], 1)
        self.assertEqual(result["transitions"]["feedback_first_execute_wrong_final_correct"], 1)
        self.assertEqual(result["transitions"]["feedback_had_correct_execute_final_failed"], 1)
        self.assertEqual(result["diagnostic_judge_calls"], 4)

    def test_duplicate_keys_and_multiple_actions_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('{"name":"schema","name":"execute_sql","arguments":{}}')
        with self.assertRaises(ValueError):
            parse_action('{"name":"schema","arguments":{}} {"name":"schema","arguments":{}}')

    def test_submission_only_acknowledged(self):
        env = SQLRepairEnv(self.tasks[0])
        try:
            feedback = env.step(
                json.dumps(
                    {
                        "name": "submit_solution",
                        "arguments": {"sql_list": self.tasks[0].reference_sql},
                    }
                )
            )
            self.assertEqual(set(feedback), {"accepted", "remaining_turns"})
            with self.assertRaises(RuntimeError):
                env.step('{"name":"schema","arguments":{}}')
        finally:
            env.close()

    def test_bird_tasks_never_use_demo_judge(self):
        task = self.tasks[0]
        task.judge = "bird"
        with self.assertRaises(JudgeError):
            Judge().grade(task, task.reference_sql)

    def test_candidate_selection_cannot_use_gold(self):
        task = self.tasks[0]
        bad = run_episode(task, IssuePolicy(), EnvConfig(max_turns=1))
        good = run_episode(
            task,
            ScriptedPolicy(
                [{"name": "submit_solution", "arguments": {"sql_list": task.reference_sql}}]
            ),
            EnvConfig(max_turns=1),
        )
        selected = choose_without_gold(task, [bad, good])
        # Both executable with different results: tie goes to earliest, even though it is wrong.
        self.assertEqual(selected, 0)
        task.reference_sql = ["deliberately not SQL"]
        self.assertEqual(choose_without_gold(task, [bad, good]), selected)

    def test_fingerprint_covers_database_setup_and_private_tests(self):
        task = self.tasks[0]
        initial = task.fingerprint()
        task.test_cases = ["new hidden assertion"]
        self.assertNotEqual(task.fingerprint(), initial)
        task.test_cases = []
        self.assertEqual(task.fingerprint(), initial)
        task.setup_sql = ["UPDATE customers SET region='changed'"]
        self.assertNotEqual(task.fingerprint(), initial)
        task.setup_sql = []
        with sqlite3.connect(task.db_path) as db:
            db.execute("UPDATE customers SET region='changed'")
        self.assertNotEqual(task.fingerprint(), initial)

    def test_evaluation_and_paired_reporting(self):
        out = str(Path(self.temp.name) / "run")
        report = evaluate(self.tasks, IssuePolicy(), Judge(), out, method="issue")
        self.assertEqual(report["success"]["mean"], 0)
        self.assertIsNone(report["mean_generated_tokens_all_attempts"])
        paired = compare_runs(str(Path(out) / "episodes.jsonl"), str(Path(out) / "episodes.jsonl"))
        self.assertEqual(paired["difference_second_minus_first"]["mean"], 0)
        with self.assertRaises(FileExistsError):
            evaluate(self.tasks, IssuePolicy(), Judge(), out)

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "vendor/bird-evaluator"
            / "single_instance_eval.py"
        ).is_file(),
        "Fetch pinned upstream evaluator",
    )
    def test_actual_upstream_evaluator_integration(self):
        evaluator = Path(__file__).resolve().parents[1] / "vendor/bird-evaluator"
        task = self.tasks[3]  # Refund logic: executable issue, wrong result.
        task.judge = "bird"
        task.test_cases = [
            "def test_case(pred_sqls, sol_sqls, db_path, conn, conditions):\n"
            "    assert ex_base(pred_sqls, sol_sqls, db_path, conn, conditions) == 1\n"
        ]
        judge = BirdJudge(str(evaluator), timeout=15)
        self.assertTrue(judge.grade(task, task.reference_sql).success)
        self.assertFalse(judge.grade(task, task.issue_sql).success)
        self.assertFalse(judge.grade(task, ["ATTACH ':memory:' AS extra"]).success)


if __name__ == "__main__":
    unittest.main()
