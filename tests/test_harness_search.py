import random
import string
import unittest
from types import SimpleNamespace

from sqlforge.harness import HarnessConfig
from sqlforge.harness_search import partition_development, select_candidate, validate_proposal


class HarnessSearchTests(unittest.TestCase):
    def tasks(self):
        rng = random.Random(13)
        tasks = [
            SimpleNamespace(
                task_id=str(i),
                split="dev",
                category="Query",
                db_id="a" if i < 64 else "b",
                question="".join(rng.choices(string.ascii_lowercase, k=70)),
            )
            for i in range(128)
        ]
        tasks[1].question = tasks[0].question
        return tasks

    def test_partition_balances_databases_and_keeps_duplicate_components(self):
        tasks = self.tasks()
        split = partition_development(tasks)
        self.assertEqual(len(split["search"]), 64)
        self.assertEqual(len(split["selection"]), 64)
        self.assertEqual("0" in split["search"], "1" in split["search"])
        for db in ["a", "b"]:
            self.assertEqual(sum(t.db_id == db and t.task_id in split["search"] for t in tasks), 32)
        self.assertEqual(split, partition_development(tasks))

    def test_cross_database_duplicates_remain_in_same_partition(self):
        tasks = self.tasks()
        tasks[64].question = tasks[0].question
        split = partition_development(tasks)
        self.assertEqual("0" in split["search"], "64" in split["search"])

    def test_partition_rejects_test_tasks(self):
        tasks = self.tasks()
        tasks[10].split = "test"
        with self.assertRaises(ValueError):
            partition_development(tasks)

    def scores(self, a, b, tokens=100):
        return {
            42: {"n": 64, "passed": a, "tokens": tokens},
            43: {"n": 64, "passed": b, "tokens": tokens},
        }

    def test_positive_mean_cannot_hide_a_seed_regression(self):
        decision = select_candidate({"H0": self.scores(20, 20), "candidate": self.scores(30, 19)})
        self.assertEqual(decision["selected"], "H0")

    def test_strict_improvement_and_token_tie_break(self):
        decision = select_candidate(
            {
                "H0": self.scores(20, 20),
                "long": self.scores(21, 20, 200),
                "short": self.scores(20, 21, 100),
            }
        )
        self.assertEqual(decision["selected"], "short")
        self.assertEqual(
            select_candidate({"H0": self.scores(20, 20), "tie": self.scores(20, 20, 1)})[
                "selected"
            ],
            "H0",
        )

    def test_missing_seed_or_unscored_tasks_cannot_be_ranked(self):
        incomplete = self.scores(21, 21)
        incomplete[42]["n"] = 63
        with self.assertRaises(ValueError):
            select_candidate({"H0": self.scores(20, 20), "candidate": incomplete})
        with self.assertRaises(ValueError):
            select_candidate({"H0": {42: self.scores(20, 20)[42]}})

    def test_task_database_and_duplicate_guidance_rejected(self):
        for text in ["For TRAIN_123 submit directly.", "For chinook always submit."]:
            with self.assertRaises(ValueError):
                validate_proposal(
                    {
                        "rationale": "example",
                        "candidate": {"prompt": {"additional_instruction": text}},
                    },
                    name="candidate",
                    existing=[],
                    database_names=["chinook"],
                )
        with self.assertRaises(ValueError):
            validate_proposal(
                {"rationale": "duplicate", "candidate": {}},
                name="candidate",
                existing=[HarnessConfig()],
                database_names=[],
            )

    def test_valid_proposal_cannot_change_environment_budgets(self):
        with self.assertRaises(ValueError):
            validate_proposal(
                {"rationale": "extra turn", "candidate": {"max_turns": 5}},
                name="candidate",
                existing=[],
                database_names=[],
            )
        config = validate_proposal(
            {
                "rationale": "Clearer public status",
                "candidate": {"feedback": {"style": "status_first"}},
            },
            name="candidate",
            existing=[],
            database_names=[],
        )
        self.assertEqual(config.feedback.style, "status_first")


if __name__ == "__main__":
    unittest.main()
