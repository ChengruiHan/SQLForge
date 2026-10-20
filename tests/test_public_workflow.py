"""Exercise the portable search and release boundaries without GPU/API calls."""

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlforge.eval_cli import freeze, verify_test_lock
from sqlforge.harness import HarnessConfig, PromptGuidance
from sqlforge.tasks import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from search_harness import Experiment  # noqa: E402


class SimulatedExperiment(Experiment):
    def __init__(self, root, improve):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.events = []
        self.improve = improve

    def prepare(self):
        self.plan = {"weights": {"raw": None, "grpo": "fixed"}}
        write_json(self.root / "PLAN.json", self.plan)
        self.static = [
            HarnessConfig(name=n, prompt=PromptGuidance(str(i)))
            for i, n in enumerate(["H0", "prompt", "feedback", "history"])
        ]
        self.search = [f"search{i}" for i in range(64)]
        self.selection = [f"selection{i}" for i in range(64)]

    def start_engine(self):
        pass

    def condition(self, phase, arm, config, seed, tasks, **kwargs):
        self.events.append((phase, arm, config.name, seed))
        if phase == "dev-search":
            self.assert_no_selection()
        path = self.root / phase / arm / config.name / str(seed)
        passed = 20 if config.name == "H0" or not self.improve else 25
        write_json(path / "score.json", {"n": 64, "passed": passed, "tokens": 100})
        return path

    def assert_no_selection(self):
        assert not (self.root / "SELECTION.json").exists()

    def wait_scores(self, paths):
        return {name: json.loads((p / "score.json").read_text()) for name, p in paths.items()}

    def proposal(self, slot, configs, paths, scores):
        self.assert_no_selection()
        assert all("dev-search" in str(p) for p in paths.values())
        self.events.append(("proposal", slot))
        return HarnessConfig(name=f"adaptive{slot}", prompt=PromptGuidance(f"proposal{slot}"))


class PublicWorkflowTests(unittest.TestCase):
    def test_import_only_requires_frozen_databases(self):
        from sqlforge.cli import main
        from sqlforge.tasks import load_tasks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with sqlite3.connect(root / "included.sqlite") as conn:
                conn.execute("CREATE TABLE example (id INTEGER)")
            rows = [
                {
                    "instance_id": tid,
                    "db_id": db,
                    "category": "Query",
                    "query": "Return one",
                    "issue_sql": ["SELECT 2"],
                    "sol_sql": ["SELECT 1"],
                    "test_cases": ["assert True"],
                }
                for tid, db in [("TRAIN_1", "included"), ("TRAIN_2", "not_downloaded")]
            ]
            write_json(root / "input.json", rows)
            write_json(root / "splits.json", {"TRAIN_1": "train"})
            with patch.object(
                sys,
                "argv",
                [
                    "sqlforge",
                    "import",
                    "--input",
                    str(root / "input.json"),
                    "--db-root",
                    str(root),
                    "--split-map",
                    str(root / "splits.json"),
                    "--output",
                    str(root / "output.jsonl"),
                ],
            ):
                main()
            tasks = load_tasks(root / "output.jsonl")
            self.assertEqual([(t.task_id, t.split) for t in tasks], [("TRAIN_1", "train")])

    def test_search_locks_without_test_and_does_not_retry_selection(self):
        for improve in [False, True]:
            with self.subTest(improve=improve), tempfile.TemporaryDirectory() as tmp:
                e = SimulatedExperiment(Path(tmp), improve)
                e.run()
                report = json.loads((e.root / "FINAL_REPORT.json").read_text())
                self.assertEqual(report["selected"] == "H0", not improve)
                self.assertFalse(report["test_used"])
                self.assertTrue((e.root / "FINAL_LOCK.json").exists())
                self.assertEqual(len([x for x in e.events if x[0] == "proposal"]), 4)
                self.assertEqual(len([x for x in e.events if x[0] == "dev-selection"]), 6)
                self.assertFalse(any(x[0] == "test" for x in e.events))
                self.assertEqual(
                    len([x for x in e.events if x[0] == "dev-transfer"]), 4 if improve else 2
                )

    def test_interrupted_api_reservation_consumes_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment.__new__(Experiment)
            e.root = Path(tmp)
            write_json(e.root / "proposals/slot4/request.json", {"already_sent": True})
            with patch("search_harness.ProposalClient") as api:
                self.assertIsNone(e.proposal(4, {}, {}, {}))
                api.assert_not_called()

    def test_lock_detects_changed_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, adapter = root / "model", root / "adapter"
            model.mkdir()
            adapter.mkdir()
            (model / "model.safetensors").write_bytes(b"fixture base")
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture adapter")
            write_json(adapter / "adapter_config.json", {"r": 16})
            write_json(
                root / "arms.json", {"raw": {"adapter": None}, "grpo": {"adapter": str(adapter)}}
            )
            (root / "tasks.jsonl").write_text("fixture")
            args = SimpleNamespace(
                arms=str(root / "arms.json"),
                output=str(root / "lock.json"),
                model=str(model),
                tasks=str(root / "tasks.jsonl"),
                harness=None,
            )
            freeze(args)
            arms = json.loads((root / "arms.json").read_text())
            lock = json.loads((root / "lock.json").read_text())
            verify_test_lock(arms, list(arms), lock)
            self.assertEqual(lock["tasks_sha256"], file_sha256(root / "tasks.jsonl"))
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_test_lock(arms, list(arms), lock)
            with self.assertRaises(FileExistsError):
                freeze(args)

    def test_native_objective_configuration_and_portable_model_path(self):
        spec = importlib.util.spec_from_file_location("public_train", ROOT / "scripts/train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "DATA.json",
                {"model": "/portable/base", "max_prompt_length": 2048, "max_response_length": 3328},
            )
            for method, objective in [("grpo", "vanilla"), ("gspo", "gspo"), ("rloo", "gpg")]:
                _, cfg = module.overrides(root, method, False)
                self.assertEqual(cfg["actor_rollout_ref.model.path"], "/portable/base")
                self.assertEqual(cfg["actor_rollout_ref.actor.policy_loss.loss_mode"], objective)
                self.assertEqual(cfg["actor_rollout_ref.rollout.n"], 4)
                self.assertEqual(cfg["algorithm.use_kl_in_reward"], method == "rloo")
                self.assertEqual(cfg["trainer.total_training_steps"], 64)


if __name__ == "__main__":
    unittest.main()
