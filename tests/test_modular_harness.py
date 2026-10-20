import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fixtures import build_demo

from sqlforge.environment import ONE_ACTION_V2_PROMPT, EnvConfig, SQLRepairEnv
from sqlforge.eval_cli import collect, select_dev_tasks, verify_test_lock
from sqlforge.evaluation import evaluate
from sqlforge.harness import HarnessConfig, harness_manifest, load_harness, render_messages
from sqlforge.judges import Judge
from sqlforge.policies import ScriptedPolicy
from sqlforge.rollout import Generation, run_episode, run_episodes
from sqlforge.tasks import load_tasks, write_json
from sqlforge.vllm_policy import VLLMPolicy


class Tokenizer:
    # Deterministic budget fixture; not a substitute for Qwen-tokenized GPU runs.
    def encode(self, text, **kwargs):
        return [text[i : i + 4] for i in range(0, len(text), 4)]


class ModularHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = load_tasks(build_demo(str(self.root / "data")))[0]
        self.tokenizer_patch = patch(
            "sqlforge.environment.observation_tokenizer", return_value=Tokenizer()
        )
        self.tokenizer_patch.start()
        self.addCleanup(self.tokenizer_patch.stop)

    def config(self, **modules):
        return EnvConfig(
            max_turns=4,
            row_limit=5,
            prompt_version="one_action_v2",
            observation_token_limit=384,
            observation_tokenizer="fixture",
            harness=HarnessConfig.from_dict(modules),
        )

    def env(self, config=None):
        env = SQLRepairEnv(self.task, config or self.config())
        self.addCleanup(env.close)
        return env

    def test_explicit_h0_preserves_messages_bytes_and_seeds_at_every_turn(self):
        env = self.env()
        no_harness = replace(env.config, harness=None)
        policy = VLLMPolicy(None, None, SimpleNamespace(seed=42), eos_ids=[1])
        actions = [
            '{"name":"execute_sql","arguments":{"sql":"SELECT 1"}}',
            '{"name":"execute_sql","arguments":{"sql":"SELECT missing"}}',
            "invalid JSON",
        ]
        for text in [None, *actions]:
            if text:
                env.step(text)
            original = copy.deepcopy(env.messages)
            view = render_messages(original, env.config)
            self.assertEqual(view, render_messages(original, no_harness))
            self.assertEqual(policy.action_seed(view), policy.action_seed(original))
            view[0]["content"] = "changed copy"
            self.assertEqual(env.messages, original)

    def test_modules_change_only_public_view_and_keep_raw_error_evidence(self):
        self.task.reference_sql = ["PRIVATE_REFERENCE"]
        self.task.test_cases = ["PRIVATE_TEST"]
        self.task.conditions = {"PRIVATE_CONDITION": True}
        cfg = self.config(
            prompt={"additional_instruction": "Check the reported SQL error."},
            feedback={"style": "status_first"},
            history={"mode": "compact_observations", "include_state": True},
        )
        env = self.env(cfg)
        env.step('{"name":"execute_sql","arguments":{"sql":"SELECT missing_column"}}')
        env.step('{"name":"execute_sql","arguments":{"sql":"SELECT 1 AS value"}}')
        original = copy.deepcopy(env.messages)
        view = render_messages(original, cfg)
        self.assertEqual(env.messages, original)
        self.assertTrue(view[0]["content"].startswith(ONE_ACTION_V2_PROMPT))
        self.assertEqual(view[1], original[1])
        self.assertEqual(view[2::2], original[2::2])
        self.assertNotIn("PRIVATE_", json.dumps(view))
        error = json.loads(view[3]["content"])
        self.assertEqual(error["status"], "error")
        self.assertIn("missing_column", error["error"])
        last = json.loads(view[-1]["content"])
        self.assertEqual(last["status"], "success")
        self.assertEqual(last["agent_state"]["execution_errors"], 1)
        self.assertEqual(last["agent_state"]["remaining_turns"], 2)
        self.assertTrue(last["agent_state"]["last_execution_ok"])
        self.assertNotIn("correct", json.dumps(last))
        policy = VLLMPolicy(None, None, SimpleNamespace(seed=42), eos_ids=[1])
        self.assertEqual(policy.action_seed(view), policy.action_seed(original))

    def test_compaction_marks_omission_without_removing_actions_or_recent_observation(self):
        cfg = self.config(history={"mode": "compact_observations"})
        env = self.env(cfg)
        for _ in range(2):
            env.step('{"name":"execute_sql","arguments":{"sql":"SELECT 1"}}')
        view = render_messages(env.messages, cfg)
        earlier = json.loads(view[3]["content"])
        self.assertTrue(earlier["harness_history_compacted"])
        self.assertNotIn("rows", earlier)
        self.assertEqual(view[-1], env.messages[-1])
        self.assertEqual(view[2::2], env.messages[2::2])

    def test_row_preview_never_executes_or_changes_submission(self):
        cfg = self.config(feedback={"row_preview": 0})
        env = self.env(cfg)
        env.step('{"name":"execute_sql","arguments":{"sql":"SELECT id FROM customers"}}')
        original = copy.deepcopy(env.messages)
        view = render_messages(env.messages, cfg)
        value = json.loads(view[-1]["content"])
        self.assertEqual(value["rows"], [])
        self.assertTrue(value["observation_truncated"])
        self.assertTrue(json.loads(original[-1]["content"])["rows"])
        self.assertEqual(env.execute_calls, 1)
        self.assertIsNone(env.submission)
        self.assertEqual(env.turns, 1)

    def test_long_truncated_error_and_state_stay_inside_observation_budget(self):
        cfg = self.config(feedback={"style": "status_first"}, history={"include_state": True})
        env = self.env(cfg)
        # A valid action with a very long missing table produces a truncated raw observation.
        env.step(
            json.dumps({"name": "execute_sql", "arguments": {"sql": "SELECT * FROM " + "x" * 3000}})
        )
        view = render_messages(env.messages, cfg)
        for message in view[3::2]:
            self.assertLessEqual(len(Tokenizer().encode(message["content"])), 384)
            value = json.loads(message["content"])
            self.assertEqual(value["remaining_turns"], 3)
            self.assertEqual(value["status"], "error")

    def test_compaction_preserves_truncated_error_preview(self):
        cfg = self.config(history={"mode": "compact_observations"})
        env = self.env(cfg)
        character_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
        with patch("sqlforge.environment.observation_tokenizer", return_value=character_tokenizer):
            env.step(
                json.dumps(
                    {"name": "execute_sql", "arguments": {"sql": "SELECT * FROM " + "x" * 3000}}
                )
            )
            before = json.loads(env.messages[-1]["content"])
            self.assertIn("observation_preview", before)
            env.step('{"name":"execute_sql","arguments":{"sql":"SELECT 1"}}')
            earlier = json.loads(render_messages(env.messages, cfg)[3]["content"])
        self.assertEqual(earlier, before)

    def test_real_rollout_and_batched_rollout_record_the_actual_rendered_prompt(self):
        cfg = self.config(prompt={"additional_instruction": "A test instruction."})
        action = {"name": "submit_solution", "arguments": {"sql_list": self.task.reference_sql}}
        seen = []

        class Policy:
            def generate_batch(self, conversations):
                seen.extend(copy.deepcopy(conversations))
                return [Generation(json.dumps(action)) for _ in conversations]

        rows = run_episodes([self.task, self.task], Policy(), cfg)
        single = run_episode(self.task, ScriptedPolicy([action]), cfg)
        for row in [*rows, single]:
            self.assertIn("A test instruction.", row.steps[0]["messages"][0]["content"])
            self.assertEqual(row.harness, harness_manifest(cfg.harness))
            self.assertEqual(row.execute_calls, 0)
            self.assertEqual(row.turns, 1)
        self.assertEqual([r.steps[0]["messages"] for r in rows], seen)

    def test_harness_cache_cannot_be_reused_across_candidates_or_as_plain_h0(self):
        cfg = self.config()
        action = {"name": "submit_solution", "arguments": {"sql_list": self.task.reference_sql}}
        result = evaluate(
            [self.task],
            ScriptedPolicy([action]),
            Judge(),
            str(self.root / "first"),
            max_turns=4,
            env_config=cfg,
        )
        self.assertEqual(result["success"]["mean"], 1)
        row = json.loads((self.root / "first/episodes.jsonl").read_text())
        forbidden = Mock(generate=Mock(side_effect=AssertionError("Unexpected inference")))
        for name, target in [
            ("different", self.config(name="different")),
            ("legacy", replace(cfg, harness=None)),
        ]:
            with self.assertRaisesRegex(ValueError, "harness identity"):
                evaluate(
                    [self.task],
                    forbidden,
                    Judge(),
                    str(self.root / name),
                    max_turns=4,
                    env_config=target,
                    cached_records={self.task.task_id: row},
                )
        result = evaluate(
            [self.task],
            forbidden,
            Judge(),
            str(self.root / "same"),
            max_turns=4,
            env_config=cfg,
            cached_records={self.task.task_id: row},
        )
        self.assertEqual(result["success"]["mean"], 1)
        forbidden.generate.assert_not_called()

    def test_harness_does_not_force_submit_or_repair_invalid_actions_at_budget_end(self):
        cfg = self.config(history={"include_state": True})

        class InvalidPolicy:
            def generate(self, messages):
                return Generation("not JSON")

        trajectory = run_episode(self.task, InvalidPolicy(), cfg)
        self.assertEqual(trajectory.turns, 4)
        self.assertEqual(len(trajectory.steps), 4)
        self.assertEqual(trajectory.invalid_actions, 4)
        self.assertEqual(trajectory.execute_calls, 0)
        self.assertIsNone(trajectory.submission)
        self.assertEqual(trajectory.termination, "turn_limit")
        self.assertEqual(
            json.loads(trajectory.steps[-1]["messages"][-1]["content"])["remaining_turns"], 1
        )

    def test_sampling_cache_refuses_another_harness(self):
        cfg = self.config()
        folder = self.root / "samples"
        folder.mkdir()
        write_json(
            folder / (self.task.task_id + ".json"),
            {
                "fingerprint": self.task.fingerprint(),
                "trajectory": {"harness": None},
            },
        )
        with self.assertRaisesRegex(ValueError, "identity changed"):
            collect([self.task], Mock(), cfg, folder, 1)

    def test_candidate_schema_cannot_change_budgets_and_rejects_duplicate_keys(self):
        for value in [
            {"max_turns": 8},
            {"feedback": {"row_preview": 6}},
            {"history": {"include_state": "true"}},
            {"schema_version": True},
            {"prompt": {"additional_instruction": "x" * 2049}},
        ]:
            with self.assertRaises(ValueError):
                HarnessConfig.from_dict(value)
        path = self.root / "candidate.json"
        path.write_text('{"name":"a","name":"b"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_harness(path)
        for cfg in [
            replace(self.config(), max_turns=5),
            replace(self.config(), sql_timeout=10),
            replace(self.config(), submit_only=True),
            replace(self.config(), observation_token_limit=512),
        ]:
            with self.assertRaises(ValueError):
                run_episode(self.task, Mock(), cfg)

    def test_harness_test_lock_and_fixed_dev_subset(self):
        manifest = harness_manifest(self.config().harness)
        lock = {"checkpoint_rule": "final256-no-test-selection", "adapter_sha256": {"raw": None}}
        verify_test_lock({"raw": {"adapter": None}}, ["raw"], lock)
        with self.assertRaisesRegex(ValueError, "Test harness"):
            verify_test_lock({"raw": {"adapter": None}}, ["raw"], lock, harness=manifest)
        lock["harness_sha256"] = manifest["identity_sha256"]
        verify_test_lock({"raw": {"adapter": None}}, ["raw"], lock, harness=manifest)
        tasks = [SimpleNamespace(task_id=str(i), split="dev") for i in range(4)]
        self.assertEqual([t.task_id for t in select_dev_tasks(tasks, ["3", "1"], 2)], ["3", "1"])
        for ids in [["1", "1"], ["unknown", "1"], ["1"]]:
            with self.assertRaises(ValueError):
                select_dev_tasks(tasks, ids, 2)


if __name__ == "__main__":
    unittest.main()
