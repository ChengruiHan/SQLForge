import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlforge.rollout import ContextOverflow
from sqlforge.vllm_policy import VLLMPolicy


class EvaluationPolicyTests(unittest.TestCase):
    def test_task_seeds_are_batch_independent_and_attempts_and_turns_differ(self):
        cfg = SimpleNamespace(seed=42, max_new_tokens=512, max_context=8192)
        policy = VLLMPolicy(None, SimpleNamespace(eos_token_id=2), cfg)
        messages = [
            {"role": "system", "content": "protocol"},
            {"role": "user", "content": json.dumps({"task_id": "TRAIN_1"})},
        ]
        first = policy.action_seed(messages)
        second_task = [messages[0], {"role": "user", "content": json.dumps({"task_id": "TRAIN_2"})}]
        self.assertNotEqual(first, policy.action_seed(second_task))
        self.assertEqual(first, policy.action_seed(messages))
        self.assertNotEqual(
            first,
            policy.action_seed(
                messages
                + [
                    {"role": "assistant", "content": "action"},
                    {"role": "user", "content": "observation"},
                ]
            ),
        )
        policy.seed_offset = 100000
        self.assertNotEqual(first, policy.action_seed(messages))

    def test_context_gate_token_identity_and_finish_reason_survive_batching(self):
        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return [1] * (
                    300 if json.loads(messages[1]["content"])["task_id"] == "long" else 20
                )

            def decode(self, ids, **kwargs):
                return "valid output"

        class Engine:
            def generate(self, prompts, params, **kwargs):
                self.params = params
                return [
                    SimpleNamespace(
                        prompt_token_ids=p["prompt_token_ids"],
                        outputs=[
                            SimpleNamespace(
                                token_ids=[8, 2], finish_reason="stop" if i == 0 else "length"
                            )
                        ],
                    )
                    for i, p in enumerate(prompts)
                ]

        engine = Engine()
        policy = VLLMPolicy(
            engine, Tokenizer(), SimpleNamespace(seed=42, max_new_tokens=50, max_context=100)
        )
        messages = [
            [
                {"role": "system", "content": "protocol"},
                {"role": "user", "content": json.dumps({"task_id": tid})},
            ]
            for tid in ["first", "long", "second"]
        ]
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SimpleNamespace)}):
            results = policy.generate_batch(messages)
        self.assertIsInstance(results[1], ContextOverflow)
        self.assertFalse(results[0].truncated)
        self.assertTrue(results[2].truncated)
        self.assertEqual(results[0].prompt_ids, [1] * 20)
        self.assertEqual(results[0].completion_ids, [8, 2])
        self.assertEqual(engine.params[0].stop_token_ids, [2])
        self.assertEqual(engine.params[0].top_k, -1)
