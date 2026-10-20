"""Evaluation-only vLLM adapter. Training is delegated to verl."""

from __future__ import annotations

import json

from .model import student_seed
from .rollout import ContextOverflow, Generation


class VLLMPolicy:
    def __init__(
        self, engine, tokenizer, config, *, lora_request=None, temperature=0.7, eos_ids=None
    ):
        self.engine, self.tokenizer, self.config = engine, tokenizer, config
        self.lora_request, self.temperature = lora_request, temperature
        self.eos_ids = eos_ids or [tokenizer.eos_token_id]
        self.seed_offset = 0

    def prompt_ids(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

    def action_seed(self, messages):
        task_id = json.loads(messages[1]["content"])["task_id"]
        turn = sum(m["role"] == "assistant" for m in messages)
        return student_seed(task_id, self.config.seed + self.seed_offset + 10000 * turn)

    def generate(self, messages):
        result = self.generate_batch([messages])[0]
        if isinstance(result, ContextOverflow):
            raise result
        return result

    def generate_batch(self, conversations):
        from vllm import SamplingParams

        prompts = [self.prompt_ids(m) for m in conversations]
        results, valid = [None] * len(prompts), []
        for i, ids in enumerate(prompts):
            if len(ids) + self.config.max_new_tokens > self.config.max_context:
                results[i] = ContextOverflow("Prompt + generation reservation exceeds max_context")
            else:
                valid.append(i)
        if not valid:
            return results
        params = [
            SamplingParams(
                temperature=self.temperature,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                max_tokens=self.config.max_new_tokens,
                stop_token_ids=self.eos_ids,
                seed=self.action_seed(conversations[i]),
            )
            for i in valid
        ]
        outputs = self.engine.generate(
            [{"prompt_token_ids": prompts[i]} for i in valid],
            params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        if len(outputs) != len(valid):
            raise RuntimeError("vLLM response count differs from request count")
        for i, result in zip(valid, outputs):
            if list(result.prompt_token_ids) != prompts[i] or len(result.outputs) != 1:
                raise RuntimeError("vLLM changed the recorded prompt or returned extra samples")
            output = result.outputs[0]
            if output.finish_reason not in {"stop", "length"}:
                raise RuntimeError("Unexpected vLLM termination: " + str(output.finish_reason))
            tokens = list(output.token_ids)
            results[i] = Generation(
                self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                prompts[i],
                tokens,
                output.finish_reason == "length",
                len(prompts[i]),
                len(tokens),
            )
        return results
