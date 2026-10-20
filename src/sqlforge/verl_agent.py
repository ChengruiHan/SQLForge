"""H0 environment adapter for verl's native asynchronous AgentLoop.

Only generated action tokens receive policy loss. Public tool observations and
chat boundaries have mask zero; private grading runs after the episode ends.
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

from .environment import EnvConfig, SQLRepairEnv
from .judges import Judge, JudgeError
from .tasks import load_tasks, write_json


@lru_cache(maxsize=1)
def task_map():
    return {t.task_id: t for t in load_tasks(os.environ["SQLFORGE_VERL_TASKS"])}


def append_observation(tokenizer, generated, observation):
    """Qwen3 chat continuation, preserving sampled tokens without re-tokenizing.

    vLLM includes the stop token when generated. If not, the missing boundary is
    environment context (mask zero), never invented model supervision.
    """
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    boundary = "" if generated and generated[-1] == end else "<|im_end|>"
    text = boundary + "\n<|im_start|>user\n" + observation + "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer.encode(text, add_special_tokens=False)


class SQLForgeAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        task = task_map()[kwargs["extra_info"]["task_id"]]
        model = self.config.actor_rollout_ref.model.path
        record_id = uuid.uuid4().hex
        folder = Path(os.environ["SQLFORGE_VERL_TRAJECTORIES"])
        folder.mkdir(parents=True, exist_ok=True)
        # Each SQLite connection is created, used and closed on the same thread.
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()

        async def call(fn, *args):
            return await loop.run_in_executor(executor, fn, *args)

        env = await call(
            SQLRepairEnv,
            task,
            EnvConfig(
                max_turns=4,
                row_limit=5,
                observation_token_limit=384,
                observation_tokenizer=model,
                prompt_version="one_action_v2",
            ),
        )
        prompt = self.tokenizer.apply_chat_template(
            env.messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Dataset prompts must be the exact H0 public prompt, not a second template.
        assert len(prompt) <= self.config.data.max_prompt_length
        response, mask, logprobs, actions = [], [], [], []
        generate_s = tool_s = 0.0
        termination = "budget_exhausted"
        try:
            for turn in range(4):
                if len(prompt) + len(response) + 512 > 8192:
                    termination = "context_limit"
                    break
                params = copy.deepcopy(sampling_params)
                params.update(top_k=-1, max_tokens=512)
                started = time.monotonic()
                output = await self.server_manager.generate(
                    request_id=record_id,
                    prompt_ids=prompt + response,
                    sampling_params=params,
                )
                generate_s += time.monotonic() - started
                tokens = list(output.token_ids)
                if not tokens:
                    raise RuntimeError("Empty vLLM generation; refusing an empty training response")
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                response.extend(tokens)
                mask.extend([1] * len(tokens))
                if output.log_probs is not None:
                    logprobs.extend(output.log_probs)
                # A capped, non-EOS action is not repaired or executed.
                truncated = len(tokens) >= 512 and tokens[-1] not in {
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                }
                if truncated:
                    actions.append(
                        {"turn": turn + 1, "text": text, "tokens": tokens, "truncated": True}
                    )
                    termination = "action_length"
                    break
                started = time.monotonic()
                result = await call(env.step, text)
                tool_s += time.monotonic() - started
                actions.append({"turn": turn + 1, "text": text, "tokens": tokens, "result": result})
                if env.done:
                    termination = "submitted" if env.submission else "budget_exhausted"
                    break
                context = append_observation(self.tokenizer, tokens, env.messages[-1]["content"])
                response.extend(context)
                mask.extend([0] * len(context))
                if output.log_probs is not None:
                    logprobs.extend([0.0] * len(context))
            assert response and len(response) == len(mask)
            assert len(response) <= self.config.data.max_response_length
            record = {
                "record_id": record_id,
                "task_id": task.task_id,
                "temperature": sampling_params["temperature"],
                "actions": actions,
                "prompt_ids": prompt,
                "response_ids": response,
                "response_mask": mask,
                "response_logprobs": logprobs if logprobs else None,
                "submission": env.submission,
                "termination": termination,
                "invalid_actions": env.invalid_actions,
                "execute_calls": env.execute_calls,
                "execution_errors": env.execution_errors,
                "messages": env.messages,
            }
            path = folder / (record_id + ".json")
            write_json(path, record)  # Persist the sample before any hidden evaluation.
            judge = Judge(
                os.environ["SQLFORGE_EVALUATOR"],
                os.environ.get("SQLFORGE_JUDGE_PYTHON", sys.executable),
            )
            for attempt in range(2):
                try:
                    grade = await call(judge.grade, task, env.submission)
                    break
                except JudgeError:
                    if attempt:
                        raise  # Infrastructure failure must not become reward zero.
            if grade.task_valid is False:
                raise JudgeError(f"Previously audited task became invalid: {task.task_id}")
            record["grade"] = grade.asdict()
            record["reward"] = float(grade.success)
            write_json(path, record)
            return AgentLoopOutput(
                prompt_ids=prompt,
                response_ids=response,
                response_mask=mask,
                response_logprobs=logprobs if logprobs else None,
                reward_score=float(grade.success),
                num_turns=2 * len(actions),
                metrics=AgentLoopMetrics(generate_sequences=generate_s, tool_calls=tool_s),
                extra_fields={
                    "reward_extra_info": {"record_id": record_id, "task_id": task.task_id}
                },
            )
        finally:
            await call(env.close)
            executor.shutdown(wait=True)
