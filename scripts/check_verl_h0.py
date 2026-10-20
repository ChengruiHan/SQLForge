"""Small numerical checks of native objectives and Qwen continuation masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage, get_policy_loss_fn
from verl.workers.config import ActorConfig

from sqlforge.tasks import write_json
from sqlforge.verl_agent import append_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    tokenizer = AutoTokenizer.from_pretrained(json.loads((root / "DATA.json").read_text())["model"])
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    action = '{"name":"schema","arguments":{}}'
    observation = '{"remaining_turns":3}'
    generated = tokenizer.encode(action + "<|im_end|>", add_special_tokens=False)
    appended = append_observation(tokenizer, generated, observation)
    full = tokenizer.apply_chat_template(
        messages
        + [{"role": "assistant", "content": action}, {"role": "user", "content": observation}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert prompt + generated + appended == full, "Incremental Qwen template differs"
    mask = torch.tensor([[1.0, 0.0, 1.0]] * 4)
    rewards = torch.zeros(4, 3)
    rewards[0, -1] = 1
    adv, _ = compute_rloo_outcome_advantage(rewards, mask, np.array(["a"] * 4))
    assert torch.allclose(adv[:, 0], torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))
    results = {}
    cfg = ActorConfig(
        strategy="fsdp2",
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.0003,
        clip_ratio_high=0.0004,
        clip_ratio_c=3.0,
    )
    for method, objective in [("grpo", "vanilla"), ("gspo", "gspo"), ("rloo", "gpg")]:
        logs = torch.full((4, 3), -2.0, requires_grad=True)
        loss, metrics = get_policy_loss_fn(objective)(
            logs.detach(),
            logs,
            adv,
            mask,
            loss_agg_mode="seq-mean-token-sum" if method == "rloo" else "seq-mean-token-mean",
            config=cfg,
        )
        loss.backward()
        assert torch.isfinite(logs.grad).all() and logs.grad[:, 0].abs().sum() > 0
        assert not logs.grad[:, 1].any(), "Tool context received a policy gradient"
        results[method] = {
            "loss": float(loss.detach()),
            "gradient_finite": True,
            "context_gradient_zero": True,
        }
    write_json(
        root / "OBJECTIVE_CHECKS.json",
        {
            "qwen_incremental_template_exact": True,
            "rloo_leave_one_out_exact": True,
            "native_objectives": results,
        },
    )


if __name__ == "__main__":
    main()
