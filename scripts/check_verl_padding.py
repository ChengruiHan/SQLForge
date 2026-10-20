"""Compare native versus trimmed active-token logprobs and parameter gradients."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from sqlforge.tasks import write_json
from sqlforge.verl_padding import native_forward, trim_edges


def check(device, dtype):
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
    copied = copy.deepcopy(model)
    ids = torch.tensor([[0, 0, 0, 11, 12, 13, 14, 21, 22, 23, 24, 0, 0]], device=device)
    attention = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], device=device)
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    batch = {
        "input_ids": ids,
        "attention_mask": attention,
        "position_ids": positions,
        "responses": ids[:, 7:],
    }
    policy_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]], device=device)
    results = []
    for actor_model, forward in [(model, native_forward), (copied, trim_edges(native_forward))]:
        actor = SimpleNamespace(
            actor_module=actor_model,
            use_remove_padding=False,
            use_fused_kernels=False,
            device_name=device,
            param_dtype=dtype,
            config=SimpleNamespace(entropy_checkpointing=False),
        )
        _, logs = forward(actor, batch, 1.0)
        (-logs * policy_mask).sum().backward()
        results.append(logs.detach())
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-5
    difference = ((results[0] - results[1]) * policy_mask).abs().max().item()
    assert difference <= tolerance, difference
    max_gradient_difference = 0.0
    original_grads, trimmed_grads = [], []
    for a, b in zip(model.parameters(), copied.parameters()):
        if a.grad is not None:
            original_grads.append(a.grad.float().flatten())
            trimmed_grads.append(b.grad.float().flatten())
            delta = (a.grad - b.grad).abs().max().item()
            max_gradient_difference = max(max_gradient_difference, delta)
            assert torch.allclose(a.grad.float(), b.grad.float(), atol=tolerance, rtol=tolerance), (
                delta
            )
    original_grads = torch.cat(original_grads)
    trimmed_grads = torch.cat(trimmed_grads)
    relative_gradient_error = (
        (original_grads - trimmed_grads).norm() / original_grads.norm()
    ).item()
    cosine = torch.nn.functional.cosine_similarity(original_grads, trimmed_grads, dim=0).item()
    assert relative_gradient_error < (0.03 if dtype == torch.bfloat16 else 1e-5)
    assert cosine > 0.999
    return {
        "active_logprob_max_abs_difference": difference,
        "parameter_gradient_max_abs_difference": max_gradient_difference,
        "relative_gradient_l2_error": relative_gradient_error,
        "gradient_cosine_similarity": cosine,
        "tolerance": tolerance,
        "passed": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    result = {"cpu_fp32": check("cpu", torch.float32)}
    # Run this check before the full pipeline acquires the GPU.
    if torch.cuda.is_available():
        result["cuda_bf16"] = check("cuda", torch.bfloat16)
    write_json(Path(args.run) / "PADDING_EQUIVALENCE.json", result)


if __name__ == "__main__":
    main()
