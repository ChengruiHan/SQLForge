"""Trim masked edge padding for single-trajectory SDPA microbatches.

This optional verl external_lib wrapper delegates all probability computation to
the native actor, then restores its output shape. No objective is replaced.
"""

from functools import wraps

import torch.nn.functional as F
from verl.workers.actor.dp_actor import DataParallelPPOActor


def trim_edges(native):
    @wraps(native)
    def forward(self, batch, temperature, calculate_entropy=False):
        if self.use_remove_padding or batch["input_ids"].shape[0] != 1:
            return native(self, batch, temperature, calculate_entropy)
        mask = batch["attention_mask"][0]
        active = mask.nonzero(as_tuple=False).flatten()
        if not len(active):
            raise ValueError("Empty trajectory attention mask")
        left, end = int(active[0]), int(active[-1]) + 1
        original_response = batch["responses"].shape[-1]
        prompt_length = mask.shape[-1] - original_response
        response_length = end - prompt_length
        if response_length < 1 or left >= prompt_length:
            raise ValueError("Trajectory must include a prompt and a response")
        if left == 0 and end == mask.shape[-1]:
            return native(self, batch, temperature, calculate_entropy)
        trimmed = dict(batch)
        for key in ("input_ids", "attention_mask", "position_ids"):
            trimmed[key] = batch[key][..., left:end]
        trimmed["responses"] = batch["responses"][..., :response_length]
        entropy, log_probs = native(self, trimmed, temperature, calculate_entropy)
        right = original_response - response_length
        return (
            F.pad(entropy, (0, right)) if entropy is not None else None,
            F.pad(log_probs, (0, right)),
        )

    return forward


if not getattr(DataParallelPPOActor._forward_micro_batch, "_sqlforge_trim_edges", False):
    native_forward = DataParallelPPOActor._forward_micro_batch
    DataParallelPPOActor._forward_micro_batch = trim_edges(native_forward)
    DataParallelPPOActor._forward_micro_batch._sqlforge_trim_edges = True
