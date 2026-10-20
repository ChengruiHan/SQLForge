"""Inference configuration; optimization is owned by verl."""

import hashlib
from dataclasses import dataclass


@dataclass
class ModelConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str | None = None
    adapter: str | None = None
    load_in_4bit: bool = False
    device: str = "cuda"
    max_context: int = 8192
    max_new_tokens: int = 512
    lora_rank: int = 16
    seed: int = 42
    base_only: bool = False


def seed_everything(seed: int):
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def student_seed(task_id, seed=42):
    return (seed + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)) % (2**31)
