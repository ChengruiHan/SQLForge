"""Launch pinned native verl objectives; SQLForge supplies only the environment."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from sqlforge.judges import Judge
from sqlforge.tasks import file_sha256, load_tasks, write_json
from sqlforge.workflow import verify_audit


def patch_upstream(source):
    # v0.6.1 accepts per-call sampling parameters but supplies max_tokens again.
    # Honor our per-action limit without touching policy losses or advantages.
    path = source / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    old = "max_tokens = self.config.max_model_len - len(prompt_ids)"
    new = 'max_tokens = min(sampling_params.pop("max_tokens", 512), self.config.max_model_len - len(prompt_ids))'
    text = path.read_text()
    if old in text:
        assert text.count(old) == 1
        path.write_text(text.replace(old, new))
    else:
        assert new in text, "Unexpected upstream generation implementation"
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "change": "Honor per-action max_tokens=512",
    }


def overrides(root, method, smoke):
    data = json.loads((root / "DATA.json").read_text())
    folder = root / ("smoke-" + method if smoke else method)
    folder.mkdir(exist_ok=True)
    cfg = {
        "data.train_files": str(root / ("smoke.parquet" if smoke else "train.parquet")),
        "data.val_files": str(root / "dev32.parquet"),
        "data.train_batch_size": 4,
        "data.val_batch_size": 8,
        "data.max_prompt_length": data["max_prompt_length"],
        "data.max_response_length": data["max_response_length"],
        "data.shuffle": False,
        "data.seed": 42,
        "data.dataloader_num_workers": 0,
        "data.return_raw_chat": True,
        "data.truncation": "error",
        "data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": data["model"],
        "actor_rollout_ref.model.lora_rank": 16,
        "actor_rollout_ref.model.lora_alpha": 32,
        "actor_rollout_ref.model.target_modules": "all-linear",
        "actor_rollout_ref.model.lora_adapter_path": str(root / "initial-adapter"),
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.external_lib": "sqlforge.verl_padding",
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.optim.lr": 1e-5,
        "actor_rollout_ref.actor.optim.weight_decay": 0.0,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.ppo_epochs": 1 if method == "rloo" else 2,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.policy_loss.loss_mode": {
            "grpo": "vanilla",
            "gspo": "gspo",
            "rloo": "gpg",
        }[method],
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum"
        if method == "rloo"
        else "seq-mean-token-mean",
        "actor_rollout_ref.actor.clip_ratio_low": 0.0003 if method == "gspo" else 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.0004 if method == "gspo" else 0.2,
        "actor_rollout_ref.actor.use_kl_loss": method != "rloo",
        "actor_rollout_ref.actor.kl_loss_coef": 0.02,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.use_torch_compile": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.max_model_len": 8192,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.40,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.seed": 42,
        "actor_rollout_ref.rollout.layered_summon": True,
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "sqlforge_h0",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(root / "agent.yaml"),
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.7,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "algorithm.adv_estimator": "rloo" if method == "rloo" else "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": method == "rloo",
        "algorithm.kl_penalty": "kl",
        "algorithm.kl_ctrl.kl_coef": 0.02,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 1 if smoke else 64,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 1,
        "trainer.project_name": "SQLForge-H0-verl",
        "trainer.experiment_name": folder.name,
        "trainer.logger": ["console"],
        "trainer.val_before_train": False,
        "trainer.test_freq": -1 if smoke else 16,
        "trainer.save_freq": 1 if smoke else 16,
        "trainer.max_actor_ckpt_to_keep": 1,
        "trainer.resume_mode": "auto",
        "trainer.default_local_dir": str(folder / "checkpoints"),
        "trainer.rollout_data_dir": str(folder / "rollout-text"),
        "trainer.validation_data_dir": str(folder / "validation-text"),
        "ray_kwargs.ray_init.num_cpus": 8,
    }
    return folder, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--method", choices=["grpo", "gspo", "rloo"], required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compose-only", action="store_true")
    p.add_argument("--evaluator", default="vendor/bird-evaluator")
    p.add_argument("--judge-python", default=sys.executable)
    p.add_argument("--verl-source", default="vendor/verl-v0.6.1")
    args = p.parse_args()
    root = Path(args.run).resolve()
    frozen_data = json.loads((root / "DATA.json").read_text())
    assert file_sha256(root / "tasks.jsonl") == frozen_data["task_file_sha256"]
    initial = json.loads((root / "INITIALIZATION.json").read_text())
    assert (
        file_sha256(root / "initial-adapter/adapter_model.safetensors") == initial["adapter_sha256"]
    )
    verify_audit(
        load_tasks(str(root / "tasks.jsonl")),
        str(root / "audit.json"),
        Judge(args.evaluator, args.judge_python),
    )
    source = Path(args.verl_source).resolve()
    patch = patch_upstream(source)
    (root / "agent.yaml").write_text(
        "- name: sqlforge_h0\n  _target_: sqlforge.verl_agent.SQLForgeAgentLoop\n"
    )
    folder, cfg = overrides(root, args.method, args.smoke)
    args_list = ["++" + k + "=" + json.dumps(v, ensure_ascii=False) for k, v in cfg.items()]
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(source / "verl/trainer/config"), version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=args_list)
    (folder / "resolved.yaml").write_text(OmegaConf.to_yaml(composed, resolve=True))
    write_json(
        folder / "launch.json",
        {"method": args.method, "smoke": args.smoke, "overrides": cfg, "upstream_patch": patch},
    )
    release = json.loads((root / "MODEL.json").read_text())
    for path, digest in release["base_sha256"].items():
        assert file_sha256(path) == digest, f"Original base changed: {path}"
    tracked = [
        Path("src/sqlforge") / name
        for name in [
            "verl_agent.py",
            "verl_padding.py",
            "environment.py",
            "judges.py",
            "database.py",
        ]
    ]
    tracked += [
        source / "verl/trainer/ppo/core_algos.py",
        source / "verl/workers/actor/dp_actor.py",
    ]
    write_json(
        folder / "provenance.json",
        {
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "peft", "vllm", "ray", "tensordict"]
            },
            "files_sha256": {str(p): file_sha256(p) for p in tracked},
            "base_sha256": release["base_sha256"],
            "initial_adapter_sha256": file_sha256(
                root / "initial-adapter/adapter_model.safetensors"
            ),
            "task_file_sha256": file_sha256(root / "tasks.jsonl"),
        },
    )
    if args.compose_only:
        return
    env = dict(
        os.environ,
        PYTHONPATH=str(Path("src").resolve()) + ":" + str(source),
        SQLFORGE_EVALUATOR=str(Path(args.evaluator).resolve()),
        SQLFORGE_JUDGE_PYTHON=str(Path(args.judge_python).absolute()),
        SQLFORGE_VERL_TASKS=str(root / "tasks.jsonl"),
        SQLFORGE_VERL_TRAJECTORIES=str(folder / "trajectories"),
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        WANDB_MODE="disabled",
        VLLM_USE_V1="1",
        RAY_DEDUP_LOGS="1",
    )
    try:
        subprocess.run(
            [sys.executable, "-m", "verl.trainer.main_ppo", *args_list], env=env, check=True
        )
    finally:
        # This entry point owns the isolated local Ray cluster. Release workers
        # before the next arm initializes another inference engine on this GPU.
        subprocess.run(
            [sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


if __name__ == "__main__":
    main()
