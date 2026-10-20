"""Frozen-model evaluation and explicit test locks, independent of SFT artifacts."""

import argparse
import sys
from pathlib import Path

from .environment import EnvConfig
from .evaluation import evaluate
from .harness import harness_manifest, load_harness
from .judges import Judge
from .model import ModelConfig
from .sampling import collect, read
from .tasks import file_sha256, load_tasks, write_json
from .vllm_policy import VLLMPolicy
from .workflow import verify_audit


def select_dev_tasks(tasks, ids, limit):
    if (
        not isinstance(ids, list)
        or not all(isinstance(tid, str) for tid in ids)
        or len(ids) != limit
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Task IDs must be a unique JSON string list matching --limit")
    mapping = {task.task_id: task for task in tasks}
    if len(mapping) != len(tasks) or any(tid not in mapping for tid in ids):
        raise ValueError("Subset contains unknown or duplicate task IDs")
    selected = [mapping[tid] for tid in ids]
    if any(task.split != "dev" for task in selected):
        raise ValueError("Subset must contain development tasks only")
    return selected


def verify_test_lock(arm_specs, arms, lock, *, harness=None):
    if lock.get("checkpoint_rule") != "final256-no-test-selection":
        raise ValueError("Invalid final checkpoint lock")
    expected = harness["identity_sha256"] if harness else None
    if lock.get("harness_sha256") != expected:
        raise ValueError("Test harness differs from frozen final lock")
    for arm in arms:
        adapter = arm_specs[arm]["adapter"]
        digest = file_sha256(Path(adapter) / "adapter_model.safetensors") if adapter else None
        if arm not in lock["adapter_sha256"] or lock["adapter_sha256"][arm] != digest:
            raise ValueError("Test adapter differs from frozen final lock")


class NoInference:
    def generate(self, messages):
        raise RuntimeError("Scoring must not regenerate cached answers")


def model_identity(model):
    root = Path(model).resolve()
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Expected a local downloaded base model")
    return {p.name: file_sha256(p) for p in files}


def freeze(args):
    arms = read(args.arms)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Use a new lock file")
    harness = harness_manifest(load_harness(args.harness)) if args.harness else None
    write_json(
        target,
        {
            "checkpoint_rule": "final256-no-test-selection",
            "adapter_sha256": {
                name: file_sha256(Path(v["adapter"]) / "adapter_model.safetensors")
                if v["adapter"]
                else None
                for name, v in arms.items()
            },
            "adapter_config_sha256": {
                name: file_sha256(Path(v["adapter"]) / "adapter_config.json")
                if v["adapter"]
                else None
                for name, v in arms.items()
            },
            "base_sha256": model_identity(args.model),
            "tasks_sha256": file_sha256(args.tasks),
            "harness_sha256": harness["identity_sha256"] if harness else None,
            "temperature": 0.7,
            "seed": 42,
            "protocol": "feedback",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--arms", required=True, help="JSON mapping names to {adapter: directory or null}"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--audit")
    p.add_argument("--evaluator", default="vendor/bird-evaluator")
    p.add_argument("--judge-python", default=sys.executable)
    p.add_argument("--harness")
    p.add_argument("--lock", help="Required frozen identities for test")
    p.add_argument("--freeze", action="store_true", help="Write identities only; no generation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["single", "independent4", "feedback"], default="feedback")
    args = p.parse_args()
    if args.freeze:
        freeze(args)
        return
    arms = read(args.arms)
    config = ModelConfig(model=str(Path(args.model).resolve()), seed=args.seed)
    hc = load_harness(args.harness) if args.harness else None
    hm = harness_manifest(hc)
    if hc and args.mode != "feedback":
        raise ValueError("Modular harness is a feedback-only protocol")
    tasks = load_tasks(args.tasks, args.split)
    if len(tasks) != 128:
        raise ValueError("Formal evaluation requires the full 128-task split")
    base_hashes = model_identity(config.model)
    if args.split == "test":
        if not args.lock or args.mode != "feedback" or args.seed != 42:
            raise ValueError("Test requires frozen feedback/seed42 identities")
        lock = read(args.lock)
        verify_test_lock(arms, list(arms), lock, harness=hm)
        if lock["base_sha256"] != base_hashes or lock["tasks_sha256"] != file_sha256(args.tasks):
            raise ValueError("Locked base model or task manifest changed")
        for name, v in arms.items():
            sha = file_sha256(Path(v["adapter"]) / "adapter_config.json") if v["adapter"] else None
            if lock["adapter_config_sha256"][name] != sha:
                raise ValueError("Adapter configuration changed")
    judge = Judge(args.evaluator, args.judge_python)
    verify_audit(tasks, args.audit, judge)
    out = Path(args.output).resolve()
    # An interrupted run keeps its cached samples. Identity must match on resume.
    marker = {
        "arms": arms,
        "adapter_sha256": {
            k: file_sha256(Path(v["adapter"]) / "adapter_model.safetensors")
            if v["adapter"]
            else None
            for k, v in arms.items()
        },
        "base_sha256": base_hashes,
        "tasks_sha256": file_sha256(args.tasks),
        "seed": args.seed,
        "mode": args.mode,
        "split": args.split,
        "harness": hm,
        "judge": judge.identity(),
    }
    if (out / "manifest.json").exists() and read(out / "manifest.json") != marker:
        raise ValueError("Resume identities changed")
    write_json(out / "manifest.json", marker)
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
    engine = LLM(
        model=config.model,
        dtype="bfloat16",
        max_model_len=8192,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_cpu_loras=4,
        gpu_memory_utilization=0.85,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        seed=42,
        generation_config="vllm",
        disable_log_stats=True,
    )
    eos = read(Path(config.model) / "generation_config.json")["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    attempts = 4 if args.mode == "independent4" else 1
    env = EnvConfig(
        max_turns=4 if args.mode == "feedback" else 1,
        submit_only=args.mode != "feedback",
        row_limit=5,
        observation_token_limit=384,
        observation_tokenizer=config.model,
        prompt_version="one_action_v2",
        harness=hc,
    )
    for index, (name, spec) in enumerate(arms.items(), 1):
        result = out / name / "graded"
        if (result / "summary.json").exists():
            if (
                file_sha256(result / "episodes.jsonl")
                != read(result / "summary.json")["records_sha256"]
            ):
                raise ValueError("Cached scores changed")
            continue
        request = LoRARequest(name, index, spec["adapter"]) if spec["adapter"] else None
        policy = VLLMPolicy(
            engine, tokenizer, config, lora_request=request, temperature=0.7, eos_ids=eos
        )
        cache = {
            t.task_id: {
                "fingerprint": t.fingerprint(),
                "trajectories": [],
                "sampling_elapsed_s": 0.0,
            }
            for t in tasks
        }
        for attempt in range(attempts):
            policy.seed_offset = attempt * 1000
            samples = out / name / "samples" / f"attempt-{attempt}"
            collect(tasks, policy, env, samples, 8)
            for t in tasks:
                item = read(samples / f"{t.task_id}.json")
                cache[t.task_id]["trajectories"].append(item["trajectory"])
                cache[t.task_id]["sampling_elapsed_s"] += item["amortized_sampling_s"]
        if result.exists():
            import time

            result.rename(result.with_name("interrupted-" + str(time.time_ns())))
        evaluate(
            tasks,
            NoInference(),
            judge,
            str(result),
            method="independent" if attempts == 4 else args.mode,
            attempts=attempts,
            max_turns=4,
            audit_report=args.audit,
            cached_records=cache,
            env_config=env,
            metadata=marker,
        )


if __name__ == "__main__":
    main()
