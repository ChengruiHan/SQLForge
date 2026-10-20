"""Freeze common task order and validate inputs before native verl training."""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlforge.environment import EnvConfig, SQLRepairEnv
from sqlforge.judges import Judge
from sqlforge.tasks import file_sha256, load_tasks, write_json, write_tasks
from sqlforge.workflow import audit_tasks, verify_audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument(
        "--model", required=True, help="Local unquantized Qwen3-4B-Instruct-2507 directory"
    )
    p.add_argument(
        "--task-order", help="Historical results/task-order.json, to reproduce task selection"
    )
    p.add_argument("--evaluator", default="vendor/bird-evaluator")
    p.add_argument("--judge-python", default=sys.executable)
    args = p.parse_args()
    root = Path(args.run).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pool = load_tasks(args.tasks, "train")
    dev = load_tasks(args.tasks, "dev")
    order = list(range(len(pool)))
    random.Random(42).shuffle(order)
    candidates = [pool[i] for i in reversed(order)]
    if args.task_order:
        ids = json.loads(Path(args.task_order).read_text())["train_task_order"]
        mapping = {t.task_id: t for t in pool}
        candidates = [mapping[i] for i in ids]
    assert len(candidates) >= 256 and len(dev) == 128
    judge = Judge(args.evaluator, args.judge_python)
    dev_report, accepted = audit_tasks(dev, judge, check_gold=True)
    assert not dev_report["audit_errors"] and len(accepted) == 128
    cache = root / "audit-train"
    cache.mkdir(exist_ok=True)

    def check(task):
        path = cache / (task.task_id + ".json")
        for candidate in (path,):
            if candidate.exists():
                try:
                    verify_audit([task], str(candidate), judge)
                    report = json.loads(candidate.read_text())
                    write_json(path, report)
                    return report
                except ValueError:
                    pass
        result, accepted = audit_tasks([task], judge, check_gold=True)
        write_json(path, result)
        return result

    ordered, reports, excluded = [], [], []
    for start in range(0, len(candidates), 32):
        block = candidates[start : start + 32]
        with ThreadPoolExecutor(max_workers=4) as executor:
            checked = list(executor.map(check, block))
        for task, result in zip(block, checked):
            if not result["validated"]:
                # Retry alone to distinguish CPU contention from a reference
                # incompatible with the unchanged three-second SQL budget.
                result, _ = audit_tasks([task], judge, check_gold=True)
                write_json(cache / (task.task_id + ".json"), result)
            assert not result["audit_errors"], f"Indeterminate audit infrastructure: {task.task_id}"
            if result["validated"]:
                ordered.append(task)
                reports.append(result)
            else:
                excluded.append(result)
            if len(ordered) == 256:
                break
        if len(ordered) == 256:
            break
    assert len(ordered) == 256
    assert set(t.db_id for t in ordered).isdisjoint(t.db_id for t in dev)
    write_tasks(root / "tasks.jsonl", ordered + dev)
    write_json(root / "pre-freeze-input-exclusions.json", excluded)
    report, _ = audit_tasks(ordered + dev, judge, check_gold=False)
    report.update(
        gold_checked=True,
        validated=384,
        rejected=[],
        audit_errors=[],
        checked_tasks=[r for x in reports for r in x["checked_tasks"]]
        + dev_report["checked_tasks"],
    )
    write_json(root / "audit.json", report)
    verify_audit(ordered + dev, str(root / "audit.json"), judge)
    import pandas as pd
    from transformers import AutoTokenizer

    model = str(Path(args.model).resolve())
    base_files = sorted(
        p
        for p in Path(model).iterdir()
        if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}
    )
    if not any(p.suffix == ".safetensors" for p in base_files):
        raise ValueError("Download original model safetensors before preparing")
    write_json(root / "MODEL.json", {"base_sha256": {str(p): file_sha256(p) for p in base_files}})
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    max_prompt = 0

    def row(task, index):
        nonlocal max_prompt
        env = SQLRepairEnv(
            task, EnvConfig(max_turns=4, row_limit=5, prompt_version="one_action_v2")
        )
        try:
            messages = env.messages
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            assert len(ids) + 512 <= 8192, f"Initial prompt too long: {task.task_id}"
            max_prompt = max(max_prompt, len(ids))
            return dict(
                prompt=messages,
                agent_name="sqlforge_h0",
                data_source="sqlforge_h0",
                ability="sql_repair",
                reward_model={"style": "rule", "ground_truth": ""},
                extra_info={"task_id": task.task_id, "index": index},
            )
        finally:
            env.close()

    # Fixed database-balanced dev32, selected before viewing any new model results.
    selected_dev = []
    for db in sorted({t.db_id for t in dev}):
        candidates = [t for t in dev if t.db_id == db]
        random.Random(42).shuffle(candidates)
        selected_dev.extend(candidates[:16])
    train_rows = [row(t, i) for i, t in enumerate(ordered)]
    dev_rows = [row(t, i) for i, t in enumerate(selected_dev)]
    pd.DataFrame(train_rows).to_parquet(root / "train.parquet", index=False)
    pd.DataFrame(train_rows[:4]).to_parquet(root / "smoke.parquet", index=False)
    pd.DataFrame(dev_rows).to_parquet(root / "dev32.parquet", index=False)
    write_json(
        root / "DATA.json",
        {
            "model": model,
            "train_task_order": [t.task_id for t in ordered],
            "dev32": [t.task_id for t in selected_dev],
            "max_prompt_length": ((max_prompt + 63) // 64) * 64,
            "max_response_length": 3328,
            "max_model_len": 8192,
            "task_file_sha256": file_sha256(root / "tasks.jsonl"),
            "audit_sha256": file_sha256(root / "audit.json"),
        },
    )
    # One immutable, zero-output adapter shared by all three arms. This is
    # random LoRA initialization, with no SFT and no training examples consumed.
    if not (root / "initial-adapter/adapter_model.safetensors").exists():
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        torch.manual_seed(42)
        base = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
            attn_implementation="sdpa",
        )
        adapted = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules="all-linear",
                bias="none",
            ),
        )
        assert all(
            torch.count_nonzero(p) == 0 for n, p in adapted.named_parameters() if "lora_B" in n
        )
        adapted.save_pretrained(root / "initial-adapter", safe_serialization=True)
    write_json(
        root / "INITIALIZATION.json",
        {
            "adapter_sha256": file_sha256(root / "initial-adapter/adapter_model.safetensors"),
            "seed": 42,
            "lora_B": "all zero",
            "sft": False,
        },
    )


if __name__ == "__main__":
    main()
