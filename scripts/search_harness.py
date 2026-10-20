"""Bounded dev-only harness search. Test evaluation is a separate explicit command."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from sqlforge.environment import EnvConfig, SQLRepairEnv
from sqlforge.eval_cli import NoInference, model_identity
from sqlforge.evaluation import evaluate
from sqlforge.harness import HarnessConfig, harness_manifest, load_harness, render_messages
from sqlforge.harness_search import partition_development, select_candidate, validate_proposal
from sqlforge.model import ModelConfig
from sqlforge.proposer import ProposalClient
from sqlforge.sampling import collect, read
from sqlforge.tasks import file_sha256, load_tasks, write_json
from sqlforge.vllm_policy import VLLMPolicy
from sqlforge.workflow import verify_audit


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def task_score(path):
    data = lines(Path(path) / "episodes.jsonl")
    summary = read(Path(path) / "summary.json")
    if file_sha256(Path(path) / "episodes.jsonl") != summary["records_sha256"]:
        raise ValueError("Cached records changed")
    return {
        "n": len(data),
        "passed": sum(int(r["grade"]["success"]) for r in data),
        "tokens": sum(r["total_generated_tokens"] for r in data),
    }


class Experiment:
    def __init__(self, args):
        from sqlforge.judges import Judge

        self.root = Path(args.run).resolve()
        self.code = Path(__file__).resolve().parents[1]
        self.dev = load_tasks(args.tasks, "dev")
        self.model = ModelConfig(model=str(Path(args.model).resolve()))
        self.arms = read(args.arms)
        if (
            set(self.arms) != {"raw", "grpo", "gspo", "rloo"}
            or self.arms["raw"]["adapter"] is not None
        ):
            raise ValueError("Expected raw and three frozen RL adapters")
        self.judge = Judge(args.evaluator, args.judge_python)
        self.audit = args.audit
        self.proposer_config = Path(args.proposer)
        self.root.mkdir(parents=True, exist_ok=True)
        self.futures = {}
        self.pool = ThreadPoolExecutor(max_workers=2)

    def prepare(self):
        split = partition_development(self.dev)
        verify_audit(self.dev, self.audit, self.judge)
        weights = {
            k: file_sha256(Path(v["adapter"]) / "adapter_model.safetensors")
            if v["adapter"]
            else None
            for k, v in self.arms.items()
        }
        self.static = [
            load_harness(self.code / "configs/harness" / (name + ".json"))
            for name in ["H0", "prompt_candidate", "feedback_candidate", "history_candidate"]
        ]
        plan = {
            "protocol": "bounded-harness-search-v1",
            "partition": split,
            "weights": weights,
            "base_sha256": model_identity(self.model.model),
            "adapter_config_sha256": {
                k: file_sha256(Path(v["adapter"]) / "adapter_config.json") if v["adapter"] else None
                for k, v in self.arms.items()
            },
            "task_fingerprints": {t.task_id: t.fingerprint() for t in self.dev},
            "static": [asdict(c) for c in self.static],
            "proposer_config_sha256": file_sha256(self.proposer_config),
            "code_sha256": {
                str(p.relative_to(self.code)): file_sha256(p)
                for directory in ["src/sqlforge", "scripts"]
                for p in (self.code / directory).glob("*.py")
            },
            "max_candidates": 8,
            "max_proposals": 4,
            "search_seed": 42,
            "selection_seeds": [42, 43],
            "representative": "grpo",
            "test_used": False,
        }
        target = self.root / "PLAN.json"
        if target.exists() and read(target) != plan:
            raise ValueError("Plan changed: use a new run directory")
        write_json(target, plan)
        self.plan = plan
        mapping = {t.task_id: t for t in self.dev}
        self.search = [mapping[i] for i in split["search"]]
        self.selection = [mapping[i] for i in split["selection"]]

    def status(self, stage, **extra):
        write_json(
            self.root / "status.json",
            {
                "status": "running",
                "stage": stage,
                "pid": os.getpid(),
                "updated": time.time(),
                **extra,
            },
        )

    def environment(self, config):
        return EnvConfig(
            max_turns=4,
            row_limit=5,
            observation_token_limit=384,
            observation_tokenizer=self.model.model,
            prompt_version="one_action_v2",
            harness=config,
        )

    def start_engine(self):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.model, local_files_only=True)
        h0 = self.environment(self.static[0])
        over_budget = []
        for task in self.dev:
            env = SQLRepairEnv(task, h0)
            try:
                assert render_messages(env.messages, h0) == render_messages(
                    env.messages, replace(h0, harness=None)
                )
                prompt = self.tokenizer.apply_chat_template(
                    env.messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
                if len(prompt) + 512 > 8192:
                    over_budget.append(task.task_id)
            finally:
                env.close()
        write_json(
            self.root / "PREFLIGHT.json",
            {
                "dev_tasks": 128,
                "actual_Qwen_tokenizer_checked": True,
                "initial_over_budget_tasks_retained": over_budget,
            },
        )
        settings = {
            "model": self.model.model,
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "enable_lora": True,
            "max_lora_rank": 16,
            "max_loras": 1,
            "max_cpu_loras": 4,
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "seed": 42,
            "generation_config": "vllm",
            "disable_log_stats": True,
        }
        write_json(self.root / "ENGINE.json", settings)
        self.engine = LLM(**settings)
        self.eos = read(Path(self.model.model) / "generation_config.json")["eos_token_id"]
        self.eos = [self.eos] if isinstance(self.eos, int) else self.eos

    def policy(self, arm, seed):
        from vllm.lora.request import LoRARequest

        index = ["grpo", "gspo", "rloo"].index(arm) + 1 if arm != "raw" else 0
        adapter = self.arms[arm]["adapter"]
        request = LoRARequest(arm, index, adapter) if adapter else None
        return VLLMPolicy(
            self.engine,
            self.tokenizer,
            replace(self.model, seed=seed),
            lora_request=request,
            eos_ids=self.eos,
            temperature=0.7,
        )

    def condition(self, phase, arm, config, seed, tasks, *, cache=None):
        destination = self.root / phase / arm / config.name / ("seed" + str(seed))
        destination.mkdir(parents=True, exist_ok=True)
        env = self.environment(config)
        manifest = {
            "phase": phase,
            "arm": arm,
            "seed": seed,
            "temperature": 0.7,
            "task_ids": [t.task_id for t in tasks],
            "fingerprints": [t.fingerprint() for t in tasks],
            "harness": harness_manifest(config),
            "adapter_sha256": self.plan["weights"][arm],
            "judge_identity": self.judge.identity(),
            "plan_sha256": file_sha256(self.root / "PLAN.json"),
            "base_model": self.model.model,
            "sampling_backend": "vllm",
            "historical_generation_reused": cache is not None,
        }
        marker = destination / "manifest.json"
        if marker.exists() and read(marker) != manifest:
            raise ValueError("Condition configuration changed")
        write_json(marker, manifest)
        output = destination / "graded"
        if (output / "summary.json").exists():
            task_score(output)
            return output
        if cache is None:
            self.status(phase, arm=arm, candidate=config.name, seed=seed)
            samples = destination / "samples"
            collect(tasks, self.policy(arm, seed), env, samples, 8)
            cache = {}
            for task in tasks:
                saved = read(samples / (task.task_id + ".json"))
                cache[task.task_id] = {
                    "task_id": task.task_id,
                    "fingerprint": saved["fingerprint"],
                    "trajectories": [saved["trajectory"]],
                    "sampling_elapsed_s": saved["amortized_sampling_s"],
                }
        if set(cache) != {t.task_id for t in tasks}:
            raise ValueError("Scoring cache must cover every frozen task")
        if output.exists():
            output.rename(output.with_name("graded-interrupted-" + str(time.time_ns())))
        metadata = {
            "arm": arm,
            "seed": seed,
            "temperature": 0.7,
            "harness": harness_manifest(config),
            "plan_sha256": file_sha256(self.root / "PLAN.json"),
            "condition_manifest_sha256": file_sha256(marker),
            "history_source": manifest["historical_generation_reused"],
        }
        future = self.pool.submit(
            evaluate,
            tasks,
            NoInference(),
            self.judge,
            str(output),
            method="feedback",
            attempts=1,
            max_turns=4,
            metadata=metadata,
            audit_report=self.audit,
            cached_records=cache,
            env_config=env,
        )
        self.futures[str(output)] = future
        return output

    def wait_scores(self, paths):
        result = {}
        for name, path in paths.items():
            if str(path) in self.futures:
                self.futures[str(path)].result()
            result[name] = task_score(path)
        return result

    def proposal(self, slot, configs, paths, scores):
        directory = self.root / "proposals" / ("slot" + str(slot))
        directory.mkdir(parents=True, exist_ok=True)
        outcome = directory / "outcome.json"
        if outcome.exists():
            x = read(outcome)
            return HarnessConfig.from_dict(x["candidate"]) if x["status"] == "accepted" else None
        if (directory / "request.json").exists():
            write_json(
                outcome,
                {
                    "status": "indeterminate_call_consumed",
                    "reason": "Interrupted after reservation; do not repeat API call",
                },
            )
            return None
        parents = sorted(
            scores, key=lambda name: (-scores[name]["passed"], scores[name]["tokens"], name)
        )[:2]
        instructions = (
            "Optimize a generic SQL debugging harness around a frozen Qwen 4B agent. Return one JSON object "
            "with exactly rationale (string) and candidate (object). Candidate schema: schema_version=1; "
            "prompt.additional_instruction string<=2048 characters; feedback.style raw/status_first, row_preview 0..5; "
            "history.mode full/compact_observations, keep_recent_observations 1..3, include_state boolean. "
            "Omitted modules use H0 defaults. Four actions, context8192/action512, observation384, SQL timeout3s are fixed. "
            "Use public execution evidence, not test or reference answers. Do not include task IDs, database names, "
            "specific table/column names, SQL solutions or per-instance rules. Do not force tool use or a fixed number "
            "of steps. Improve semantic repair and useful feedback handling. Execution validity is not correctness. "
            "Propose a nonduplicate candidate, optionally combining strengths of parents."
        )
        context = {
            "parents": [
                {"candidate": asdict(configs[name]), "score": scores[name]} for name in parents[:1]
            ],
            "all_search_scores": scores,
            "examples": [],
        }
        public_examples = []
        for name in parents:
            data = lines(paths[name] / "episodes.jsonl")
            for row in sorted(data, key=lambda r: (r["grade"]["success"], r["task_id"])):
                tr = row["trajectories"][0]
                question = (
                    json.loads(tr["steps"][0]["messages"][1]["content"])["question"]
                    if tr["steps"]
                    else ""
                )
                public_examples.append(
                    {
                        "parent": name,
                        "success": row["grade"]["success"],
                        "question": question[:450],
                        "termination": tr["termination"],
                        "steps": [
                            {
                                "action_preview": s["generation"]["text"][:450],
                                "public_observation_preview": json.dumps(
                                    s["observation"], ensure_ascii=False
                                )[:600],
                            }
                            for s in tr["steps"]
                        ][-2:],
                    }
                )

        def messages():
            return [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]

        # UTF-8 byte bound is deliberately conservative relative to tokenizer counts.
        for example in public_examples:
            context["examples"].append(example)
            if len(json.dumps(messages(), ensure_ascii=False).encode()) + 1024 > 8192:
                context["examples"].pop()
                continue
            if len(context["examples"]) >= 4:
                break
        request = messages()
        bound = len(json.dumps(request, ensure_ascii=False).encode()) + 1024
        if bound > 8192:
            write_json(outcome, {"status": "invalid_context_consumed", "input_bound": bound})
            return None
        write_json(
            directory / "request.json",
            {"slot": slot, "messages": request, "input_byte_bound": bound},
        )
        try:
            config = read(self.proposer_config)
            config.update(max_calls=4, max_tokens=4096)
            key = os.environ.get("SQLFORGE_API_KEY", "")
            teacher = ProposalClient(config, directory, key)
            teacher.calls = slot - 4
            generation = teacher.generate(request)
            if generation.truncated:
                raise ValueError("Proposal was truncated; slot consumed")
            proposed = json.loads(generation.text)
            candidate = validate_proposal(
                proposed,
                name="adaptive" + str(slot),
                existing=list(configs.values()),
                database_names=sorted({t.db_id for t in self.dev}),
            )
            write_json(
                outcome,
                {
                    "status": "accepted",
                    "candidate": asdict(candidate),
                    "rationale": proposed["rationale"],
                },
            )
            return candidate
        except Exception as exc:
            # Do not store raw HTTP headers, credentials or environment values.
            write_json(
                outcome,
                {
                    "status": "invalid_or_failed_slot_consumed",
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:500]
                    if isinstance(exc, (ValueError, json.JSONDecodeError))
                    else "API/infrastructure failure; no automatic retry",
                },
            )
            return None

    def run(self):
        self.prepare()
        self.start_engine()
        configs = {c.name: c for c in self.static}
        paths = {
            c.name: self.condition("dev-search", "grpo", c, 42, self.search) for c in self.static
        }
        scores = self.wait_scores(paths)
        for slot in range(4, 8):
            candidate = self.proposal(slot, configs, paths, scores)
            if candidate:
                configs[candidate.name] = candidate
                paths[candidate.name] = self.condition(
                    "dev-search", "grpo", candidate, 42, self.search
                )
                scores.update(self.wait_scores({candidate.name: paths[candidate.name]}))
        write_json(
            self.root / "SEARCH_REPORT.json",
            {"scores": scores, "candidates": {k: asdict(v) for k, v in configs.items()}},
        )
        top = sorted(
            (k for k in configs if k != "H0"),
            key=lambda k: (-scores[k]["passed"], scores[k]["tokens"], k),
        )[:2]
        selected_scores = {name: {} for name in ["H0", *top]}
        for name in selected_scores:
            for seed in [42, 43]:
                path = self.condition("dev-selection", "grpo", configs[name], seed, self.selection)
                selected_scores[name][seed] = self.wait_scores({name: path})[name]
        decision = select_candidate(selected_scores)
        write_json(self.root / "SELECTION.json", decision)
        chosen = configs[decision["selected"]]
        transfer = {}
        for name in ["H0"] if chosen.name == "H0" else ["H0", chosen.name]:
            for seed in [42, 43]:
                path = self.condition("dev-transfer", "raw", configs[name], seed, self.selection)
                transfer[f"{name}/seed{seed}"] = self.wait_scores({name: path})[name]
        write_json(self.root / "selected-harness.json", asdict(chosen))
        write_json(
            self.root / "FINAL_LOCK.json",
            {
                "selected": chosen.name,
                "harness": harness_manifest(chosen),
                "weights": self.plan["weights"],
                "plan_sha256": file_sha256(self.root / "PLAN.json"),
                "selection_sha256": file_sha256(self.root / "SELECTION.json"),
                "test_used": False,
            },
        )
        write_json(
            self.root / "FINAL_REPORT.json",
            {
                "status": "completed",
                "selected": chosen.name,
                "search": scores,
                "selection": decision,
                "raw_transfer": transfer,
                "test_used": False,
            },
        )
        self.pool.shutdown(wait=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["run", "tasks", "model", "arms", "audit"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--evaluator", default="vendor/bird-evaluator")
    p.add_argument("--judge-python", default=sys.executable)
    p.add_argument("--proposer", default="configs/proposer.json")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    experiment = Experiment(args)
    import fcntl

    with (experiment.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.prepare_only:
                experiment.prepare()
            else:
                experiment.run()
        finally:
            experiment.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
