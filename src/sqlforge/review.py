"""Post-hoc baseline analysis. Hidden labels never reach inference or selection."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

from .environment import parse_action
from .evaluation import compare_runs, summarize
from .tasks import file_sha256, load_tasks, write_json
from .workflow import verify_audit


def review_baselines(tasks_path, run_paths, output, judge, audit_report=None):
    """Pair fixed dev outcomes and independently grade already-finished feedback SQL."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = {task.task_id: task for task in load_tasks(tasks_path, "dev")}
    verify_audit(list(tasks.values()), audit_report, judge)
    records, manifests = {}, {}
    for name in ["single", "independent4", "feedback"]:
        path = Path(run_paths[name])
        manifest = json.loads((path / "manifest.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        if file_sha256(path / "episodes.jsonl") != summary["records_sha256"]:
            raise ValueError(f"Changed episode records: {name}")
        rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
        mapping = {row["task_id"]: row for row in rows}
        if len(rows) != len(mapping) or mapping.keys() != tasks.keys():
            raise ValueError(f"All three baselines must cover the same complete dev split: {name}")
        if manifest.get("seed") != 42 or manifest.get("temperature") != 0.7:
            raise ValueError(f"Unexpected baseline seed/temperature: {name}")
        if name == "independent4" and manifest["attempts"] != 4:
            raise ValueError("Independent baseline requires exactly four attempts")
        for tid, row in mapping.items():
            if row["fingerprint"] != tasks[tid].fingerprint():
                raise ValueError(f"Task changed: {tid}")
            if row["judge_identity"] != judge.identity():
                raise ValueError("Evaluator identity changed")
        records[name], manifests[name] = mapping, manifest
    for key in ["model", "adapter", "seed", "temperature"]:
        if len({json.dumps(m[key], sort_keys=True) for m in manifests.values()}) != 1:
            raise ValueError(f"Baselines must share {key}")
    paired, transitions = [], Counter()
    diagnostic_calls, diagnostic_started = 0, time.monotonic()
    with (output / "feedback-step-diagnostics.jsonl").open("w") as diagnostics:
        for tid in records["feedback"]:
            rows = {name: mapping[tid] for name, mapping in records.items()}
            passed = {name: bool(row["grade"]["success"]) for name, row in rows.items()}
            if passed["feedback"] and not passed["single"]:
                relative = "feedback_pass_single_fail"
            elif passed["single"] and not passed["feedback"]:
                relative = "single_pass_feedback_fail"
            else:
                relative = "both_pass" if passed["single"] else "both_fail"
            transitions[relative] += 1
            transitions["all_three_fail"] += int(not any(passed.values()))
            trajectory = rows["feedback"]["trajectories"][rows["feedback"]["selected_attempt"]]
            intermediate = []
            for index, step in enumerate(trajectory["steps"]):
                if step["generation"]["truncated"]:
                    continue
                try:
                    action = parse_action(step["generation"]["text"])
                except (ValueError, TypeError):
                    continue
                if action["name"] != "execute_sql":
                    continue
                grade = judge.grade(tasks[tid], [action["arguments"]["sql"]])
                diagnostic_calls += 1
                intermediate.append(bool(grade.success))
                diagnostics.write(
                    json.dumps(
                        {
                            "task_id": tid,
                            "step": index + 1,
                            "grade": grade.asdict(),
                            "sql": action["arguments"]["sql"],
                        }
                    )
                    + "\n"
                )
                diagnostics.flush()
            corrected = bool(intermediate and not intermediate[0] and passed["feedback"])
            lost_correct = bool(any(intermediate) and not passed["feedback"])
            transitions["feedback_first_execute_wrong_final_correct"] += int(corrected)
            transitions["feedback_had_correct_execute_final_failed"] += int(lost_correct)
            paired.append(
                {
                    "task_id": tid,
                    "db_id": rows["feedback"]["db_id"],
                    **{f"{name}_success": value for name, value in passed.items()},
                    "relative_to_single": relative,
                    "all_three_fail": not any(passed.values()),
                    "feedback_intermediate_success": intermediate,
                    "feedback_corrected_first_execute": corrected,
                    "feedback_lost_correct_execute": lost_correct,
                    **{
                        f"{name}_tokens": row["total_generated_tokens"]
                        for name, row in rows.items()
                    },
                    **{f"{name}_elapsed_s": row["elapsed_s"] for name, row in rows.items()},
                }
            )
    summary = {
        "n": len(paired),
        "transitions": dict(transitions),
        "baselines": {name: summarize(list(mapping.values())) for name, mapping in records.items()},
        "paired_differences": {
            "independent4_minus_single": compare_runs(
                str(Path(run_paths["single"]) / "episodes.jsonl"),
                str(Path(run_paths["independent4"]) / "episodes.jsonl"),
            ),
            "feedback_minus_single": compare_runs(
                str(Path(run_paths["single"]) / "episodes.jsonl"),
                str(Path(run_paths["feedback"]) / "episodes.jsonl"),
            ),
            "feedback_minus_independent4": compare_runs(
                str(Path(run_paths["independent4"]) / "episodes.jsonl"),
                str(Path(run_paths["feedback"]) / "episodes.jsonl"),
            ),
        },
        "diagnostic_judge_calls": diagnostic_calls,
        "diagnostic_analysis_elapsed_s": time.monotonic() - diagnostic_started,
        "inputs_sha256": {
            name: file_sha256(Path(path) / "episodes.jsonl") for name, path in run_paths.items()
        },
        "interpretation": [
            "Cross-condition wins/losses are paired stochastic outcomes, not causal correction labels.",
            "Independent4 reports public-selector accuracy, not oracle pass@4.",
            "Intermediate SQL is judged post hoc on a fresh DB; labels never enter policy/selector.",
            "Standalone intermediate query success need not imply correct exploratory intent.",
            "Diagnostic judge calls are analysis costs, excluded from inference/selector costs.",
        ],
    }
    write_json(output / "summary.json", summary)
    with (output / "paired-tasks.jsonl").open("w") as f:
        for row in paired:
            f.write(json.dumps(row) + "\n")
    with (output / "paired-tasks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    lines = [
        "# 三组开发集对照",
        "",
        f"共同开发集：{len(paired)} 题。",
        "",
        "| 模式 | 严格正确率 | 平均 token | 平均 SQL 执行（含选择） | 平均秒/题 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, s in summary["baselines"].items():
        tokens = s["mean_generated_tokens_all_attempts"]
        lines.append(
            f"| {name} | {s['success']['mean']:.1%} | {tokens} | "
            f"{s['mean_execute_calls_including_selector']:.2f} | {s['mean_elapsed_s']:.2f} |"
        )
    lines += [
        "",
        "逐题结果见 paired-tasks.csv，轨迹内检查见 feedback-step-diagnostics.jsonl。",
        "",
        "跨模式成败差异受随机采样和提示差异影响，不能直接当作反馈的因果效果。",
        "中间 SQL 的隐藏评分只用于离线审查，没有进入模型观察或公开选择器。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary
