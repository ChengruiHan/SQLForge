"""Persist batched generations before hidden scoring; resumed runs never resample."""

import json
import time
from pathlib import Path

from .harness import harness_manifest
from .rollout import run_episodes


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def collect(tasks, policy, env, out, batch_size):
    out.mkdir(parents=True, exist_ok=True)
    expected_harness = harness_manifest(env.harness)
    for offset in range(0, len(tasks), batch_size):
        block = tasks[offset : offset + batch_size]
        batch_path = out / f"batch-{offset:04d}.json"
        if all((out / f"{t.task_id}.json").exists() for t in block):
            for task in block:
                saved = read(out / f"{task.task_id}.json")
                if (
                    saved["fingerprint"] != task.fingerprint()
                    or saved["trajectory"].get("harness") != expected_harness
                ):
                    raise ValueError("Cached task or harness identity changed")
            continue
        if batch_path.exists():
            batch = read(batch_path)
        else:
            started = time.monotonic()
            trajectories = run_episodes(block, policy, env)
            batch = {
                "task_ids": [t.task_id for t in block],
                "fingerprints": [t.fingerprint() for t in block],
                "batch_elapsed_s": time.monotonic() - started,
                "trajectories": [t.asdict() for t in trajectories],
            }
            atomic(batch_path, batch)
        if batch["task_ids"] != [t.task_id for t in block] or batch["fingerprints"] != [
            t.fingerprint() for t in block
        ]:
            raise ValueError("Cached sampling batch differs from frozen tasks")
        if any(t.get("harness") != expected_harness for t in batch["trajectories"]):
            raise ValueError("Cached sampling batch differs from frozen harness")
        for task, trajectory in zip(block, batch["trajectories"]):
            atomic(
                out / f"{task.task_id}.json",
                {
                    "task_id": task.task_id,
                    "fingerprint": task.fingerprint(),
                    "trajectory": trajectory,
                    "amortized_sampling_s": batch["batch_elapsed_s"] / len(block),
                },
            )
        atomic(out / "progress.json", {"completed": offset + len(block), "total": len(tasks)})
