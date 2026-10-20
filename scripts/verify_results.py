"""Recompute reported scores and paired gains from the exported task outcomes."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    summary = json.loads((root / "rl-summary.json").read_text())
    rows = [json.loads(s) for s in (root / "test-outcomes.jsonl").read_text().splitlines()]
    by_model = {}
    for row in rows:
        model = by_model.setdefault(row["model"], {})
        assert row["task_id"] not in model
        model[row["task_id"]] = row
    raw = by_model["raw"]
    for name, records in by_model.items():
        assert set(records) == set(raw) and len(records) == 128
        expected = summary["results"]["test"][name]
        assert sum(r["success"] for r in records.values()) == expected["passed"]
        wins = sum(r["success"] and not raw[k]["success"] for k, r in records.items())
        losses = sum(not r["success"] and raw[k]["success"] for k, r in records.items())
        print(name, expected["passed"], "/128", "wins", wins, "losses", losses)
    assert sum(r["success"] for r in by_model["rloo"].values()) == 42
    splits = json.loads((root / "task-splits.json").read_text())
    assert {s: list(splits.values()).count(s) for s in ["train", "dev", "test"]} == {
        "train": 256,
        "dev": 128,
        "test": 128,
    }
    harness = json.loads((root / "harness-summary.json").read_text())
    assert harness["selected"] == "H0" and harness["new_test_generations"] == 0


if __name__ == "__main__":
    main()
