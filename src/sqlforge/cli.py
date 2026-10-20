"""Dataset normalization and executable audit; no demonstration generation."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .import_data import import_bird, read_records
from .judges import Judge
from .tasks import load_tasks, write_json, write_tasks
from .workflow import audit_tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--tasks", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--evaluator", default="vendor/bird-evaluator")
    audit.add_argument("--judge-python", default=sys.executable)
    imp = sub.add_parser("import")
    imp.add_argument("--input", required=True)
    imp.add_argument("--db-root", required=True)
    imp.add_argument("--split-map", required=True, help="Versioned task ID to split map")
    imp.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Use a new output file")
    if args.command == "audit":
        report, _ = audit_tasks(
            load_tasks(args.tasks), Judge(args.evaluator, args.judge_python), check_gold=True
        )
        write_json(args.output, report)
        if report["audit_errors"] or report["rejected"]:
            raise SystemExit("Audit has rejected or indeterminate tasks; inspect the saved report")
    else:
        mapping = json.loads(Path(args.split_map).read_text())
        with tempfile.TemporaryDirectory() as folder:
            records = [
                row
                for row in read_records(Path(args.input))
                if str(row.get("instance_id")) in mapping
            ]
            selected_input = Path(folder) / Path(args.input).name
            selected_input = selected_input.with_suffix(".json")
            selected_input.write_text(json.dumps(records))
            normalized = Path(folder) / "tasks.jsonl"
            import_bird(str(selected_input), args.db_root, str(normalized), split="test")
            all_tasks = {t.task_id: t for t in load_tasks(normalized)}
            if set(mapping) - all_tasks.keys():
                raise ValueError("Source dataset is missing frozen task IDs")
            selected = []
            for tid, split in mapping.items():
                if split not in {"train", "dev", "test"}:
                    raise ValueError("Invalid split")
                task = all_tasks[tid]
                task.split = split
                selected.append(task)
            write_tasks(args.output, selected)


if __name__ == "__main__":
    main()
