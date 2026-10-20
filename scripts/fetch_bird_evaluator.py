#!/usr/bin/env python3
"""Fetch a small, commit-pinned upstream evaluator with license and SHA256 manifest.

Only downloads source files. Does not download datasets, execute tests, or install anything.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "b26c4285ef69dfb6c096b076a7499018c6b370ca"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="vendor/bird-evaluator")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    root = f"https://raw.githubusercontent.com/bird-bench/BIRD-RL/{REVISION}/"
    files = {"LICENSE": "LICENSE"}
    for name in [
        "__init__.py",
        "single_instance_eval.py",
        "db_utils.py",
        "test_utils.py",
        "logger.py",
        "utils.py",
    ]:
        files[name] = "evaluation/critic/src/" + name
    hashes = {}
    for name, relative in files.items():
        with urllib.request.urlopen(root + relative, timeout=30) as response:
            content = response.read()
        (out / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (out / "UPSTREAM.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/bird-bench/BIRD-RL",
                "revision": REVISION,
                "files_sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(out.resolve())


if __name__ == "__main__":
    main()
