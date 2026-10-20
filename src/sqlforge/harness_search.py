"""Frozen development partitioning and selection for bounded harness search."""

from __future__ import annotations

import difflib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict

from .harness import HarnessConfig


def normalized_question(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def partition_development(tasks, seed=20261020):
    """Keep near-duplicate question components together and halve every stratum."""
    if len(tasks) != 128 or any(t.split != "dev" for t in tasks):
        raise ValueError("Expected exactly the frozen 128 development tasks")
    if len({t.task_id for t in tasks}) != 128:
        raise ValueError("Duplicate development task IDs")
    parent = list(range(len(tasks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    texts = [normalized_question(t.question) for t in tasks]
    edges = []
    for i in range(len(tasks)):
        for j in range(i):
            matcher = difflib.SequenceMatcher(None, texts[i], texts[j], autojunk=False)
            if matcher.real_quick_ratio() < 0.92 or matcher.quick_ratio() < 0.92:
                continue
            ratio = matcher.ratio()
            if ratio >= 0.92:
                parent[find(i)] = find(j)
                edges.append([tasks[j].task_id, tasks[i].task_id, ratio])
    components = {}
    for i in range(len(tasks)):
        components.setdefault(find(i), []).append(i)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    strata = sorted({(t.db_id, t.category) for t in tasks})
    counts = Counter((t.db_id, t.category) for t in tasks)
    if any(counts[s] % 2 for s in strata):
        raise ValueError("Cannot halve odd strata without changing the frozen rule")
    target = tuple(counts[s] // 2 for s in strata)
    states = {tuple(0 for _ in strata): ()}
    for gi, group in enumerate(groups):
        vector = tuple(sum((tasks[i].db_id, tasks[i].category) == s for i in group) for s in strata)
        additions = {}
        for current, chosen in states.items():
            nxt = tuple(a + b for a, b in zip(current, vector))
            if all(a <= b for a, b in zip(nxt, target)) and nxt not in states:
                additions.setdefault(nxt, chosen + (gi,))
        states.update(additions)
    if target not in states:
        raise ValueError("Duplicate groups prevent the declared balanced 64/64 split")
    chosen = {i for gi in states[target] for i in groups[gi]}
    search = [t.task_id for i, t in enumerate(tasks) if i in chosen]
    selection = [t.task_id for i, t in enumerate(tasks) if i not in chosen]
    assert len(search) == len(selection) == 64 and not set(search) & set(selection)
    return {
        "seed": seed,
        "search": search,
        "selection": selection,
        "strata": [
            {"db": s[0], "category": s[1], "search": n, "selection": n}
            for s, n in zip(strata, target)
        ],
        "grouping": "connected components; normalized question SequenceMatcher ratio >=0.92",
        "near_duplicate_edges": edges,
        "groups": [[tasks[i].task_id for i in g] for g in groups if len(g) > 1],
    }


def candidate_content(config):
    value = asdict(config)
    value.pop("name")
    return json.dumps(value, sort_keys=True)


def validate_proposal(value, *, name, existing, database_names):
    if not isinstance(value, dict) or set(value) != {"rationale", "candidate"}:
        raise ValueError("Expected exactly rationale and candidate")
    if not isinstance(value["rationale"], str) or len(value["rationale"]) > 3000:
        raise ValueError("Invalid rationale")
    candidate = dict(value["candidate"])
    candidate["name"] = name
    config = HarnessConfig.from_dict(candidate)
    text = config.prompt.additional_instruction
    if re.search(r"\b(?:TRAIN|DEV|TEST)_\d+\b", text, re.I):
        raise ValueError("Task identifiers are not valid generic harness guidance")
    if any(re.search(r"\b" + re.escape(db) + r"\b", text, re.I) for db in database_names):
        raise ValueError("Database-specific guidance is outside the search space")
    if candidate_content(config) in {candidate_content(c) for c in existing}:
        raise ValueError("Duplicate candidate; proposal slot consumed without reroll")
    return config


def select_candidate(scores):
    """Mean must improve and neither of the two seeds may regress versus H0."""
    if "H0" not in scores:
        raise ValueError("H0 control is required")
    for name, seeds in scores.items():
        if set(seeds) != {42, 43}:
            raise ValueError("Both fixed sampling seeds are required")
        for row in seeds.values():
            if (
                row["n"] != 64
                or type(row["passed"]) is not int
                or not 0 <= row["passed"] <= 64
                or not math.isfinite(row["tokens"])
                or row["tokens"] < 0
            ):
                raise ValueError("Incomplete or invalid selection condition")
    baseline = scores["H0"]

    def total(seeds):
        return sum(row["passed"] for row in seeds.values())

    eligible = [
        name
        for name, seeds in scores.items()
        if name != "H0"
        and total(seeds) > total(baseline)
        and all(seeds[s]["passed"] >= baseline[s]["passed"] for s in [42, 43])
    ]
    chosen = (
        min(
            eligible,
            key=lambda name: (
                -total(scores[name]),
                sum(r["tokens"] for r in scores[name].values()),
                name,
            ),
        )
        if eligible
        else "H0"
    )
    return {
        "selected": chosen,
        "eligible": sorted(eligible),
        "rule": "mean success improves; neither seed below H0; tie by generated tokens then name",
        "scores": scores,
    }
