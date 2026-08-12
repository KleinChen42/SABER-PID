"""Deterministic source-aware and intentionally random comparison splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable


SPLITS = ("train", "calibration", "test")


def _partition_keys(keys: Iterable[str], seed: int) -> dict[str, str]:
    unique_keys = sorted(set(keys))
    if len(unique_keys) < 3:
        raise ValueError("At least three distinct keys are required for a three-way split")
    rng = random.Random(seed)
    rng.shuffle(unique_keys)
    total = len(unique_keys)
    train_end = round(total * 0.60)
    calibration_end = train_end + round(total * 0.20)
    allocation: dict[str, str] = {}
    for position, key in enumerate(unique_keys):
        allocation[key] = (
            "train"
            if position < train_end
            else "calibration"
            if position < calibration_end
            else "test"
        )
    return allocation


def make_source_split(records: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows = list(records)
    assignment = _partition_keys((str(row["source_id"]) for row in rows), seed)
    return [
        {
            "instance_id": str(row["instance_id"]),
            "source_id": str(row["source_id"]),
            "split": assignment[str(row["source_id"])],
        }
        for row in rows
    ]


def make_random_split(records: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows = list(records)
    assignment = _partition_keys((str(row["instance_id"]) for row in rows), seed)
    return [
        {
            "instance_id": str(row["instance_id"]),
            "source_id": str(row["source_id"]),
            "split": assignment[str(row["instance_id"])],
        }
        for row in rows
    ]


def summarize_split(assignments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(assignments)
    by_split = Counter(str(row["split"]) for row in rows)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_splits[str(row["source_id"])].add(str(row["split"]))
    leakage_sources = sorted(
        source for source, splits in source_splits.items() if len(splits) > 1
    )
    return {
        "record_counts": {name: by_split.get(name, 0) for name in SPLITS},
        "source_count": len(source_splits),
        "sources_across_multiple_splits": len(leakage_sources),
        "source_examples_across_multiple_splits": leakage_sources[:10],
    }


def assert_source_isolated(assignments: Iterable[dict[str, Any]]) -> None:
    summary = summarize_split(assignments)
    if summary["sources_across_multiple_splits"]:
        examples = summary["source_examples_across_multiple_splits"]
        raise ValueError(f"Source leakage in split; examples: {examples}")
