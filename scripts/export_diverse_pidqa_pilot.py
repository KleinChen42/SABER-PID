"""Select a compact per-source PIDQA pilot without first-row answer bias."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from pidbench.io import read_jsonl, write_json, write_jsonl


def stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--partition", choices=("train", "calibration", "test"), default="test")
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    assignment_by_id = {
        str(row["instance_id"]): row
        for row in read_jsonl(args.split)
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(args.records):
        assignment = assignment_by_id.get(str(record["instance_id"]))
        if assignment and assignment["split"] == args.partition:
            grouped[(str(record["source_id"]), str(record["task"]))].append(record)
    sources = sorted({source for source, _ in grouped})[: args.max_sources]
    tasks = sorted({task for _, task in grouped})
    selected: list[dict[str, Any]] = []
    answer_counts: dict[str, Counter[str]] = {task: Counter() for task in tasks}
    for task in tasks:
        for source in sorted(sources, key=lambda source: stable_rank(args.seed, task, source)):
            candidates = grouped[(source, task)]
            if not candidates:
                continue
            # Favor a currently rare answer class, then use a stable seed-tie.
            selected_row = min(
                candidates,
                key=lambda row: (
                    answer_counts[task][str(row["answer"])],
                    stable_rank(args.seed, task, source, str(row["instance_id"])),
                ),
            )
            answer_counts[task][str(selected_row["answer"])] += 1
            selected.append(selected_row)
    selected.sort(key=lambda row: (str(row["source_id"]), str(row["task"])))
    public_rows = [
        {key: value for key, value in row.items() if key not in {"answer", "cypher"}}
        for row in selected
    ]
    write_jsonl(args.output, public_rows)
    summary: dict[str, Any] = {
        "partition": args.partition,
        "seed": args.seed,
        "record_count": len(public_rows),
        "source_count": len({str(row["source_id"]) for row in selected}),
        "tasks": tasks,
        "answer_isolated": True,
        "answer_diversity": {
            task: {answer: count for answer, count in sorted(counts.items())}
            for task, counts in sorted(answer_counts.items())
        },
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
