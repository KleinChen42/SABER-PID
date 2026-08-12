"""Export paraphrased same-source query duplicates for the split-audit appendix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pidbench.exposure import semantic_query_signature
from pidbench.io import read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-per-task", type=int, default=5)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    split_by_id = {str(row["instance_id"]): str(row["split"]) for row in read_jsonl(args.split)}
    train_by_key: dict[tuple[str, tuple[str, tuple[tuple[str, str], ...]]], list[dict]] = defaultdict(list)
    for row in records:
        if split_by_id[str(row["instance_id"])] == "train":
            train_by_key[(str(row["source_id"]), semantic_query_signature(row))].append(row)

    selected: list[dict[str, str]] = []
    selected_by_task: dict[str, int] = defaultdict(int)
    for test in records:
        if split_by_id[str(test["instance_id"])] != "test":
            continue
        task = str(test["task"])
        if selected_by_task[task] >= args.max_per_task:
            continue
        matches = train_by_key.get((str(test["source_id"]), semantic_query_signature(test)), [])
        training = next((row for row in matches if row["question"] != test["question"]), None)
        if training is None:
            continue
        selected.append(
            {
                "task": task,
                "source_id": str(test["source_id"]),
                "semantic_fields": json.dumps(test["fields"], sort_keys=True),
                "train_instance_id": str(training["instance_id"]),
                "train_question": str(training["question"]),
                "train_answer": str(training["answer"]),
                "test_instance_id": str(test["instance_id"]),
                "test_question": str(test["question"]),
                "test_answer": str(test["answer"]),
            }
        )
        selected_by_task[task] += 1

    target = Path(args.csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]) if selected else [])
        if selected:
            writer.writeheader()
            writer.writerows(selected)
    summary = {
        "example_count": len(selected),
        "max_per_task": args.max_per_task,
        "selected_by_task": dict(sorted(selected_by_task.items())),
        "description": "Paraphrased same-source semantic query duplicates; exported for split-audit illustration only.",
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
