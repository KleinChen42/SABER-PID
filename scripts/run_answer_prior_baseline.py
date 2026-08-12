"""Run transparent answer-prior baselines for split-leakage diagnostics.

These are deliberately non-visual lower/reference baselines.  ``source-task``
is a diagnostic that is allowed to read source identifiers from the split; it
shows the advantage a random record split can confer when the same sheet is
represented on both sides.  It is not a P&ID understanding method.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from pidbench.io import read_jsonl, write_json, write_jsonl
from pidbench.metrics import score_predictions


def majority(rows: list[dict[str, Any]]) -> Any:
    counts = Counter(str(row["answer"]) for row in rows)
    if not counts:
        raise ValueError("Cannot form an answer prior from zero records")
    # Stable lexical tie breaking keeps the diagnostic deterministic.
    return min(counts, key=lambda answer: (-counts[answer], answer))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=("task", "source-task"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    split_by_id = {
        str(row["instance_id"]): str(row["split"])
        for row in read_jsonl(args.split)
    }
    train = [
        row for row in records if split_by_id.get(str(row["instance_id"])) == "train"
    ]
    test = [
        row for row in records if split_by_id.get(str(row["instance_id"])) == "test"
    ]
    if len(train) + len(test) == 0:
        raise ValueError("Split assignments did not match any records")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        task = str(row["task"])
        by_task[task].append(row)
        by_source_task[(str(row["source_id"]), task)].append(row)
    task_prior = {task: majority(rows) for task, rows in by_task.items()}
    source_task_prior = {
        key: majority(rows) for key, rows in by_source_task.items()
    }

    predictions = []
    source_hits = 0
    for row in test:
        task = str(row["task"])
        key = (str(row["source_id"]), task)
        if args.mode == "source-task" and key in source_task_prior:
            answer = source_task_prior[key]
            source_hits += 1
        else:
            answer = task_prior[task]
        predictions.append(
            {
                "instance_id": row["instance_id"],
                "action": "ANSWER",
                "answer": answer,
                "baseline": args.mode,
            }
        )

    write_jsonl(args.output, predictions)
    summary = score_predictions(test, predictions)
    summary.update(
        {
            "baseline": args.mode,
            "purpose": "non-visual answer-prior reference; source-task is a split-overlap diagnostic, not a P&ID method",
            "train_record_count": len(train),
            "test_record_count": len(test),
            "source_prior_hit_count": source_hits,
            "source_prior_hit_rate": source_hits / len(test) if test else 0.0,
            "task_priors": task_prior,
        }
    )
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
