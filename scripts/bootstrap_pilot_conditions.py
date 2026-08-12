"""Source-cluster bootstrap comparisons for paired P&ID pilot conditions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from pidbench.io import read_jsonl, write_json
from pidbench.pidqa_metrics import normalize_pidqa_answer


def correctness(records: list[dict], prediction_path: str) -> dict[str, int]:
    prediction_by_id = {
        str(row["instance_id"]): row for row in read_jsonl(prediction_path)
    }
    result: dict[str, int] = {}
    for record in records:
        instance_id = str(record["instance_id"])
        prediction = prediction_by_id.get(instance_id, {})
        result[instance_id] = int(
            str(prediction.get("action", "INVALID")) == "ANSWER"
            and normalize_pidqa_answer(prediction.get("answer"), str(record["task"]))
            == normalize_pidqa_answer(record.get("answer"), str(record["task"]))
        )
    return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="May be repeated.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    if not args.condition:
        raise ValueError("At least one --condition is required")
    records = list(read_jsonl(args.records))
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_id"])].append(record)
    sources = sorted(by_source)
    baseline_correct = correctness(records, args.baseline)

    rows: list[dict[str, object]] = []
    for item in args.condition:
        if "=" not in item:
            raise ValueError(f"Invalid --condition {item!r}; use LABEL=PATH")
        label, prediction_path = item.split("=", 1)
        condition_correct = correctness(records, prediction_path)
        baseline_accuracy = sum(baseline_correct.values()) / len(records)
        condition_accuracy = sum(condition_correct.values()) / len(records)
        rng = random.Random(args.seed)
        differences: list[float] = []
        for _ in range(args.bootstrap_reps):
            sampled = [rng.choice(sources) for _ in sources]
            total = sum(len(by_source[source]) for source in sampled)
            difference = sum(
                condition_correct[str(record["instance_id"])]
                - baseline_correct[str(record["instance_id"])]
                for source in sampled
                for record in by_source[source]
            ) / total
            differences.append(difference)
        rows.append(
            {
                "baseline": args.baseline_label,
                "condition": label,
                "source_count": len(sources),
                "record_count": len(records),
                "baseline_accuracy": baseline_accuracy,
                "condition_accuracy": condition_accuracy,
                "difference_condition_minus_baseline": condition_accuracy - baseline_accuracy,
                "bootstrap_ci95_low": quantile(differences, 0.025),
                "bootstrap_ci95_high": quantile(differences, 0.975),
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed,
            }
        )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "paired source-cluster bootstrap",
        "unit": "source sheet",
        "baseline": args.baseline_label,
        "rows": rows,
    }
    write_json(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
