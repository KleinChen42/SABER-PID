"""Score a PIDQA pilot with task-aware exact/list normalization and coverage."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from pidbench.io import read_jsonl, write_json
from pidbench.pidqa_metrics import normalize_pidqa_answer, score_pidqa_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    predictions = list(read_jsonl(args.predictions))
    records_by_id = {str(row["instance_id"]): row for row in records}
    actions = Counter(str(row.get("action", "INVALID")) for row in predictions)
    statuses = Counter(str(row.get("status", "unknown")) for row in predictions)
    answered = [
        row
        for row in predictions
        if str(row.get("instance_id")) in records_by_id
        and str(row.get("action", "INVALID")) == "ANSWER"
    ]
    answered_correct = sum(
        normalize_pidqa_answer(
            row.get("answer"), str(records_by_id[str(row["instance_id"])]["task"])
        )
        == normalize_pidqa_answer(
            records_by_id[str(row["instance_id"])].get("answer"),
            str(records_by_id[str(row["instance_id"])]["task"]),
        )
        for row in answered
    )
    latencies = [
        float(row["latency_seconds"])
        for row in predictions
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    raw_score = score_pidqa_predictions(records, predictions)
    summary = {
        key: value for key, value in raw_score.items() if key != "source_accuracy"
    }
    summary.update(
        {
            "label": args.label,
            "normalization": "PIDQA task-aware: yes/no booleans and comma/list tag sets",
            "action_counts": dict(sorted(actions.items())),
            "status_counts": dict(sorted(statuses.items())),
            "coverage": len(answered) / len(records) if records else 0.0,
            "answered_count": len(answered),
            "answered_correct_count": answered_correct,
            "answered_accuracy": answered_correct / len(answered) if answered else None,
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "total_generation_seconds": sum(latencies),
        }
    )
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
