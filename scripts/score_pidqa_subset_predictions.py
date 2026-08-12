"""Score a task-aware PIDQA prediction file against a private subset."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from pidbench.io import read_jsonl, write_json
from pidbench.pidqa_metrics import score_pidqa_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    predictions = list(read_jsonl(args.predictions))
    score = score_pidqa_predictions(records, predictions)
    latencies = [
        float(row["latency_seconds"])
        for row in predictions
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    actions = Counter(str(row.get("action", "INVALID")) for row in predictions)
    statuses = Counter(str(row.get("status", "unknown")) for row in predictions)
    score.update(
        {
            "label": args.label,
            "coverage": actions.get("ANSWER", 0) / len(records) if records else 0.0,
            "action_counts": dict(sorted(actions.items())),
            "status_counts": dict(sorted(statuses.items())),
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "p95_latency_seconds": sorted(latencies)[round((len(latencies) - 1) * 0.95)] if latencies else None,
            "total_generation_seconds": sum(latencies),
            "normalization": "PIDQA task-aware: yes/no booleans and comma/list tag sets",
        }
    )
    write_json(args.output, score)
    print(json.dumps(score, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
