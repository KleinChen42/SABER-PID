"""Score a returned answer-isolated VLM pilot against its local hidden truth."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from pidbench.io import read_jsonl, write_json
from pidbench.metrics import score_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="Local hidden truth subset")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    predictions = list(read_jsonl(args.predictions))
    score = score_predictions(records, predictions)
    latencies = [
        float(row["latency_seconds"])
        for row in predictions
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    actions = Counter(str(row.get("action", "INVALID")) for row in predictions)
    statuses = Counter(str(row.get("status", "unknown")) for row in predictions)
    compact_score = {
        key: value
        for key, value in score.items()
        if key != "source_accuracy"
    }
    compact_score.update(
        {
            "label": args.label,
            "status_counts": dict(sorted(statuses.items())),
            "action_counts": dict(sorted(actions.items())),
            "valid_answer_rate": actions.get("ANSWER", 0) / len(predictions) if predictions else 0.0,
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "total_generation_seconds": sum(latencies),
        }
    )
    write_json(args.output, compact_score)
    print(json.dumps(compact_score, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
