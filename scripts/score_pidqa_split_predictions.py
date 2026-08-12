"""Score predictions against one named partition of a PIDQA split."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from pidbench.io import read_jsonl, write_json
from pidbench.pidqa_metrics import score_pidqa_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()

    assignments = {
        str(row["instance_id"]): str(row["split"])
        for row in read_jsonl(args.split)
    }
    records = [
        row
        for row in read_jsonl(args.records)
        if assignments.get(str(row["instance_id"])) == args.partition
    ]
    predictions = list(read_jsonl(args.predictions))
    summary = score_pidqa_predictions(records, predictions)
    summary.update(
        {
            "label": args.label,
            "partition": args.partition,
            "normalization": "PIDQA task-aware: yes/no booleans and comma/list tag sets",
            "action_counts": dict(
                sorted(Counter(str(row.get("action", "INVALID")) for row in predictions).items())
            ),
            "status_counts": dict(
                sorted(Counter(str(row.get("status", "unknown")) for row in predictions).items())
            ),
            "coverage": (
                sum(str(row.get("action", "INVALID")) == "ANSWER" for row in predictions)
                / len(records)
                if records
                else 0.0
            ),
        }
    )
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
