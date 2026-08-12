"""Create an answer-containing scoring set for an answer-isolated public input."""

from __future__ import annotations

import argparse
import json

from pidbench.io import read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="Full normalized records with answers")
    parser.add_argument("--public-input", required=True, help="Answer-isolated input records")
    parser.add_argument("--output", required=True, help="Private local scoring subset")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    requested_ids = {
        str(row["instance_id"])
        for row in read_jsonl(args.public_input)
    }
    all_records = {
        str(row["instance_id"]): row
        for row in read_jsonl(args.records)
    }
    missing = sorted(requested_ids - set(all_records))
    if missing:
        raise KeyError("Missing public input IDs in normalized records: " + ", ".join(missing[:10]))
    selected = [all_records[instance_id] for instance_id in sorted(requested_ids)]
    write_jsonl(args.output, selected)
    summary = {
        "record_count": len(selected),
        "source_count": len({str(row["source_id"]) for row in selected}),
        "tasks": sorted({str(row["task"]) for row in selected}),
        "contains_answers": all("answer" in row for row in selected),
        "public_input": args.public_input,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
