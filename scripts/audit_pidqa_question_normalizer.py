"""Validate a text-only semantic normalizer against released PIDQA fields."""

from __future__ import annotations

import argparse
import json

from pidbench.exposure import semantic_query_signature
from pidbench.io import read_jsonl, write_json
from pidbench.question_keys import question_semantic_signature_for_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    totals: dict[str, dict[str, object]] = {}
    examples: list[dict[str, object]] = []
    for record in read_jsonl(args.records):
        task = str(record["task"])
        entry = totals.setdefault(task, {"record_count": 0, "match_count": 0, "parse_error_count": 0})
        entry["record_count"] = int(entry["record_count"]) + 1
        try:
            parsed = question_semantic_signature_for_record(record)
        except ValueError as error:
            entry["parse_error_count"] = int(entry["parse_error_count"]) + 1
            if len(examples) < 10:
                examples.append({"instance_id": record["instance_id"], "error": str(error)})
            continue
        expected = semantic_query_signature(record)
        if parsed == expected:
            entry["match_count"] = int(entry["match_count"]) + 1
        elif len(examples) < 10:
            examples.append(
                {
                    "instance_id": record["instance_id"],
                    "parsed": parsed,
                    "released": expected,
                    "question": record["question"],
                }
            )
    record_count = sum(int(entry["record_count"]) for entry in totals.values())
    match_count = sum(int(entry["match_count"]) for entry in totals.values())
    parse_error_count = sum(int(entry["parse_error_count"]) for entry in totals.values())
    payload = {
        "method": "released-template text-only semantic normalizer",
        "scope": "PIDQA's four released task families only",
        "record_count": record_count,
        "match_count": match_count,
        "parse_error_count": parse_error_count,
        "match_rate": match_count / record_count if record_count else 0.0,
        "by_task": {
            task: {
                **entry,
                "match_rate": int(entry["match_count"]) / int(entry["record_count"]),
            }
            for task, entry in sorted(totals.items())
        },
        "examples": examples,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
