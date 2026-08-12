"""Measure same-source semantic-query exposure across deterministic split seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pidbench.exposure import build_same_source_cache_audit
from pidbench.io import read_jsonl, write_json
from pidbench.splits import make_random_split, make_source_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--seeds", default="3,17,29,43,71")
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    for mode, splitter in (("random", make_random_split), ("source", make_source_split)):
        for seed in seeds:
            _, summary = build_same_source_cache_audit(records, splitter(records, seed))
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "test_records": summary["test_records"],
                    "train_source_count": summary["train_source_count"],
                    "same_source_test_rate": summary["same_source_test_rate"],
                    "semantic_query_overlap_rate": summary["semantic_query_overlap_rate"],
                    "exact_question_overlap_rate": summary["exact_question_overlap_rate"],
                    "unambiguous_cache_hit_rate": summary["unambiguous_cache_hit_rate"],
                    "ambiguous_train_cache_keys": summary["ambiguous_train_cache_keys"],
                }
            )

    aggregate: dict[str, dict[str, float]] = {}
    metrics = (
        "same_source_test_rate",
        "semantic_query_overlap_rate",
        "exact_question_overlap_rate",
        "unambiguous_cache_hit_rate",
    )
    for mode in ("random", "source"):
        group = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = {
            f"mean_{metric}": sum(float(row[metric]) for row in group) / len(group)
            for metric in metrics
        }
        aggregate[mode].update(
            {
                f"min_{metric}": min(float(row[metric]) for row in group)
                for metric in metrics
            }
        )
        aggregate[mode].update(
            {
                f"max_{metric}": max(float(row[metric]) for row in group)
                for metric in metrics
            }
        )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "same_source_semantic_cache_audit",
        "method_role": "diagnostic exposure audit, not an input-only VLM baseline",
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }
    write_json(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
