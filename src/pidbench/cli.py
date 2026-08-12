"""Command-line entry points for the lightweight PIDQA pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json, write_jsonl
from .metrics import score_predictions
from .pidqa import load_pidqa, normalize_pidqa, summarize_pidqa
from .splits import (
    assert_source_isolated,
    make_random_split,
    make_source_split,
    summarize_split,
)


def _records(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def command_normalize(args: argparse.Namespace) -> int:
    summary = normalize_pidqa(args.raw, args.output)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    summary = summarize_pidqa(_records(args.records))
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_split(args: argparse.Namespace) -> int:
    records = _records(args.records)
    assignments = (
        make_source_split(records, args.seed)
        if args.mode == "source"
        else make_random_split(records, args.seed)
    )
    if args.mode == "source":
        assert_source_isolated(assignments)
    write_jsonl(args.output, assignments)
    summary = summarize_split(assignments)
    summary.update({"mode": args.mode, "seed": args.seed})
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_export_pilot(args: argparse.Namespace) -> int:
    records = _records(args.records)
    assignment_by_id = {
        str(row["instance_id"]): row for row in read_jsonl(args.split)
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        assignment = assignment_by_id.get(str(record["instance_id"]))
        if assignment and assignment["split"] == args.partition:
            grouped[str(record["source_id"])].append(record)
    selected: list[dict[str, Any]] = []
    for source_id in sorted(grouped)[: args.max_sources]:
        source_rows = grouped[source_id]
        source_selected: list[dict[str, Any]] = []
        # PIDQA CSV files are grouped by task. Select one example of every
        # available task first so a small pilot is not accidentally count-only.
        for task in sorted({str(row["task"]) for row in source_rows}):
            source_selected.append(next(row for row in source_rows if row["task"] == task))
        for record in source_rows:
            if len(source_selected) >= args.per_source:
                break
            if record not in source_selected:
                source_selected.append(record)
        selected.extend(source_selected[: args.per_source])
    public_rows = [
        {
            key: value
            for key, value in record.items()
            if key not in {"answer", "cypher"}
        }
        for record in selected
    ]
    write_jsonl(args.output, public_rows)
    summary = {
        "partition": args.partition,
        "source_count": len({row["source_id"] for row in selected}),
        "record_count": len(selected),
        "tasks": sorted({row["task"] for row in selected}),
        "answer_isolated": True,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_score(args: argparse.Namespace) -> int:
    summary = score_predictions(_records(args.records), read_jsonl(args.predictions))
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pidbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize-pidqa")
    normalize.add_argument("--raw", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--summary", required=True)
    normalize.set_defaults(func=command_normalize)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--records", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)

    split = subparsers.add_parser("split")
    split.add_argument("--records", required=True)
    split.add_argument("--mode", choices=("source", "random"), required=True)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--output", required=True)
    split.add_argument("--summary", required=True)
    split.set_defaults(func=command_split)

    export = subparsers.add_parser("export-pilot")
    export.add_argument("--records", required=True)
    export.add_argument("--split", required=True)
    export.add_argument("--partition", choices=("train", "calibration", "test"), default="train")
    export.add_argument("--max-sources", type=int, default=12)
    export.add_argument("--per-source", type=int, default=6)
    export.add_argument("--output", required=True)
    export.add_argument("--summary", required=True)
    export.set_defaults(func=command_export_pilot)

    score = subparsers.add_parser("score")
    score.add_argument("--records", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(func=command_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
