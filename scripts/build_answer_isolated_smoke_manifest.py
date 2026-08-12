#!/usr/bin/env python3
"""Build a deterministic, answer-isolated smoke manifest from a public JSONL input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="connectivity,count,spatial_count,value",
        help="Comma-separated task order; the first record of each is retained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not requested:
        raise SystemExit("--tasks must contain at least one task")

    selected: dict[str, dict] = {}
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task = str(record.get("task", ""))
            if task in requested and task not in selected:
                selected[task] = record
            if len(selected) == len(requested):
                break

    missing = [task for task in requested if task not in selected]
    if missing:
        raise SystemExit(f"missing requested task(s): {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for task in requested:
            handle.write(json.dumps(selected[task], sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    print(f"wrote {len(requested)} answer-isolated smoke records to {args.output}")


if __name__ == "__main__":
    main()
