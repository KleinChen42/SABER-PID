"""Freeze a deterministic, answer-isolated stratified public subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = ("connectivity", "count", "spatial_count", "value")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-task", type=int, default=12)
    parser.add_argument("--seed-label", default="evidence-smoke-v1")
    args = parser.parse_args()
    if args.per_task <= 0:
        raise ValueError("--per-task must be positive")
    records = read_jsonl(Path(args.input))
    if any("answer" in row or "cypher" in row for row in records):
        raise ValueError("Input must be answer-isolated public records.")
    selected: list[dict[str, Any]] = []
    for task in TASKS:
        rows = [row for row in records if str(row.get("task")) == task]
        rows.sort(key=lambda row: hashlib.sha256(f"{args.seed_label}|{row['instance_id']}".encode("utf-8")).hexdigest())
        if len(rows) < args.per_task:
            raise ValueError(f"Task {task!r} has only {len(rows)} rows; requested {args.per_task}.")
        selected.extend(rows[: args.per_task])
    selected.sort(key=lambda row: (TASKS.index(str(row["task"])), str(row["source_id"]), str(row["instance_id"])))
    output = Path(args.output)
    write_jsonl(output, selected)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "status": "pass",
        "selection_rule": "within-task SHA-256(seed_label|instance_id) ascending",
        "seed_label": args.seed_label,
        "input": Path(args.input).as_posix(),
        "output": output.as_posix(),
        "output_sha256": digest,
        "record_count": len(selected),
        "source_count": len({str(row["source_id"]) for row in selected}),
        "per_task": {task: sum(str(row["task"]) == task for row in selected) for task in TASKS},
        "answer_isolated": True,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
