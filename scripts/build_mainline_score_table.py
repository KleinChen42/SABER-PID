"""Build a compact, machine-readable table from PIDQA score JSON files.

The script intentionally consumes only scorer outputs.  It does not inspect
hidden answers and therefore cannot change a prediction or introduce a
condition-dependent selection rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_items(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not label or not path:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        parsed.append((label, Path(path)))
    return parsed


def load_score(label: str, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload.get("task_accuracy", {})
    count = int(payload.get("prediction_count", 0) or 0)
    total = float(payload.get("total_generation_seconds", 0.0) or 0.0)
    row: dict[str, object] = {
        "label": label,
        "score_file": str(path),
        "record_count": int(payload.get("record_count", 0) or 0),
        "prediction_count": count,
        "coverage": float(payload.get("coverage", 0.0) or 0.0),
        "overall_accuracy": float(payload.get("overall_accuracy", 0.0) or 0.0),
        "source_macro_accuracy": float(payload.get("source_macro_accuracy", 0.0) or 0.0),
        "connectivity_accuracy": float(task.get("connectivity", 0.0) or 0.0),
        "count_accuracy": float(task.get("count", 0.0) or 0.0),
        "spatial_count_accuracy": float(task.get("spatial_count", 0.0) or 0.0),
        "value_accuracy": float(task.get("value", 0.0) or 0.0),
        "mean_latency_seconds": float(payload.get("mean_latency_seconds", 0.0) or 0.0),
        "median_latency_seconds": float(payload.get("median_latency_seconds", 0.0) or 0.0),
        "p95_latency_seconds": float(payload.get("p95_latency_seconds", 0.0) or 0.0),
        "total_generation_seconds": total,
        "requests_per_second": (count / total) if count and total > 0 else 0.0,
        "status_counts": payload.get("status_counts", {}),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = [load_score(label, path) for label, path in parse_items(args.score)]
    json_path = Path(args.json)
    csv_path = Path(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    columns = [
        key for key, value in rows[0].items() if not isinstance(value, (dict, list))
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
