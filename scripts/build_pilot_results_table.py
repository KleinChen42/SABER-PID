"""Build compact manuscript-ready tables from PIDQA pilot score JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    rows = []
    for score_path in args.scores:
        payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
        task_accuracy = payload.get("task_accuracy", {})
        rows.append(
            {
                "label": payload["label"],
                "overall_accuracy": payload["overall_accuracy"],
                "coverage": payload.get("coverage"),
                "answered_accuracy": payload.get("answered_accuracy"),
                "connectivity_accuracy": task_accuracy.get("connectivity"),
                "count_accuracy": task_accuracy.get("count"),
                "spatial_count_accuracy": task_accuracy.get("spatial_count"),
                "value_accuracy": task_accuracy.get("value"),
                "mean_latency_seconds": payload.get("mean_latency_seconds"),
            }
        )
    fieldnames = list(rows[0]) if rows else []
    csv_target = Path(args.csv)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_target = Path(args.markdown)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    header = ["Run", "Overall", "Coverage", "Answered acc.", "Conn.", "Count", "Spatial", "Value", "Mean s"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(row["overall_accuracy"]),
                    fmt(row["coverage"]),
                    fmt(row["answered_accuracy"]),
                    fmt(row["connectivity_accuracy"]),
                    fmt(row["count_accuracy"]),
                    fmt(row["spatial_count_accuracy"]),
                    fmt(row["value_accuracy"]),
                    fmt(row["mean_latency_seconds"]),
                ]
            )
            + " |"
        )
    markdown_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_count": len(rows), "csv": str(csv_target), "markdown": str(markdown_target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
