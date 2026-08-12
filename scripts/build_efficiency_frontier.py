"""Build accuracy/latency/memory efficiency artifacts from scorer JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_items(items: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        result.append((label, Path(path)))
    return result


def parse_memory(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=GiB, got {item!r}")
        label, value = item.split("=", 1)
        result[label] = float(value)
    return result


def dominated(row: dict[str, object], others: list[dict[str, object]]) -> bool:
    accuracy = float(row["overall_accuracy"])
    latency = float(row["mean_latency_seconds"])
    memory = float(row["memory_gib"])
    for other in others:
        if other is row:
            continue
        other_accuracy = float(other["overall_accuracy"])
        other_latency = float(other["mean_latency_seconds"])
        other_memory = float(other["memory_gib"])
        no_worse = (
            other_accuracy >= accuracy
            and other_latency <= latency
            and other_memory <= memory
        )
        strictly_better = (
            other_accuracy > accuracy
            or other_latency < latency
            or other_memory < memory
        )
        if no_worse and strictly_better:
            return True
    return False


def render_svg(rows: list[dict[str, object]], target: Path) -> None:
    width, height = 900, 620
    left, top, chart_w, chart_h = 95, 70, 730, 430
    max_latency = max(float(row["mean_latency_seconds"]) for row in rows) * 1.15
    max_memory = max(float(row["memory_gib"]) for row in rows) * 1.15
    max_accuracy = max(float(row["overall_accuracy"]) for row in rows) * 1.15

    def esc(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(x: float, y: float, value: str, size: int = 14, anchor: str = "start", weight: str = "400") -> str:
        return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{esc(value)}</text>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(45, 35, "PIDQA efficiency frontier (100-source main set)", 21, weight="700"),
        text(45, 57, "x: mean request latency (s), y: task-aware accuracy, bubble: observed H200 memory", 13),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#222"/>',
    ]
    for tick in range(0, 6):
        y = top + chart_h - chart_h * tick / 5
        value = max_accuracy * tick / 5
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        svg.append(text(left - 8, y + 5, f"{value * 100:.0f}%", 12, "end"))
    for tick in range(0, 6):
        x = left + chart_w * tick / 5
        value = max_latency * tick / 5
        svg.append(f'<line x1="{x:.1f}" y1="{top + chart_h}" x2="{x:.1f}" y2="{top + chart_h + 5}" stroke="#222"/>')
        svg.append(text(x, top + chart_h + 24, f"{value:.2f}", 12, "middle"))
    svg.append(text(25, top + chart_h / 2, "Accuracy", 14, "middle"))
    svg.append(text(left + chart_w / 2, top + chart_h + 48, "Mean latency (seconds)", 14, "middle"))

    palette = {"8b": "#0072b2", "32b": "#d55e00"}
    for row in rows:
        label = str(row["label"])
        model_color = palette["32b"] if "32b" in label else palette["8b"]
        x = left + chart_w * float(row["mean_latency_seconds"]) / max_latency
        y = top + chart_h - chart_h * float(row["overall_accuracy"]) / max_accuracy
        radius = 7 + 18 * float(row["memory_gib"]) / max_memory
        stroke = "#111" if row["pareto"] else "none"
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{model_color}" fill-opacity="0.72" stroke="{stroke}" stroke-width="2"/>')
        svg.append(text(x + radius + 4, y + 4, label, 11))
    svg.extend([text(45, 575, "Outlined points are non-dominated under accuracy↑, latency↓, memory↓; energy: not measured.", 13), "</svg>"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--memory", action="append", default=[], metavar="LABEL=GiB")
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--svg", required=True)
    args = parser.parse_args()

    memory = parse_memory(args.memory)
    rows: list[dict[str, object]] = []
    for label, path in parse_items(args.score):
        payload = json.loads(path.read_text(encoding="utf-8"))
        count = int(payload.get("prediction_count", 0) or 0)
        total = float(payload.get("total_generation_seconds", 0.0) or 0.0)
        rows.append(
            {
                "label": label,
                "score_file": str(path),
                "model_memory_key": "32b" if "32b" in label else "8b",
                "memory_gib": memory.get(label, memory.get("32b" if "32b" in label else "8b", 0.0)),
                "record_count": int(payload.get("record_count", 0) or 0),
                "overall_accuracy": float(payload.get("overall_accuracy", 0.0) or 0.0),
                "coverage": float(payload.get("coverage", 0.0) or 0.0),
                "mean_latency_seconds": float(payload.get("mean_latency_seconds", 0.0) or 0.0),
                "median_latency_seconds": float(payload.get("median_latency_seconds", 0.0) or 0.0),
                "p95_latency_seconds": float(payload.get("p95_latency_seconds", 0.0) or 0.0),
                "total_generation_seconds": total,
                "requests_per_second": count / total if count and total > 0 else 0.0,
                "energy_status": "not_measured",
            }
        )
    for row in rows:
        row["pareto"] = not dominated(row, rows)

    payload = {"energy_status": "not_measured", "memory_basis": "observed H200 run memory; not hardware-independent peak", "rows": rows}
    json_path, csv_path, svg_path = Path(args.json), Path(args.csv), Path(args.svg)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    columns = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    render_svg(rows, svg_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
