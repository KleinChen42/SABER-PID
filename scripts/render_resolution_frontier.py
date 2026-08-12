"""Render a dependency-free SVG for the 8B resolution/latency curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.table).read_text(encoding="utf-8"))["rows"]
    rows = [row for row in rows if "qwen3vl8b_" in str(row["label"])]
    order = {"qwen3vl8b_768": 768, "qwen3vl8b_1536": 1536, "qwen3vl8b_2304": 2304, "qwen3vl8b_3072": 3072}
    rows.sort(key=lambda row: order.get(str(row["label"]), 0))
    width, height = 900, 560
    left, top, chart_w, chart_h = 90, 65, 720, 350
    max_acc = max(float(row["overall_accuracy"]) for row in rows) * 1.18
    max_lat = max(float(row["mean_latency_seconds"]) for row in rows) * 1.2

    def esc(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(x: float, y: float, value: str, size: int = 14, anchor: str = "start", weight: str = "400") -> str:
        return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{esc(value)}</text>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(45, 32, "Qwen3-VL 8B source-disjoint resolution curve", 21, weight="700"),
        text(45, 52, "100 source sheets × 4 tasks; points are scored on the identical hidden answer store", 13),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#222"/>',
    ]
    for tick in range(0, 6):
        y = top + chart_h - chart_h * tick / 5
        value = max_acc * tick / 5
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        svg.append(text(left - 8, y + 5, f"{value * 100:.0f}%", 12, "end"))
    for tick in range(0, 6):
        x = left + chart_w * tick / 5
        value = max_lat * tick / 5
        svg.append(f'<line x1="{x:.1f}" y1="{top + chart_h}" x2="{x:.1f}" y2="{top + chart_h + 5}" stroke="#222"/>')
        svg.append(text(x, top + chart_h + 22, f"{value:.2f}", 12, "middle"))
    svg.append(text(24, top + chart_h / 2, "Accuracy", 14, "middle"))
    svg.append(text(left + chart_w / 2, top + chart_h + 44, "Mean latency (seconds)", 14, "middle"))

    points: list[tuple[float, float]] = []
    for row in rows:
        x = left + chart_w * float(row["mean_latency_seconds"]) / max_lat
        y = top + chart_h - chart_h * float(row["overall_accuracy"]) / max_acc
        points.append((x, y))
    if len(points) > 1:
        svg.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '" fill="none" stroke="#d55e00" stroke-width="3"/>')
    for row, (x, y) in zip(rows, points, strict=True):
        label = str(row["label"]).split("_")[-1]
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#d55e00" stroke="#111" stroke-width="1.5"/>')
        svg.append(text(x + 12, y - 10, label, 13, weight="700"))
        svg.append(text(x + 12, y + 7, f"{float(row[\"overall_accuracy\"]) * 100:.2f}%", 12))
    svg.extend([text(45, 505, "3072−768: +5.75 percentage points; source-cluster bootstrap 95% CI [−10.50, −1.25] for 768−3072.", 13), '</svg>'])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "points": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
