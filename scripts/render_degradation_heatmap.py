"""Render a compact SVG task-condition heatmap from score-table JSON."""

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
    columns = ["connectivity_accuracy", "count_accuracy", "spatial_count_accuracy", "value_accuracy", "overall_accuracy"]
    names = ["Connectivity", "Count", "Spatial", "Value", "Overall"]
    width, height = 850, 360
    left, top, cell_w, cell_h = 210, 90, 115, 44

    def esc(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(x: float, y: float, value: str, size: int = 14, anchor: str = "start", weight: str = "400") -> str:
        return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{esc(value)}</text>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(35, 35, "Qwen3-VL 8B controlled degradation at 1536", 21, weight="700"),
        text(35, 57, "All conditions use the same 100-source × 4-task hidden answer store", 13),
    ]
    for col, name in enumerate(names):
        svg.append(text(left + col * cell_w + cell_w / 2, top - 18, name, 12, "middle", "700"))
    for row_index, row in enumerate(rows):
        label = str(row["label"]).replace("qwen3vl8b_", "")
        svg.append(text(left - 12, top + row_index * cell_h + cell_h / 2 + 5, label, 13, "end", "700"))
        for col, key in enumerate(columns):
            value = float(row[key])
            intensity = max(0.0, min(1.0, value / 0.6))
            red = int(245 - 130 * intensity)
            green = int(245 - 65 * intensity)
            blue = int(245 - 20 * intensity)
            x = left + col * cell_w
            y = top + row_index * cell_h
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="rgb({red},{green},{blue})" stroke="white"/>')
            svg.append(text(x + (cell_w - 4) / 2, y + (cell_h - 4) / 2 + 5, f"{value * 100:.1f}%", 14, "middle", "700"))
    svg.extend([text(35, 330, "Clean: 22.25%; blur: 21.25%; JPEG35: 22.75%; center-crop: 25.25%.", 13), "</svg>"])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "rows": len(rows), "columns": len(columns)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
