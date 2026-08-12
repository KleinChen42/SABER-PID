"""Render a dependency-free SVG for the manuscript's two main findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def text(x: float, y: float, value: str, size: int = 18, anchor: str = "start", weight: str = "400") -> str:
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escaped}</text>'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    exposure = json.loads(Path(args.exposure).read_text(encoding="utf-8"))["aggregate"]
    scores = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.resolution.split(",")]
    labels = ["3072 clean", "1536 clean", "1536 crop"]
    values = [100 * float(score["overall_accuracy"]) for score in scores]
    exposure_values = [
        100 * float(exposure["random"]["mean_unambiguous_cache_hit_rate"]),
        100 * float(exposure["source"]["mean_unambiguous_cache_hit_rate"]),
    ]

    width, height = 1180, 610
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">P&amp;ID reliability evidence</title>',
        '<desc id="desc">Left: same-drawing semantic cache exposure is present under random QA splits but removed by source isolation. Right: Qwen3-VL 8B accuracy is higher at 3072 than 1536 maximum image side on 50 source sheets.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        text(60, 52, "A. Same-drawing semantic-query exposure", 24, weight="700"),
        text(650, 52, "B. Source-disjoint resolution result", 24, weight="700"),
    ]

    def bars(x0: int, chart_width: int, chart_height: int, top: int, names: list[str], percents: list[float], color: str, max_value: float, y_label: str) -> None:
        baseline = top + chart_height
        for tick in range(0, int(max_value) + 1, 10):
            y = baseline - chart_height * tick / max_value
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + chart_width}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="1"/>')
            svg.append(text(x0 - 10, y + 6, str(tick), 14, "end"))
        svg.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{baseline}" stroke="#222" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x0}" y1="{baseline}" x2="{x0 + chart_width}" y2="{baseline}" stroke="#222" stroke-width="1.5"/>')
        svg.append(text(x0 - 44, top + chart_height / 2, y_label, 15, "middle"))
        bar_width = min(120, chart_width / (len(names) * 1.8))
        gap = (chart_width - len(names) * bar_width) / (len(names) + 1)
        for index, (name, value) in enumerate(zip(names, percents, strict=True)):
            x = x0 + gap * (index + 1) + bar_width * index
            bar_height = chart_height * value / max_value
            y = baseline - bar_height
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
            svg.append(text(x + bar_width / 2, y - 10, f"{value:.1f}%", 17, "middle", "700"))
            svg.append(text(x + bar_width / 2, baseline + 28, name, 15, "middle"))

    bars(105, 420, 360, 130, ["Random QA", "Source-isolated"], exposure_values, "#1f77b4", 30, "Retrievable test answers (%)")
    bars(690, 420, 360, 130, labels, values, "#d55e00", 40, "Task-aware accuracy (%)")
    svg.extend(
        [
            text(315, 550, "5 seeds; cache is a diagnostic, not a VLM score", 14, "middle"),
            text(900, 550, "50 source sheets × 4 tasks; direct Qwen3-VL 8B", 14, "middle"),
            text(590, 590, "Source isolation removes same-drawing query exposure; higher visual resolution improves measured accuracy at added latency.", 16, "middle"),
            "</svg>",
        ]
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "panels": 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
