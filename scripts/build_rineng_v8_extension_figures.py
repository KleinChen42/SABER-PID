"""Build publication figures and tables for the completed v8 extensions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#666666",
    "light": "#D9D9D9",
    "dark": "#222222",
}
QUALITY_ORDER = ["clean", "jpeg_q70", "blur_r1", "downsample_s075"]
QUALITY_LABELS = {
    "clean": "Clean",
    "jpeg_q70": "JPEG Q70",
    "blur_r1": "Blur r=1",
    "downsample_s075": "0.75× restore",
}
POOLED = "pooled_three_source_disjoint_subsets"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_quality(
    report: dict[str, Any], quality: str, contrast: str, dataset: str = POOLED
) -> dict[str, Any]:
    for row in report["quality_comparisons"]:
        if (
            row["dataset"] == dataset
            and row["quality"] == quality
            and row["contrast"] == contrast
        ):
            return row
    raise KeyError(f"Missing quality row: {dataset}/{quality}/{contrast}")


def find_internvl(
    report: dict[str, Any], contrast: str, dataset: str = POOLED
) -> dict[str, Any]:
    for row in report["internvl_comparisons"]:
        if row["dataset"] == dataset and row["contrast"] == contrast:
            return row
    raise KeyError(f"Missing InternVL row: {dataset}/{contrast}")


def save_figure(fig: Any, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [pdf, png]


def interval_point(
    ax: Any,
    y: float,
    point: float,
    interval: list[float],
    color: str,
    label: str,
) -> None:
    low, high = sorted(map(float, interval))
    ax.plot([low, high], [y, y], color=color, linewidth=1.4, zorder=2)
    ax.plot([low, low], [y - 0.07, y + 0.07], color=color, linewidth=1.0, zorder=2)
    ax.plot([high, high], [y - 0.07, y + 0.07], color=color, linewidth=1.0, zorder=2)
    ax.plot(point, y, marker="o", markersize=5, color=color, zorder=3)
    ax.text(point, y + 0.16, label, ha="center", va="bottom", fontsize=6.4, color=color)


def style_axis(ax: Any, *, zero: bool = True) -> None:
    if zero:
        ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.7, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def build_quality_figure(report: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    direct_rows = [
        find_quality(report, quality, "correct_minus_shuffled")
        for quality in QUALITY_ORDER
    ]
    did_rows = [
        find_quality(
            report,
            quality,
            "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1",
        )
        for quality in QUALITY_ORDER
        if quality != "clean"
    ]
    qwen_clean = direct_rows[0]
    internvl = find_internvl(report, "correct_minus_shuffled")

    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.92], hspace=0.52, wspace=0.44)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    y = np.arange(len(direct_rows))[::-1]
    for position, row, quality, color in zip(
        y,
        direct_rows,
        QUALITY_ORDER,
        [COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]],
    ):
        point = float(row["value_f1_difference"])
        interval = list(map(float, row["value_f1_source_bootstrap_ci95"]))
        interval_point(ax_a, position, point, interval, color, f"{point:+.3f}")
    ax_a.set_yticks(y, [QUALITY_LABELS[name] for name in QUALITY_ORDER], fontsize=7)
    ax_a.set_xlabel("Correct − shuffled strict tag F1", fontsize=7.5)
    ax_a.set_title(
        "A  Requested-drawing effect\n     under image degradation",
        loc="left",
        fontsize=8.1,
        fontweight="bold",
    )
    style_axis(ax_a)

    y_b = np.arange(len(did_rows))[::-1]
    for position, row, quality, color in zip(
        y_b,
        did_rows,
        [name for name in QUALITY_ORDER if name != "clean"],
        [COLORS["orange"], COLORS["purple"], COLORS["green"]],
    ):
        point = float(row["value_f1_difference_in_differences"])
        interval = list(map(float, row["source_bootstrap_ci95"]))
        interval_point(ax_b, position, point, interval, color, f"{point:+.3f}")
    ax_b.set_yticks(y_b, [QUALITY_LABELS[name] for name in QUALITY_ORDER if name != "clean"], fontsize=7)
    ax_b.set_xlabel("Change in requested-drawing effect vs clean", fontsize=7.5)
    ax_b.set_title(
        "B  Paired degradation change\n     relative to clean",
        loc="left",
        fontsize=8.1,
        fontweight="bold",
    )
    style_axis(ax_b)

    cross = [
        (
            "Qwen3-VL-8B\n3072-side processor",
            float(qwen_clean["value_f1_difference"]),
            list(map(float, qwen_clean["value_f1_source_bootstrap_ci95"])),
            COLORS["blue"],
            "35.98M input tensor elements",
        ),
        (
            "InternVL3.5-8B\n54 × 448 tiles",
            float(internvl["value_f1_difference"]),
            list(map(float, internvl["value_f1_source_bootstrap_ci95"])),
            COLORS["green"],
            "32.51M input tensor elements",
        ),
    ]
    y_c = [1, 0]
    for position, (label, point, interval, color, budget) in zip(y_c, cross):
        interval_point(ax_c, position, point, interval, color, f"{point:+.3f}")
        ax_c.text(
            0.01,
            position - 0.22,
            budget,
            transform=ax_c.get_yaxis_transform(),
            fontsize=6.2,
            color=COLORS["gray"],
        )
    ax_c.set_yticks(y_c, [row[0] for row in cross], fontsize=7)
    ax_c.set_xlabel("Correct − shuffled strict tag F1", fontsize=7.5)
    ax_c.set_title("C  Closest safe cross-family visual-budget comparison", loc="left", fontsize=8.3, fontweight="bold")
    style_axis(ax_c)
    ax_c.text(
        1.0,
        -0.37,
        "Tensor-element matching does not equalize encoders or effective information throughput.",
        transform=ax_c.transAxes,
        ha="right",
        fontsize=6.3,
        color=COLORS["gray"],
    )
    fig.suptitle(
        "Qualified 3072/512 tag retrieval remains counterfactually testable under mild quality shifts",
        fontsize=10.2,
        fontweight="bold",
        y=0.995,
    )
    outputs = save_figure(fig, output_dir / "figure_5_quality_and_budget_matched_v8")
    return outputs, {"direct_rows": direct_rows, "did_rows": did_rows, "cross_family": cross}


def build_external_figure(report: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    conditions = ["correct", "shuffled", "text_only", "paddleocr_full_image"]
    labels = ["Qwen\ncorrect", "Qwen\nshuffled", "Qwen\nno image", "PaddleOCR\nfull image"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    metrics = report["metrics"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(conditions))
    width = 0.23
    metric_specs = [
        ("precision", "Precision", -width),
        ("recall", "Recall", 0.0),
        ("f1", "F1", width),
    ]
    hatches = ["", "//", "xx"]
    for (metric, label, offset), hatch in zip(metric_specs, hatches):
        values = [float(metrics[condition][metric]) for condition in conditions]
        bars = ax_a.bar(
            x + offset,
            values,
            width,
            label=label,
            color=colors,
            edgecolor="white" if not hatch else COLORS["dark"],
            linewidth=0.5,
            hatch=hatch,
            alpha=0.95,
            zorder=2,
        )
        if metric == "f1":
            for bar, value in zip(bars, values):
                ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=6.1)
    ax_a.set_xticks(x, labels, fontsize=6.8)
    ax_a.set_ylim(0, 1.10)
    ax_a.set_ylabel("Tag-set metric", fontsize=7.5)
    ax_a.set_title("A  Public DEXPI external operating points", loc="left", fontsize=8.3, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper center")
    ax_a.grid(axis="y", color=COLORS["light"], linewidth=0.7, zorder=0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    comparison_order = [
        "correct_minus_shuffled",
        "correct_minus_text_only",
        "correct_minus_paddleocr_full_image",
        "ocr_minus_text_only",
    ]
    comparison_labels = [
        "Correct − shuffled",
        "Correct − no image",
        "Correct − OCR",
        "OCR − no image",
    ]
    comparison_colors = [COLORS["orange"], COLORS["purple"], COLORS["green"], COLORS["gray"]]
    y = np.arange(len(comparison_order))[::-1]
    for position, key, color in zip(y, comparison_order, comparison_colors):
        row = report["comparisons"][key]
        point = float(row["difference"])
        interval = list(map(float, row["ci95"]))
        interval_point(ax_b, position, point, interval, color, f"{point:+.3f}")
    ax_b.set_yticks(y, comparison_labels, fontsize=7)
    ax_b.set_xlabel("Paired logical-group bootstrap F1 difference", fontsize=7.3)
    ax_b.set_title("B  Correct-image and OCR evidence contrasts", loc="left", fontsize=8.3, fontweight="bold")
    style_axis(ax_b)

    selection = report["selection"]
    fig.suptitle(
        f"Second public P&ID family: {selection['selected_image_count']} images, "
        f"{selection['logical_test_case_count']} logical cases, {selection['question_count']} questions",
        fontsize=9.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    outputs = save_figure(fig, output_dir / "figure_6_dexpi_external_v8")
    return outputs, {
        "conditions": {condition: metrics[condition] for condition in conditions},
        "comparisons": {key: report["comparisons"][key] for key in comparison_order},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--extension-score", default="reports/generated/rineng_v8_extension_score.json"
    )
    parser.add_argument(
        "--external-score", default="reports/generated/rineng_v8_dexpi_external_score.json"
    )
    parser.add_argument("--figure-dir", default="paper/figures")
    parser.add_argument(
        "--metadata", default="paper/figures/figure_metadata_v8.json"
    )
    parser.add_argument(
        "--quality-csv", default="reports/generated/rineng_v8_quality_effects.csv"
    )
    parser.add_argument(
        "--internvl-csv", default="reports/generated/rineng_v8_internvl_budget_effects.csv"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    extension_path = root / args.extension_score
    external_path = root / args.external_score
    extension = read_json(extension_path)
    external = read_json(external_path)
    if extension.get("status") != "pass" or external.get("status") != "pass":
        raise ValueError("Both frozen score reports must pass before plotting")
    write_csv(root / args.quality_csv, extension["quality_comparisons"])
    write_csv(root / args.internvl_csv, extension["internvl_comparisons"])

    figure_dir = root / args.figure_dir
    quality_outputs, quality_values = build_quality_figure(extension, figure_dir)
    external_outputs, external_values = build_external_figure(external, figure_dir)
    metadata = {
        "version": "rineng-v8-figures",
        "status": "pass",
        "sources": {
            args.extension_score: sha256(extension_path),
            args.external_score: sha256(external_path),
        },
        "figures": {
            "figure_5_quality_and_budget_matched_v8": {
                "files": [path.relative_to(root).as_posix() for path in quality_outputs],
                "values": quality_values,
            },
            "figure_6_dexpi_external_v8": {
                "files": [path.relative_to(root).as_posix() for path in external_outputs],
                "values": external_values,
            },
        },
    }
    metadata_path = root / args.metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "figures": 2,
                "metadata": args.metadata,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
