"""Render the application-forward v5 main figures from frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_paper_figures_v4 import (
    COLORS,
    draw_interval,
    find_comparison,
    output_card,
    protocol_strip,
    read_json,
    reverse,
    save,
    sha256,
    tidy_axis,
)


def value_stats(report: dict[str, Any], cell: str) -> dict[str, float]:
    metrics = report["cells"][cell]["metrics"]
    tags = metrics["strict_value_tags"]
    return {
        "precision": float(tags["precision"]),
        "recall": float(tags["recall"]),
        "f1": float(tags["f1"]),
        "exact": float(metrics["task"]["value"]["strict_accuracy"]),
    }


def validation_seed(report: dict[str, Any], seed: int) -> dict[str, Any]:
    for row in report["seeds"]:
        if int(row["seed"]) == seed:
            return row
    raise KeyError(f"Missing fusion validation seed {seed}")


def figure_1(root: Path, directory: Path, analysis: dict[str, Any]) -> list[Path]:
    value = analysis["value_evidence_case"]
    fig = plt.figure(figsize=(7.15, 4.75))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.72, 0.48, 3.25],
        width_ratios=[0.42, 0.58],
        hspace=0.16,
        wspace=0.18,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.07)
    fig.suptitle(
        "Counterfactual validation of image-grounded P&ID tag reading",
        fontsize=11.2,
        fontweight="bold",
    )
    protocol_strip(fig.add_subplot(grid[0, :]))

    ax = fig.add_subplot(grid[1, :])
    ax.set_axis_off()
    ax.text(
        0.0,
        0.76,
        "Deterministically selected complete-recovery case",
        fontsize=9.1,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.12,
        f"Question: {value['question']}   |   Reference: {', '.join(value['reference_tags'])}",
        fontsize=6.8,
        va="bottom",
    )

    ax_img = fig.add_subplot(grid[2, 0])
    ax_img.imshow(Image.open(root / value["image_path"]).convert("RGB"))
    ax_img.set_axis_off()
    ax_img.set_title(
        "PIDQA sheet 282 (CC0)\ncomplete source drawing; no answer-guided crop",
        loc="left",
        fontsize=6.7,
    )
    ax_img.text(
        0.0,
        -0.08,
        f"16 eligible records; minimum SHA-256 rank {value['rank_sha256'][:12]}...",
        transform=ax_img.transAxes,
        fontsize=5.8,
        color=COLORS["gray"],
        va="top",
    )

    cards = grid[2, 1].subgridspec(2, 2, hspace=0.13, wspace=0.10)
    labels = [
        ("correct_768", "768 correct image", COLORS["gray"]),
        ("correct_3072", "3072 correct image", COLORS["blue"]),
        ("shuffled_3072", "3072 source-shuffled image", COLORS["orange"]),
        ("text_only", "No image", COLORS["purple"]),
    ]
    axes = [fig.add_subplot(cards[i, j]) for i in range(2) for j in range(2)]
    for axis, (key, title, color) in zip(axes, labels):
        item = value["conditions"][key]
        output_card(
            axis,
            title,
            item["raw_output"],
            item["parsed_tags"],
            float(item["f1"]),
            int(item["exact_set"]),
            color,
        )
    return save(fig, directory, "figure_1_image_grounded_tag_reading_v5")


def figure_2(
    directory: Path,
    e2: dict[str, Any],
    e3: dict[str, Any],
    e6: dict[str, Any],
    e8: dict[str, Any],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.25), gridspec_kw={"width_ratios": [1.05, 1.20]})
    fig.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.18, wspace=0.62)
    fig.suptitle(
        "Large tag-reading effects persist across controls and source partitions",
        fontsize=11.0,
        fontweight="bold",
    )

    rows = [
        (
            "3072 - 768\n(512-token cap)",
            find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value"),
            COLORS["blue"],
        ),
        (
            "Correct - shuffled\n(3072 / 192)",
            reverse(find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_value_tag_f1", "value")),
            COLORS["orange"],
        ),
        (
            "Correct - no image\n(3072 / 192)",
            find_comparison(e8, "e8_correct_image_minus_text_only_3072", "strict_value_tag_f1", "value"),
            COLORS["purple"],
        ),
    ]
    for y, (_, row, color) in zip([2, 1, 0], rows):
        draw_interval(axes[0], row, y, color, fontsize=6.3)
    axes[0].axvline(0, color=COLORS["gray"], linewidth=0.8)
    axes[0].set_yticks([2, 1, 0], [label for label, _, _ in rows])
    axes[0].set_xlim(-0.05, 0.74)
    axes[0].set_ylim(-0.55, 2.55)
    axes[0].set_xlabel("Strict value-tag F1 difference")
    axes[0].set_title("A. Counterfactual effects", loc="left", fontsize=9.2, fontweight="bold")
    tidy_axis(axes[0])

    stability = [
        ("Set B / 192", find_comparison(e2, "e2_192_3072_minus_768_reference", "strict_value_tag_f1", "value")),
        ("Set B / 512", find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value")),
        ("Seed 29", find_comparison(e6, "e6_seed29_3072_minus_768", "strict_value_tag_f1", "value")),
        ("Seed 31", find_comparison(e6, "e6_seed31_3072_minus_768", "strict_value_tag_f1", "value")),
    ]
    for y, (label, row) in zip([3, 2, 1, 0], stability):
        color = COLORS["blue"] if label.startswith("Set B") else COLORS["green"]
        draw_interval(axes[1], row, y, color, fontsize=6.2)
    axes[1].axvline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_yticks([3, 2, 1, 0], [label for label, _ in stability])
    axes[1].set_xlim(-0.05, 0.86)
    axes[1].set_ylim(-0.55, 3.55)
    axes[1].set_xlabel("3072 - 768 strict tag F1")
    axes[1].set_title("B. Caps and source partitions", loc="left", fontsize=9.2, fontweight="bold")
    tidy_axis(axes[1])
    fig.text(
        0.18,
        0.045,
        "Intervals use 10,000 source-cluster bootstrap replicates; seed partitions are descriptive sensitivity analyses.",
        fontsize=6.8,
        color=COLORS["gray"],
    )
    return save(fig, directory, "figure_2_tag_reading_robustness_v5")


def figure_3(directory: Path, fusion: dict[str, Any], validation: dict[str, Any]) -> list[Path]:
    order = [
        ("Qwen", "qwen"),
        ("PaddleOCR", "paddleocr_geometry"),
        ("Set union", "set_union"),
        ("Set intersection", "set_intersection"),
        ("OCR-first fallback", "ocr_if_nonempty_else_qwen"),
    ]
    stats = [(label, value_stats(fusion, key)) for label, key in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.45), gridspec_kw={"width_ratios": [1.35, 1.00]})
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.22, wspace=0.42)
    fig.suptitle(
        "Complementary OCR and VLM predictions create a controllable operating envelope",
        fontsize=10.7,
        fontweight="bold",
    )

    x = np.arange(len(stats))
    width = 0.24
    for offset, metric, color in [
        (-width, "precision", COLORS["blue"]),
        (0.0, "recall", COLORS["orange"]),
        (width, "f1", COLORS["green"]),
    ]:
        values = [row[metric] for _, row in stats]
        axes[0].bar(x + offset, values, width=width, label=metric.capitalize(), color=color, alpha=0.90)
    axes[0].set_xticks(x, [label.replace(" ", "\n", 1) for label, _ in stats])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Strict tag-set metric")
    axes[0].set_title("A. Precision-recall choices", loc="left", fontsize=9.1, fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, fontsize=6.7, loc="upper center")
    axes[0].grid(axis="y", color=COLORS["light"], linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        axes[0].spines[spine].set_visible(False)
    axes[0].tick_params(axis="x", labelsize=6.5)

    seed29 = validation_seed(validation, 29)
    seed31 = validation_seed(validation, 31)
    comparisons = [
        ("Set B union - Qwen", find_comparison(fusion, "union_minus_qwen", "strict_value_tag_f1", "value"), COLORS["green"]),
        ("Seed 29 union - Qwen", find_comparison(seed29, "seed29_union_minus_qwen", "strict_value_tag_f1", "value"), COLORS["blue"]),
        ("Seed 31 union - Qwen", find_comparison(seed31, "seed31_union_minus_qwen", "strict_value_tag_f1", "value"), COLORS["blue"]),
        ("Set B union - OCR", find_comparison(fusion, "union_minus_ocr", "strict_value_tag_f1", "value"), COLORS["gray"]),
    ]
    for y, (_, row, color) in zip([3, 2, 1, 0], comparisons):
        draw_interval(axes[1], row, y, color, fontsize=6.2)
    axes[1].axvline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_yticks([3, 2, 1, 0], [label for label, _, _ in comparisons])
    endpoints = [
        float(row[key])
        for _, row, _ in comparisons
        for key in ("source_bootstrap_ci95_low", "source_bootstrap_ci95_high")
    ] + [0.0]
    span = max(endpoints) - min(endpoints)
    pad = max(0.025, 0.12 * span)
    axes[1].set_xlim(min(endpoints) - pad, max(endpoints) + pad)
    axes[1].set_ylim(-0.55, 3.55)
    axes[1].set_xlabel("Strict tag-F1 difference")
    axes[1].set_title("B. Frozen union checks", loc="left", fontsize=9.1, fontweight="bold")
    tidy_axis(axes[1])
    fig.text(
        0.10,
        0.045,
        "Fusion is reference-free; all four rules are reported. Set B was post-hoc; seed 29/31 were scored after rule freezing within PIDQA.",
        fontsize=6.7,
        color=COLORS["gray"],
    )
    return save(fig, directory, "figure_3_hybrid_tag_operating_envelope_v5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    directory = root / args.output_dir
    generated = root / "reports/generated"
    analysis = read_json(generated / "editorial_revision_evidence_v4.json")
    e2 = read_json(generated / "qwen8_value_budget_sensitivity_v1.json")
    e3 = read_json(generated / "image_dependence_control_v1.json")
    e6 = read_json(generated / "source_seed_resolution_sensitivity_v1.json")
    e8 = read_json(generated / "text_only_image_grounding_control_v1.json")
    fusion = read_json(generated / "positive_narrative_hybrid_analysis_v5.json")
    validation = read_json(generated / "positive_narrative_fusion_validation_v5.json")

    outputs: list[Path] = []
    outputs.extend(figure_1(root, directory, analysis))
    outputs.extend(figure_2(directory, e2, e3, e6, e8))
    outputs.extend(figure_3(directory, fusion, validation))
    sources = [
        "reports/generated/editorial_revision_evidence_v4.json",
        "reports/generated/qwen8_value_budget_sensitivity_v1.json",
        "reports/generated/image_dependence_control_v1.json",
        "reports/generated/source_seed_resolution_sensitivity_v1.json",
        "reports/generated/text_only_image_grounding_control_v1.json",
        "reports/generated/positive_narrative_hybrid_analysis_v5.json",
        "reports/generated/positive_narrative_fusion_validation_v5.json",
        "paper/assets/pidqa_sheet_282.jpg",
    ]
    metadata = {
        "status": "pass",
        "version": "positive-narrative-v5",
        "generator": "scripts/build_positive_narrative_figures_v5.py",
        "source_artifacts": [{"path": path, "sha256": sha256(root / path)} for path in sources],
        "files": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "image_policy": "deterministic composition of frozen evidence and one SHA-selected CC0 PIDQA drawing; no generative image model or answer-guided crop",
    }
    metadata_path = directory / "figure_metadata_v5.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figure_count": len(outputs), "metadata": str(metadata_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
