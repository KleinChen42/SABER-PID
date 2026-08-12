"""Render the v3 manuscript and supplement figures from frozen evidence.

The figures are deterministic Matplotlib graphics.  They do not call a
generative model and they do not alter any inference or scoring artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#5C5C5C",
    "light": "#E8E8E8",
    "dark": "#222222",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_comparison(
    report: dict[str, Any], label: str, metric: str, task: str
) -> dict[str, Any]:
    for row in report["comparisons"]:
        if row["comparison"] == label and row["metric"] == metric and row["task"] == task:
            return row
    raise KeyError(f"Missing comparison {label}/{metric}/{task}")


def reverse_orientation(row: dict[str, Any]) -> dict[str, Any]:
    """Return an equivalent effect with baseline/condition direction reversed."""

    reversed_row = dict(row)
    reversed_row["baseline_mean"] = row["condition_mean"]
    reversed_row["condition_mean"] = row["baseline_mean"]
    reversed_row["difference_condition_minus_baseline"] = -float(
        row["difference_condition_minus_baseline"]
    )
    reversed_row["source_bootstrap_ci95_low"] = -float(
        row["source_bootstrap_ci95_high"]
    )
    reversed_row["source_bootstrap_ci95_high"] = -float(
        row["source_bootstrap_ci95_low"]
    )
    return reversed_row


def save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    output = [directory / f"{stem}.pdf", directory / f"{stem}.png"]
    fig.savefig(output[0], bbox_inches="tight")
    fig.savefig(output[1], dpi=400, bbox_inches="tight")
    plt.close(fig)
    return output


def tidy_axis(ax: Any, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=COLORS["light"], linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def draw_interval(
    ax: Any,
    row: dict[str, Any],
    y: float,
    color: str,
    *,
    label: bool = True,
) -> None:
    value = float(row["difference_condition_minus_baseline"])
    lower = float(row["source_bootstrap_ci95_low"])
    upper = float(row["source_bootstrap_ci95_high"])
    ax.errorbar(
        value,
        y,
        xerr=[[value - lower], [upper - value]],
        fmt="o",
        color=color,
        ecolor=color,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
        capsize=3.5,
        linewidth=1.7,
        zorder=3,
    )
    if label:
        offset = 0.014 if value >= 0 else -0.014
        alignment = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            y + 0.13,
            f"{value:+.3f} [{lower:+.3f}, {upper:+.3f}]",
            color=COLORS["dark"],
            fontsize=7.2,
            ha=alignment,
            va="bottom",
        )


def retrieval_summary(report: dict[str, Any]) -> dict[str, float]:
    rows = [
        row
        for row in report["rows"]
        if row["method"] == "L5_image_semantic_with_prior"
        and row["split"] in {"random", "source"}
    ]
    result: dict[str, float] = {}
    for split in ("random", "source"):
        values = [float(row["overall_accuracy"]) for row in rows if row["split"] == split]
        if len(values) != 5:
            raise ValueError(f"Expected five L5 {split} retrieval rows, got {len(values)}")
        result[f"{split}_mean"] = sum(values) / len(values)
        result[f"{split}_min"] = min(values)
        result[f"{split}_max"] = max(values)
    result["gap"] = result["random_mean"] - result["source_mean"]
    return result


def protocol_box(ax: Any, x: float, title: str, detail: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, 0.37),
        0.16,
        0.32,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(x + 0.08, 0.60, title, ha="center", va="center", fontsize=7.7, fontweight="bold")
    ax.text(x + 0.08, 0.47, detail, ha="center", va="center", fontsize=6.2, color=COLORS["gray"], wrap=True)


def figure_overview(
    directory: Path,
    retrieval: dict[str, float],
    e2_512: dict[str, Any],
    e3_correct_shuffled: dict[str, Any],
    e5_3072: dict[str, Any],
) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 5.25))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        width_ratios=[0.82, 1.58],
        hspace=0.62,
        wspace=0.48,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12)
    fig.suptitle(
        "SABER-PID: traceable P&ID VLM evidence",
        fontsize=11.5,
        fontweight="bold",
    )

    ax = axes[0]
    values = [retrieval["source_mean"], retrieval["random_mean"]]
    labels = ["Source-isolated", "Question-random"]
    colors = [COLORS["blue"], COLORS["orange"]]
    ax.barh(labels, values, color=colors, height=0.55, zorder=2)
    for y, value in enumerate(values):
        ax.text(value + 0.014, y, f"{value * 100:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("L5 input-retrieval + prior accuracy")
    ax.set_title("A. Source split", loc="left", fontsize=9.8, fontweight="bold")
    tidy_axis(ax)

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("Source", "IDs stay\ndisjoint", COLORS["blue"]),
        ("Answer", "answers\nhidden", COLORS["green"]),
        ("Budget", "record actual\npixels/tokens", COLORS["orange"]),
        ("Evidence", "wrong image /\ntext only", COLORS["purple"]),
        ("Report", "by task +\nsource CI", COLORS["blue"]),
    ]
    positions = [0.015, 0.215, 0.415, 0.615, 0.815]
    for index, ((title, detail, color), x) in enumerate(zip(steps, positions)):
        protocol_box(ax, x, title, detail, color)
        if index < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.16, 0.53),
                    (positions[index + 1] - 0.012, 0.53),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.1,
                    color=COLORS["gray"],
                )
            )
    ax.text(0.5, 0.84, "B. SABER-PID contract", ha="center", va="center", fontsize=9.8, fontweight="bold")
    ax.text(0.5, 0.16, "Source • Answer • Budget • Explicit context • Report by task", ha="center", va="center", fontsize=6.8, color=COLORS["gray"])

    ax = axes[2]
    effects = [
        ("High-detail tag F1\n(512-token control)", e2_512, COLORS["blue"]),
        ("Correct image minus\nshuffled image", e3_correct_shuffled, COLORS["orange"]),
        ("Visible symbol context\n(spatial count, 3072)", e5_3072, COLORS["green"]),
    ]
    y_positions = [2, 1, 0]
    for y, (label, row, color) in zip(y_positions, effects):
        draw_interval(ax, row, y, color, label=False)
        value = float(row["difference_condition_minus_baseline"])
        ax.text(value + 0.025, y, f"{value * 100:+.1f} pp", fontsize=9, va="center")
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8, zorder=1)
    ax.set_yticks(y_positions, [label for label, _, _ in effects])
    ax.set_xlim(-0.08, 0.73)
    ax.set_xlabel("Paired source-level effect (metric stated at left)")
    ax.set_title("C. Actionable task-specific findings", loc="left", fontsize=9.8, fontweight="bold")
    tidy_axis(ax)
    fig.text(0.08, 0.025, f"Five fixed splits: random - source = {retrieval['gap'] * 100:+.1f} points.", fontsize=7, color=COLORS["gray"])
    fig.text(0.47, 0.025, "Effects use task-specific metrics; Figure 2 gives intervals and mapping attribution.", fontsize=7, color=COLORS["gray"])
    return save(fig, directory, "figure_1_saber_pid_overview")


def figure_core_effects(
    directory: Path,
    retrieval: dict[str, float],
    e2_512: dict[str, Any],
    e3_correct_shuffled: dict[str, Any],
    e8_correct_text: dict[str, Any],
    e5_768: dict[str, Any],
    e5_3072: dict[str, Any],
    e7_768: dict[str, Any],
    e7_3072: dict[str, Any],
) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 6.25))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.20],
        width_ratios=[0.82, 1.45],
        hspace=0.55,
        wspace=0.56,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.10)
    fig.suptitle("Core evidence: shortcut closure, image grounding, and symbol context", fontsize=11.2, fontweight="bold")

    ax = axes[0]
    mean = retrieval["gap"]
    low = retrieval["random_min"] - retrieval["source_max"]
    high = retrieval["random_max"] - retrieval["source_min"]
    ax.errorbar(mean, 0, xerr=[[mean - low], [high - mean]], fmt="o", color=COLORS["orange"], capsize=4, linewidth=1.8, markersize=8)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.text(mean, 0.14, f"{mean:+.3f}\nrange [{low:+.3f}, {high:+.3f}]", ha="center", fontsize=8.6)
    ax.set_xlim(-0.03, 0.27)
    ax.set_ylim(-0.3, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("Random split - source split")
    ax.set_title("A. Retrieval exposure\n(five fixed splits)", loc="left", fontsize=9.4, fontweight="bold")
    tidy_axis(ax)

    ax = axes[1]
    tag_rows = [
        ("3072 - 768\n(512-token control)", e2_512, COLORS["blue"]),
        ("Correct image -\nshuffled image", e3_correct_shuffled, COLORS["orange"]),
        ("Correct image -\ntext-only", e8_correct_text, COLORS["purple"]),
    ]
    y_positions = [2, 1, 0]
    for y, (_, row, color) in zip(y_positions, tag_rows):
        draw_interval(ax, row, y, color)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y_positions, [label for label, _, _ in tag_rows])
    ax.set_xlim(-0.08, 0.76)
    ax.set_ylim(-0.45, 2.55)
    ax.set_xlabel("Strict value-tag F1 difference")
    ax.set_title("B. Image-grounded tag reading", loc="left", fontsize=9.4, fontweight="bold")
    tidy_axis(ax)

    ax = axes[2]
    context_rows = [
        ("Visible legend - raw, 768", e5_768, COLORS["green"]),
        ("Visible legend - raw, 3072", e5_3072, COLORS["green"]),
        ("Correct map - permuted map, 768", e7_768, COLORS["purple"]),
        ("Correct map - permuted map, 3072", e7_3072, COLORS["purple"]),
    ]
    y_positions = [3, 2, 1, 0]
    for y, (_, row, color) in zip(y_positions, context_rows):
        draw_interval(ax, row, y, color)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y_positions, [label for label, _, _ in context_rows])
    ax.set_xlim(-0.12, 0.48)
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlabel("Semantic spatial-count accuracy difference")
    ax.set_title("C. Visible context, not mapping attribution", loc="left", fontsize=9.4, fontweight="bold")
    tidy_axis(ax)
    fig.text(0.11, 0.025, "Cyclic numeric-label permutation preserves the E5 gain; it does not support a mapping-semantic claim.", fontsize=6.8, color=COLORS["gray"])
    return save(fig, directory, "figure_2_core_effects")


def figure_tag_stability(
    directory: Path,
    e2_192: dict[str, Any],
    e2_512: dict[str, Any],
    seed29: dict[str, Any],
    seed31: dict[str, Any],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.15, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.84, bottom=0.20)
    rows = [
        ("Set B, 192 tokens", e2_192, COLORS["blue"]),
        ("Set B, 512 tokens", e2_512, COLORS["blue"]),
        ("Source split seed 29", seed29, COLORS["orange"]),
        ("Source split seed 31", seed31, COLORS["orange"]),
    ]
    positions = [3, 2, 1, 0]
    for y, (_, row, color) in zip(positions, rows):
        draw_interval(ax, row, y, color)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(positions, [label for label, _, _ in rows])
    ax.set_xlim(-0.08, 0.86)
    ax.set_ylim(-0.5, 3.48)
    ax.set_xlabel("Strict value-tag F1: 3072 - 768")
    ax.set_title("Image-grounded tag-reading gains across budgets and source partitions", loc="left", fontsize=10.6, fontweight="bold")
    tidy_axis(ax)
    fig.text(0.12, 0.045, "Points are paired source-level differences; bars are 95% source-cluster bootstrap intervals (n = 100 sources per row).", fontsize=7.2, color=COLORS["gray"])
    return save(fig, directory, "figure_3_tag_reading_stability")


def cell_task_scores(report: dict[str, Any], cell_name: str) -> dict[str, float]:
    metrics = report["cells"][cell_name]["metrics"]
    return {
        task: float(metrics["task"][task]["strict_accuracy"])
        for task in ("connectivity", "count", "spatial_count", "value")
    }


def figure_calibration_controls(directory: Path, e8: dict[str, Any], e3: dict[str, Any], e4: dict[str, Any]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14, wspace=0.46)
    tasks = ["connectivity", "count", "spatial_count", "value"]
    labels = ["Connectivity", "Count", "Spatial\ncount", "Value"]
    conditions = [
        ("Text-only", cell_task_scores(e8, "qwen8_b_p0_text_only"), COLORS["gray"]),
        ("Shuffled image", cell_task_scores(e8, "qwen8_b_p0_shuffled_3072"), COLORS["orange"]),
        ("Correct image", cell_task_scores(e8, "qwen8_b_p0_correct_3072"), COLORS["blue"]),
    ]
    offsets = [-0.25, 0.0, 0.25]
    for offset, (label, scores, color) in zip(offsets, conditions):
        axes[0].bar([index + offset for index in range(4)], [scores[task] for task in tasks], width=0.23, label=label, color=color, zorder=2)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylim(0, 0.72)
    axes[0].set_ylabel("Strict accuracy")
    axes[0].set_title("S1A. Task-level calibration at 3072", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    tidy_axis(axes[0], grid_axis="y")

    shuffled_structural = [
        reverse_orientation(find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_correct", task))
        for task in ("connectivity", "count", "spatial_count")
    ]
    tile_rows = [
        find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", task)
        for task in ("overall", "connectivity", "count", "spatial_count")
    ]
    left_labels = ["Correct - shuffled\nconnectivity", "Correct - shuffled\ncount", "Correct - shuffled\nspatial count"]
    right_labels = ["7 - 1 tile\noverall", "7 - 1 tile\nconnectivity", "7 - 1 tile\ncount", "7 - 1 tile\nspatial count"]
    y_positions = [6, 5, 4]
    for y, row in zip(y_positions, shuffled_structural):
        draw_interval(axes[1], row, y, COLORS["orange"])
    for y, row in zip([2.5, 1.5, 0.5, -0.5], tile_rows):
        draw_interval(axes[1], row, y, COLORS["purple"])
    axes[1].axvline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_yticks(y_positions + [2.5, 1.5, 0.5, -0.5], left_labels + right_labels)
    axes[1].set_xlim(-0.32, 0.18)
    axes[1].set_xlabel("Paired source-level difference")
    axes[1].set_title("S1B. Boundary controls retained in full", loc="left", fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=6.5)
    tidy_axis(axes[1])
    return save(fig, directory, "figure_s1_task_calibration_and_boundaries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    generated = root / "reports" / "generated"
    directory = root / args.output_dir

    retrieval = retrieval_summary(read_json(generated / "pidqa_input_retrieval_seed_sweep.json"))
    e2 = read_json(generated / "qwen8_value_budget_sensitivity_v1.json")
    e3 = read_json(generated / "image_dependence_control_v1.json")
    e4 = read_json(generated / "internvl_tile_budget_v1.json")
    e5 = read_json(generated / "ontology_visibility_effect_v1.json")
    e6 = read_json(generated / "source_seed_resolution_sensitivity_v1.json")
    e7 = read_json(generated / "ontology_mapping_control_v1.json")
    e8 = read_json(generated / "text_only_image_grounding_control_v1.json")

    e2_192 = find_comparison(e2, "e2_192_3072_minus_768_reference", "strict_value_tag_f1", "value")
    e2_512 = find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value")
    e3_correct_shuffled = reverse_orientation(
        find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_value_tag_f1", "value")
    )
    e8_correct_text = find_comparison(e8, "e8_correct_image_minus_text_only_3072", "strict_value_tag_f1", "value")
    e5_768 = find_comparison(e5, "e5_ontology_visible_minus_raw_768", "semantic_correct", "spatial_count")
    e5_3072 = find_comparison(e5, "e5_ontology_visible_minus_raw_3072", "semantic_correct", "spatial_count")
    e7_768 = find_comparison(e7, "e7_correct_legend_minus_permuted_768", "semantic_correct", "spatial_count")
    e7_3072 = find_comparison(e7, "e7_correct_legend_minus_permuted_3072", "semantic_correct", "spatial_count")
    seed29 = find_comparison(e6, "e6_seed29_3072_minus_768", "strict_value_tag_f1", "value")
    seed31 = find_comparison(e6, "e6_seed31_3072_minus_768", "strict_value_tag_f1", "value")

    outputs: list[Path] = []
    outputs.extend(figure_overview(directory, retrieval, e2_512, e3_correct_shuffled, e5_3072))
    outputs.extend(figure_core_effects(directory, retrieval, e2_512, e3_correct_shuffled, e8_correct_text, e5_768, e5_3072, e7_768, e7_3072))
    outputs.extend(figure_tag_stability(directory, e2_192, e2_512, seed29, seed31))
    outputs.extend(figure_calibration_controls(directory, e8, e3, e4))

    sources = [
        "reports/generated/pidqa_input_retrieval_seed_sweep.json",
        "reports/generated/qwen8_value_budget_sensitivity_v1.json",
        "reports/generated/image_dependence_control_v1.json",
        "reports/generated/internvl_tile_budget_v1.json",
        "reports/generated/ontology_visibility_effect_v1.json",
        "reports/generated/source_seed_resolution_sensitivity_v1.json",
        "reports/generated/ontology_mapping_control_v1.json",
        "reports/generated/text_only_image_grounding_control_v1.json",
    ]
    metadata = {
        "status": "pass",
        "generator": "scripts/build_paper_figures_v3.py",
        "source_artifacts": [
            {"path": path, "sha256": sha256(root / path)} for path in sources
        ],
        "files": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "image_generation": "deterministic Matplotlib rendering from frozen numerical artifacts; no generative image model used",
    }
    output = directory / "figure_metadata_v3.json"
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figure_count": len(outputs), "metadata": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
