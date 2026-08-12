"""Render v4 manuscript figures from frozen machine-readable evidence.

All panels are deterministic Matplotlib/Pillow compositions.  Figure 1 uses
two CC0 PIDQA source drawings selected by a pre-declared hash rule; no crop,
bounding box, generative image, or manual case selection is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#5C5C5C",
    "light": "#E8E8E8",
    "dark": "#222222",
    "red_light": "#FBE9E7",
    "green_light": "#E7F4EF",
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


def save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.pdf", directory / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    # Elsevier's combination-artwork guidance requests at least 500 dpi for a
    # raster fallback; the companion PDF remains the preferred vector asset.
    fig.savefig(paths[1], dpi=500, bbox_inches="tight")
    plt.close(fig)
    return paths


def find_comparison(report: dict[str, Any], label: str, metric: str, task: str) -> dict[str, Any]:
    for row in report["comparisons"]:
        if row["comparison"] == label and row["metric"] == metric and row["task"] == task:
            return row
    raise KeyError(f"Missing comparison {label}/{metric}/{task}")


def reverse(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["baseline_mean"] = row["condition_mean"]
    result["condition_mean"] = row["baseline_mean"]
    result["difference_condition_minus_baseline"] = -float(row["difference_condition_minus_baseline"])
    result["source_bootstrap_ci95_low"] = -float(row["source_bootstrap_ci95_high"])
    result["source_bootstrap_ci95_high"] = -float(row["source_bootstrap_ci95_low"])
    return result


def tidy_axis(ax: Any, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=COLORS["light"], linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def draw_interval(ax: Any, row: dict[str, Any], y: float, color: str, *, annotate: bool = True, fontsize: float = 7.0) -> None:
    value = float(row["difference_condition_minus_baseline"])
    low = float(row["source_bootstrap_ci95_low"])
    high = float(row["source_bootstrap_ci95_high"])
    ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", color=color, ecolor=color, markersize=6.5, markeredgecolor="white", markeredgewidth=0.7, capsize=3, linewidth=1.5, zorder=4)
    if annotate:
        offset = 0.012 if value >= 0 else -0.012
        ax.text(value + offset, y + 0.12, f"{value:+.3f} [{low:+.3f}, {high:+.3f}]", ha="left" if value >= 0 else "right", va="bottom", fontsize=fontsize, color=COLORS["dark"])


def protocol_strip(ax: Any) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("Source-disjoint", "drawing IDs"),
        ("Answer-hidden", "scorer store"),
        ("Actual budget", "pixels / tokens"),
        ("Counterfactual", "correct / wrong / none"),
        ("Task interval", "source bootstrap"),
    ]
    positions = np.linspace(0.01, 0.81, len(steps))
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["blue"]]
    for index, ((title, detail), x, color) in enumerate(zip(steps, positions, colors)):
        patch = FancyBboxPatch((x, 0.23), 0.18, 0.55, boxstyle="round,pad=0.008,rounding_size=0.025", facecolor="white", edgecolor=color, linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + 0.09, 0.58, title, ha="center", va="center", fontsize=7.1, fontweight="bold")
        ax.text(x + 0.09, 0.38, detail, ha="center", va="center", fontsize=6.2, color=COLORS["gray"])
        if index < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.18, 0.50), (positions[index + 1] - 0.008, 0.50), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=COLORS["gray"]))


def output_card(ax: Any, title: str, raw: str, tags: list[str], f1: float, exact: int, color: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    background = COLORS["green_light"] if exact else COLORS["red_light"]
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.012,rounding_size=0.03", facecolor=background, edgecolor=color, linewidth=1.3))
    ax.text(0.04, 0.92, title, ha="left", va="top", fontsize=6.4, fontweight="bold", color=color)
    ax.text(0.96, 0.82, f"F1={f1:.2f} | exact={exact}", ha="right", va="top", fontsize=5.3, fontweight="bold")
    wrapped = "\n".join(textwrap.wrap(raw if raw else "[empty]", width=38, break_long_words=False, break_on_hyphens=False))
    ax.text(0.04, 0.74, "Raw: " + wrapped, ha="left", va="top", fontsize=4.15, family="monospace", linespacing=1.00)
    tags_text = ", ".join(tags) if tags else "∅"
    tags_wrapped = "\n".join(textwrap.wrap(tags_text, width=42, break_long_words=False, break_on_hyphens=False))
    ax.text(0.04, 0.045, "Parsed: " + tags_wrapped, ha="left", va="bottom", fontsize=4.05, linespacing=0.98)


def figure_evidence_ladder(root: Path, directory: Path, analysis: dict[str, Any]) -> list[Path]:
    value = analysis["value_evidence_case"]
    structural = analysis["structural_counterexample"]
    fig = plt.figure(figsize=(7.15, 7.65))
    grid = fig.add_gridspec(5, 2, height_ratios=[0.65, 0.42, 2.65, 0.42, 1.55], width_ratios=[0.42, 0.58], hspace=0.22, wspace=0.20)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.04)
    fig.suptitle("From a P&ID score to task-bounded evidence", fontsize=11.3, fontweight="bold")
    protocol_strip(fig.add_subplot(grid[0, :]))

    ax = fig.add_subplot(grid[1, :]); ax.set_axis_off()
    ax.text(0.0, 0.82, "A. Image-grounded value/tag reading", fontsize=9.2, fontweight="bold", va="top")
    ax.text(0.0, 0.22, f"Question: {value['question']}   |   Reference: {', '.join(value['reference_tags'])}", fontsize=7.0, va="bottom")

    ax_img = fig.add_subplot(grid[2, 0])
    ax_img.imshow(Image.open(root / value["image_path"]).convert("RGB"))
    ax_img.set_axis_off()
    ax_img.set_title(f"PIDQA sheet 282 (CC0)\nfull source image; no coordinate crop available", loc="left", fontsize=6.8)
    ax_img.text(0.0, -0.08, f"Deterministic strict rule: 16 eligible; selected minimum hash {value['rank_sha256'][:12]}…", transform=ax_img.transAxes, fontsize=5.8, color=COLORS["gray"], va="top")

    cards = grid[2, 1].subgridspec(2, 2, hspace=0.13, wspace=0.10)
    labels = [
        ("correct_768", "768 correct image (192 cap)", COLORS["gray"]),
        ("correct_3072", "3072 correct image (192 cap)", COLORS["blue"]),
        ("shuffled_3072", "3072 wrong image (192 cap)", COLORS["orange"]),
        ("text_only", "No image (192 cap)", COLORS["purple"]),
    ]
    for cell, (key, title, color) in zip([fig.add_subplot(cards[i, j]) for i in range(2) for j in range(2)], labels):
        item = value["conditions"][key]
        output_card(cell, title, item["raw_output"], item["parsed_tags"], float(item["f1"]), int(item["exact_set"]), color)

    ax = fig.add_subplot(grid[3, :]); ax.set_axis_off()
    ax.text(0.0, 0.82, "B. Structural counterexample: an image does not add value uniformly", fontsize=9.2, fontweight="bold", va="top")
    ax.text(0.0, 0.18, f"Question: {structural['question']}   |   Reference: {structural['reference']}", fontsize=7.0, va="bottom")

    ax_img = fig.add_subplot(grid[4, 0])
    ax_img.imshow(Image.open(root / structural["image_path"]).convert("RGB"))
    ax_img.set_axis_off()
    ax_img.set_title("PIDQA sheet 184 (CC0), full source image", loc="left", fontsize=6.8)
    ax_img.text(0.0, -0.08, f"26 eligible; selected minimum hash {structural['rank_sha256'][:12]}…", transform=ax_img.transAxes, fontsize=5.8, color=COLORS["gray"], va="top")

    ax_cards = fig.add_subplot(grid[4, 1]); ax_cards.set_axis_off(); ax_cards.set_xlim(0, 1); ax_cards.set_ylim(0, 1)
    for x, title, output, correct, color in [
        (0.01, "No image", structural["text_only_output"], True, COLORS["purple"]),
        (0.51, "3072 correct image", structural["correct_3072_output"], False, COLORS["blue"]),
    ]:
        ax_cards.add_patch(FancyBboxPatch((x, 0.20), 0.47, 0.62, boxstyle="round,pad=0.012,rounding_size=0.03", facecolor=COLORS["green_light"] if correct else COLORS["red_light"], edgecolor=color, linewidth=1.4))
        ax_cards.text(x + 0.03, 0.73, title, fontsize=8.0, fontweight="bold", color=color)
        ax_cards.text(x + 0.03, 0.49, f"Verbatim output: {output}", fontsize=7.2, family="monospace")
        ax_cards.text(x + 0.03, 0.29, "strictly correct" if correct else "strictly wrong", fontsize=7.2, fontweight="bold")
    ax_cards.text(0.01, 0.04, "Supported conclusion: visual value is task-specific; correct-image input is not a guarantee of structural accuracy.", fontsize=6.6, color=COLORS["dark"])
    return save(fig, directory, "figure_1_counterfactual_evidence_ladder")


def figure_core_effects(directory: Path, analysis: dict[str, Any], e2: dict[str, Any], e3: dict[str, Any], e5: dict[str, Any], e7: dict[str, Any], e8: dict[str, Any]) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 6.35))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[0.88, 1.38], hspace=0.54, wspace=0.72)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.10)
    fig.suptitle("Core effects: source route, image grounding, and visible context", fontsize=11.1, fontweight="bold")

    ax = fig.add_subplot(grid[0, 0])
    points = analysis["retrieval_points"]
    for index, row in enumerate(points, start=1):
        ax.scatter(float(row["gap"]), index, s=34, color=COLORS["orange"], edgecolor="white", zorder=3)
        ax.text(float(row["gap"]) - 0.006, index, f"{100*float(row['gap']):.1f}", fontsize=6.7, va="center", ha="right")
    gaps = [float(row["gap"]) for row in points]
    mean = sum(gaps) / len(gaps)
    ax.errorbar(mean, 0, xerr=[[mean - min(gaps)], [max(gaps) - mean]], fmt="D", color=COLORS["dark"], capsize=3, markersize=5, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(range(0, len(points) + 1), [f"mean {100*mean:.1f}\nrange {100*min(gaps):.1f}–{100*max(gaps):.1f}"] + [f"seed {row['seed']}" for row in points])
    ax.set_ylim(-0.55, len(points) + 0.55)
    ax.set_xlim(0, max(gaps) + 0.045)
    ax.set_xlabel("Question-random − source-isolated accuracy")
    ax.set_title("A. Five paired split diagnostics", loc="left", fontsize=9.3, fontweight="bold")
    tidy_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    rows = [
        ("3072 − 768\n(512-token cap)", find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value"), COLORS["blue"]),
        ("Correct − shuffled\n(3072, 192-token cap)", reverse(find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_value_tag_f1", "value")), COLORS["orange"]),
        ("Correct − no image\n(3072, 192-token cap)", find_comparison(e8, "e8_correct_image_minus_text_only_3072", "strict_value_tag_f1", "value"), COLORS["purple"]),
    ]
    for y, (_, row, color) in zip([2, 1, 0], rows):
        draw_interval(ax, row, y, color)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks([2, 1, 0], [label for label, _, _ in rows])
    ax.set_ylim(-0.48, 2.55); ax.set_xlim(-0.08, 0.76)
    ax.set_xlabel("Strict value-tag F1 difference")
    ax.set_title("B. Qwen tag reading (n = 100 sources)", loc="left", fontsize=9.3, fontweight="bold")
    tidy_axis(ax)

    ax = fig.add_subplot(grid[1, :])
    context = [
        ("Visible legend − raw, 768", find_comparison(e5, "e5_ontology_visible_minus_raw_768", "semantic_correct", "spatial_count"), COLORS["green"]),
        ("Visible legend − raw, 3072", find_comparison(e5, "e5_ontology_visible_minus_raw_3072", "semantic_correct", "spatial_count"), COLORS["green"]),
        ("Correct labels − permuted, 768", find_comparison(e7, "e7_correct_legend_minus_permuted_768", "semantic_correct", "spatial_count"), COLORS["purple"]),
        ("Correct labels − permuted, 3072", find_comparison(e7, "e7_correct_legend_minus_permuted_3072", "semantic_correct", "spatial_count"), COLORS["purple"]),
    ]
    for y, (_, row, color) in zip([3, 2, 1, 0], context): draw_interval(ax, row, y, color)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks([3, 2, 1, 0], [label for label, _, _ in context])
    ax.set_ylim(-0.62, 3.62); ax.set_xlim(-0.12, 0.48)
    ax.set_xlabel("Semantic spatial-count accuracy difference")
    ax.set_title("C. Visible context changes counts; no mapping-specific advantage was detected", loc="left", fontsize=9.3, fontweight="bold")
    tidy_axis(ax)
    fig.text(0.10, 0.025, "Panels use different task metrics and operating points; magnitudes should not be compared across panels.", fontsize=7.0, color=COLORS["gray"])
    return save(fig, directory, "figure_2_core_effects_v4")


def figure_task_calibration(directory: Path, analysis: dict[str, Any]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 5.25), gridspec_kw={"width_ratios": [0.95, 1.35]})
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.48)
    fig.suptitle("Where the image helps—and where prior pathways remain", fontsize=11.1, fontweight="bold")
    heat = analysis["heatmap_strict_accuracy"]
    tasks = ["connectivity", "count", "spatial_count", "value", "overall"]
    labels = ["Connectivity", "Count", "Spatial count", "Value", "Overall"]
    columns = ["task_prior", "text_only", "shuffled", "correct"]
    matrix = np.array([[next(row for row in heat if row["task"] == task)[column] for column in columns] for task in tasks], dtype=float)
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=0.65, aspect="auto")
    axes[0].set_xticks(range(4), ["Task\nprior", "No\nimage", "Wrong\nimage", "Correct\nimage"])
    axes[0].set_yticks(range(5), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[0].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if matrix[i,j] > 0.40 else COLORS["dark"])
    axes[0].add_patch(Rectangle((-0.48, 2.52), 3.96, 0.96, fill=False, edgecolor=COLORS["orange"], linewidth=2.0))
    axes[0].set_title("A. Strict exact accuracy at 3072 / 192", loc="left", fontsize=9.2, fontweight="bold")
    axes[0].tick_params(axis="both", length=0, labelsize=7.5)

    axes[1].axvline(0, color=COLORS["gray"], linewidth=0.8)
    effects = analysis["task_effects"]
    y_base = {"connectivity": 3, "count": 2, "spatial_count": 1, "value": 0}
    for row in effects:
        y = y_base[row["task"]] + (0.14 if row["contrast"] == "correct_minus_text_only" else -0.14)
        adapted = {
            "difference_condition_minus_baseline": row["difference"],
            "source_bootstrap_ci95_low": row["ci95_low"],
            "source_bootstrap_ci95_high": row["ci95_high"],
        }
        color = COLORS["purple"] if row["contrast"] == "correct_minus_text_only" else COLORS["orange"]
        draw_interval(axes[1], adapted, y, color, annotate=False)
        if row["task"] == "spatial_count" and row["contrast"] == "correct_minus_text_only":
            axes[1].text(row["difference"] - 0.012, y + 0.16, f"{row['difference']:+.2f}", ha="right", fontsize=7.3, fontweight="bold")
    axes[1].set_yticks([3, 2, 1, 0], ["Connectivity\n(strict accuracy)", "Count\n(strict accuracy)", "Spatial count\n(strict accuracy)", "Value\n(strict tag F1)"])
    axes[1].set_xlim(-0.36, 0.68); axes[1].set_ylim(-0.55, 3.55)
    axes[1].set_xlabel("Correct-image minus control\n(95% source-cluster CI)")
    axes[1].set_title("B. Paired task effects (n = 100 each)", loc="left", fontsize=9.2, fontweight="bold")
    tidy_axis(axes[1])
    axes[1].scatter([], [], color=COLORS["purple"], label="Correct − no image")
    axes[1].scatter([], [], color=COLORS["orange"], label="Correct − wrong image")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.text(0.10, 0.025, "The outlined value row is the positive discovery; structural effects retain null or negative image increments.", fontsize=7.0, color=COLORS["gray"])
    return save(fig, directory, "figure_3_task_calibration_v4")


def figure_s1(directory: Path, analysis: dict[str, Any], e3: dict[str, Any], e4: dict[str, Any], e8: dict[str, Any]) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 8.0))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.35, 1.35], hspace=0.62)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.93, bottom=0.06)
    fig.suptitle("Supplementary boundary controls and operating quantities", fontsize=11.0, fontweight="bold")

    ax = fig.add_subplot(grid[0])
    structural = []
    for task in ("connectivity", "count", "spatial_count"):
        structural.append((f"{task.replace('_',' ').title()}: correct − no image", find_comparison(e8, "e8_correct_image_minus_text_only_3072", "strict_correct", task), COLORS["purple"]))
        structural.append((f"{task.replace('_',' ').title()}: correct − wrong image", reverse(find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_correct", task)), COLORS["orange"]))
    positions = list(reversed(range(len(structural))))
    for y, (_, row, color) in zip(positions, structural): draw_interval(ax, row, y, color, fontsize=6.2)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(positions, [label for label, _, _ in structural]); ax.tick_params(axis="y", labelsize=6.7)
    ax.set_xlim(-0.36, 0.16); ax.set_ylim(-0.55, len(structural)-0.45)
    ax.set_xlabel("Strict accuracy difference")
    ax.set_title("S1A. Structural correct-image increments at 3072 / 192", loc="left", fontsize=9.0, fontweight="bold")
    tidy_axis(ax)

    ax = fig.add_subplot(grid[1])
    internvl_rows = [
        ("Strict overall", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "strict_correct", "overall")),
        ("Semantic overall", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "overall")),
        ("Connectivity (semantic)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "connectivity")),
        ("Count (semantic)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "count")),
        ("Spatial count (semantic)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "spatial_count")),
        ("Value tag F1 (strict)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "strict_value_tag_f1", "value")),
    ]
    positions = list(reversed(range(len(internvl_rows))))
    for y, (_, row) in zip(positions, internvl_rows): draw_interval(ax, row, y, COLORS["blue"], fontsize=6.2)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(positions, [label for label, _ in internvl_rows]); ax.tick_params(axis="y", labelsize=6.8)
    ax.set_xlim(-0.25, 0.15); ax.set_ylim(-0.55, len(internvl_rows)-0.45)
    ax.set_xlabel("InternVL 7 tiles − 1 tile")
    ax.set_title("S1B. Corrected InternVL tile-budget boundary (n = 400)", loc="left", fontsize=9.0, fontweight="bold")
    tidy_axis(ax)

    ax = fig.add_subplot(grid[2]); ax.set_axis_off()
    rows = analysis["operating_rows"]
    column_labels = ["Condition", "Actual input", "Tokens\nmean / p95", "Cap rate", "Latency s\nmean / p95", "Peak GiB"]
    table_rows = []
    for row in rows:
        input_text = str(row["input_value"])
        tokens = f"{row['output_token_mean']:.1f} / {row['output_token_p95']:.0f}" if row["output_token_mean"] is not None else "NR"
        cap = f"{100*row['token_cap_rate']:.1f}%" if row["token_cap_rate"] is not None else "NR"
        latency = f"{row['latency_seconds_mean']:.2f} / {row['latency_seconds_p95']:.2f}" if row["latency_seconds_mean"] is not None else "NR"
        peak = f"{row['peak_allocated_gib']:.2f}" if row["peak_allocated_gib"] is not None else "NR"
        table_rows.append([row["label"], input_text, tokens, cap, latency, peak])
    table = ax.table(cellText=table_rows, colLabels=column_labels, cellLoc="center", colLoc="center", loc="center", colWidths=[0.34, 0.18, 0.17, 0.12, 0.18, 0.11])
    table.auto_set_font_size(False); table.set_fontsize(6.3); table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B8B8B8")
        if row == 0: cell.set_facecolor("#E7EEF8"); cell.set_text_props(fontweight="bold")
        elif col == 0: cell.set_text_props(ha="left")
    ax.set_title("S1C. Recorded operating quantities", loc="left", fontsize=9.0, fontweight="bold", pad=14)
    ax.text(0.0, -0.08, "Latency is end-to-end generation time and co-varies with output length; it is not interpreted as visual-encoding cost. NR = not recorded.", transform=ax.transAxes, fontsize=6.6, color=COLORS["gray"], va="top")
    return save(fig, directory, "figure_s1_controls_and_operating_quantities_v4")


def figure_s2(directory: Path, e2: dict[str, Any], e6: dict[str, Any]) -> list[Path]:
    rows = [
        ("Set B, 192-token cap", find_comparison(e2, "e2_192_3072_minus_768_reference", "strict_value_tag_f1", "value"), "primary operating point"),
        ("Set B, 512-token cap", find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value"), "output-cap control"),
        ("Seed 29 partition", find_comparison(e6, "e6_seed29_3072_minus_768", "strict_value_tag_f1", "value"), "descriptive sensitivity"),
        ("Seed 31 partition", find_comparison(e6, "e6_seed31_3072_minus_768", "strict_value_tag_f1", "value"), "descriptive sensitivity"),
    ]
    fig, ax = plt.subplots(figsize=(7.15, 4.55))
    fig.subplots_adjust(left=0.30, right=0.76, top=0.86, bottom=0.19)
    positions = [3, 2, 1, 0]
    for y, (_, row, note) in zip(positions, rows):
        draw_interval(ax, row, y, COLORS["blue"] if "Set B" in _ else COLORS["orange"], fontsize=6.5)
        ax.text(1.04, y, f"{float(row['baseline_mean']):.3f}", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=7.2)
        ax.text(1.22, y, f"{float(row['condition_mean']):.3f}", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=7.2)
        ax.text(1.40, y, note, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, color=COLORS["gray"])
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(positions, [label for label, _, _ in rows])
    ax.set_xlim(-0.08, 0.86); ax.set_ylim(-0.5, 3.48)
    ax.set_xlabel("Strict value-tag F1: 3072 − 768")
    ax.set_title("S2. Tag-reading contrast across caps and descriptive partitions", loc="left", fontsize=10.2, fontweight="bold")
    tidy_axis(ax)
    ax.text(1.04, 1.08, "768 F1", transform=ax.transAxes, ha="center", fontsize=7.0, fontweight="bold")
    ax.text(1.22, 1.08, "3072 F1", transform=ax.transAxes, ha="center", fontsize=7.0, fontweight="bold")
    ax.text(1.40, 1.08, "Role", transform=ax.transAxes, ha="left", fontsize=7.0, fontweight="bold")
    fig.text(0.10, 0.045, "Seed 29 and seed 31 are pre-specified descriptive sensitivity partitions, not independent replications.", fontsize=7.0, color=COLORS["gray"])
    return save(fig, directory, "figure_s2_tag_reading_stability_v4")


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
    e4 = read_json(generated / "internvl_tile_budget_v1.json")
    e5 = read_json(generated / "ontology_visibility_effect_v1.json")
    e6 = read_json(generated / "source_seed_resolution_sensitivity_v1.json")
    e7 = read_json(generated / "ontology_mapping_control_v1.json")
    e8 = read_json(generated / "text_only_image_grounding_control_v1.json")
    outputs: list[Path] = []
    outputs.extend(figure_evidence_ladder(root, directory, analysis))
    outputs.extend(figure_core_effects(directory, analysis, e2, e3, e5, e7, e8))
    outputs.extend(figure_task_calibration(directory, analysis))
    outputs.extend(figure_s1(directory, analysis, e3, e4, e8))
    outputs.extend(figure_s2(directory, e2, e6))
    sources = [
        "reports/generated/editorial_revision_evidence_v4.json",
        "reports/generated/qwen8_value_budget_sensitivity_v1.json",
        "reports/generated/image_dependence_control_v1.json",
        "reports/generated/internvl_tile_budget_v1.json",
        "reports/generated/ontology_visibility_effect_v1.json",
        "reports/generated/source_seed_resolution_sensitivity_v1.json",
        "reports/generated/ontology_mapping_control_v1.json",
        "reports/generated/text_only_image_grounding_control_v1.json",
        "paper/assets/pidqa_sheet_282.jpg",
        "paper/assets/pidqa_sheet_184.jpg",
    ]
    metadata = {
        "status": "pass",
        "generator": "scripts/build_paper_figures_v4.py",
        "source_artifacts": [{"path": path, "sha256": sha256(root / path)} for path in sources],
        "files": [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs],
        "image_policy": "deterministic composition of frozen numerical artifacts and two SHA-selected CC0 PIDQA drawings; no generative image model, crop, or hand-selected box",
    }
    metadata_path = directory / "figure_metadata_v4.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figure_count": len(outputs), "metadata": str(metadata_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
