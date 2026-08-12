"""Render the active SABER-PID paper figures in a compact publication style.

The builder is inference-free.  It reads only validated machine-readable
reports, preserves every plotted value, and replaces the nine active V9 figure
assets with V10 vector/raster pairs designed at the final double-column width.
It also performs a renderer-level text collision and clipping audit before any
asset is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from build_paper_figures_v4 import find_comparison, reverse  # noqa: E402
from build_rineng_overnight_v7_paper_artifacts import (  # noqa: E402
    DATASET_COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    PROMPT_MARKERS,
    comparison_index,
)
from build_rineng_revision_figures_v6 import revision_comparison  # noqa: E402
from build_rineng_v8_extension_figures import (  # noqa: E402
    POOLED,
    QUALITY_ORDER,
    find_internvl,
    find_quality,
)
from build_rineng_v9_editorial_assets import read_cost_grid  # noqa: E402


WIDTH = 7.15
COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "magenta": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#666666",
    "mid_gray": "#A8A8A8",
    "light_gray": "#E8E8E8",
    "pale_blue": "#EAF3F8",
    "pale_green": "#E7F4EF",
    "pale_orange": "#FAEEE8",
    "pale_magenta": "#F7ECF3",
}
METHODS = {
    "set_intersection": ("Intersection", COLORS["magenta"]),
    "paddleocr_geometry": ("OCR joined", COLORS["green"]),
    "ocr_if_nonempty_else_qwen": ("OCR-first", COLORS["blue"]),
    "set_union": ("Union", COLORS["orange"]),
}
QUALITY_LABELS = {
    "clean": "Clean",
    "jpeg_q70": "JPEG Q70",
    "blur_r1": "Blur r=1",
    "downsample_s075": "0.75x restore",
}


def configure_style() -> None:
    """Use a uniform, editable, color-blind-safe journal style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_title(ax: Any, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=7, fontweight="bold")


def clean_axis(ax: Any, *, grid_axis: str = "x", zero: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=COLORS["light_gray"], linewidth=0.65)
    if zero:
        ax.axvline(0, color=COLORS["gray"], linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["black"])
    ax.spines["bottom"].set_color(COLORS["black"])


def interval_mark(
    ax: Any,
    y: float,
    point: float,
    low: float,
    high: float,
    color: str,
    *,
    marker: str = "o",
    annotate: bool = False,
) -> None:
    ax.errorbar(
        point,
        y,
        xerr=[[point - low], [high - point]],
        fmt=marker,
        markersize=5.7,
        markeredgecolor="white",
        markeredgewidth=0.7,
        color=color,
        ecolor=color,
        elinewidth=1.25,
        capsize=2.7,
        zorder=3,
    )
    if annotate:
        ax.annotate(
            f"{point:+.3f}",
            (point, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color=COLORS["black"],
        )


def audit_text_layout(fig: Any, name: str) -> dict[str, Any]:
    """Fail on clipped or materially overlapping rendered text boxes."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    texts: list[tuple[str, Any]] = []
    clipped: list[str] = []
    for artist in fig.findobj(match=Text):
        value = artist.get_text().strip()
        if not value or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        label = " ".join(value.split())
        texts.append((label, box))
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            clipped.append(label)

    collisions: list[dict[str, Any]] = []
    for index, (left_text, left_box) in enumerate(texts):
        for right_text, right_box in texts[index + 1 :]:
            x0 = max(left_box.x0, right_box.x0)
            y0 = max(left_box.y0, right_box.y0)
            x1 = min(left_box.x1, right_box.x1)
            y1 = min(left_box.y1, right_box.y1)
            if x1 <= x0 + 0.8 or y1 <= y0 + 0.8:
                continue
            area = (x1 - x0) * (y1 - y0)
            smaller = min(left_box.width * left_box.height, right_box.width * right_box.height)
            if smaller > 0 and area / smaller >= 0.08:
                collisions.append(
                    {
                        "left": left_text,
                        "right": right_text,
                        "overlap_fraction": round(area / smaller, 4),
                    }
                )
    if clipped or collisions:
        raise ValueError(
            f"{name}: text layout audit failed; clipped={clipped}; collisions={collisions}"
        )
    return {"figure": name, "text_count": len(texts), "clipped": [], "collisions": []}


def save_figure(
    fig: Any,
    root: Path,
    stem: str,
    audits: list[dict[str, Any]],
) -> list[Path]:
    audits.append(audit_text_layout(fig, stem))
    base = root / "paper" / "figures" / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    metadata = {"Creator": "SABER-PID deterministic V10 figure builder"}
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03, metadata=metadata)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [pdf, png]


def build_overview(
    root: Path,
    summary: dict[str, Any],
    cost: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    quality = {row["quality"]: row for row in summary["quality"]}
    quality_effects = [
        quality[key]["correct_minus_shuffled"]["value_f1_difference"]
        for key in QUALITY_ORDER
    ]
    internvl = next(
        row
        for row in summary["internvl_budget54"]
        if row["dataset"] == "pooled_three_source_disjoint_subsets"
    )
    dexpi = {row["condition"]: row for row in summary["dexpi_external"]["conditions"]}

    fig = plt.figure(figsize=(WIDTH, 3.65))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.92, 1.08, 0.72],
        left=0.04,
        right=0.985,
        top=0.97,
        bottom=0.07,
        hspace=0.46,
    )

    ax = fig.add_subplot(grid[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.08, "a  Source-isolated qualification pipeline", fontweight="bold", va="top")
    steps = [
        ("Isolate", "unseen drawings"),
        ("Intervene", "correct / shuffled / none"),
        ("Match", "visual + output budgets"),
        ("Transfer", "quality / model / family"),
        ("Operate", "mode selected by cost"),
    ]
    positions = np.linspace(0.075, 0.925, len(steps))
    for index in range(len(steps) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (positions[index] + 0.035, 0.50),
                (positions[index + 1] - 0.035, 0.50),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color=COLORS["mid_gray"],
                zorder=1,
            )
        )
    for index, (x, (heading, detail)) in enumerate(zip(positions, steps), start=1):
        active = index == len(steps)
        color = COLORS["green"] if active else COLORS["blue"]
        fill = COLORS["pale_green"] if active else COLORS["pale_blue"]
        # A marker is sized in display points, so the step badges stay circular
        # even though this intentionally wide flow-diagram axis is not square.
        ax.scatter([x], [0.50], s=225, facecolor=fill, edgecolor=color, linewidth=1.0, zorder=2)
        ax.text(x, 0.50, str(index), ha="center", va="center", fontweight="bold", color=color, zorder=3)
        ax.text(x, 0.79, heading, ha="center", va="center", fontweight="bold")
        ax.text(x, 0.19, detail, ha="center", va="center", fontsize=7.2, color=COLORS["gray"])

    ax = fig.add_subplot(grid[1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.08, "b  Evidence retained across controlled transfers", fontweight="bold", va="top")
    evidence = [
        ("PIDQA primary", "+0.549", "correct - shuffled", "100 sources", COLORS["blue"]),
        (
            "Mild quality shifts",
            f"{min(quality_effects):.3f}-{max(quality_effects):.3f}",
            "requested-drawing effect",
            "230 sources",
            COLORS["orange"],
        ),
        (
            "InternVL budget match",
            f"+{internvl['correct_minus_shuffled']['value_f1_difference']:.3f}",
            "correct - shuffled",
            "54 x 448 tiles",
            COLORS["green"],
        ),
        (
            "Public DEXPI",
            f"+{dexpi['correct']['f1'] - dexpi['shuffled']['f1']:.3f}",
            "correct - shuffled",
            "35 images / 26 cases",
            COLORS["magenta"],
        ),
    ]
    for index, (heading, value, measure, scope, color) in enumerate(evidence):
        x0 = index * 0.25
        if index:
            ax.plot([x0, x0], [0.08, 0.91], color=COLORS["light_gray"], linewidth=0.8)
        ax.plot([x0 + 0.025, x0 + 0.075], [0.82, 0.82], color=color, linewidth=2.2, solid_capstyle="round")
        ax.text(x0 + 0.025, 0.66, heading, fontweight="bold", va="center")
        ax.text(x0 + 0.025, 0.43, value, fontsize=10.0, fontweight="bold", color=COLORS["black"], va="center")
        ax.text(x0 + 0.025, 0.25, measure, fontsize=7.2, color=COLORS["gray"], va="center")
        ax.text(x0 + 0.025, 0.09, scope, fontsize=7.2, color=COLORS["gray"], va="bottom")

    ax = fig.add_subplot(grid[2])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.10, r"c  Cost-aware operating rule ($r=C_{FN}/C_{FP}$)", fontweight="bold", va="top")
    intervals = cost["exact_decision_intervals"]
    fills = [COLORS["pale_magenta"], COLORS["pale_green"], COLORS["pale_blue"], COLORS["pale_orange"]]
    for index, (row, fill) in enumerate(zip(intervals, fills)):
        x0 = 0.012 + index * 0.244
        ax.add_patch(Rectangle((x0, 0.22), 0.232, 0.48, facecolor=fill, edgecolor=COLORS["mid_gray"], linewidth=0.7))
        ax.text(x0 + 0.116, 0.53, row["recommended_mode"], ha="center", va="center", fontweight="bold")
        lower = float(row["lower_ratio_inclusive"])
        upper = row["upper_ratio_exclusive"]
        if lower == 0:
            label = f"r < {float(upper):.3f}"
        elif upper == "infinity":
            label = f"r >= {lower:.3f}"
        else:
            label = f"{lower:.3f} <= r < {float(upper):.3f}"
        ax.text(x0 + 0.116, 0.34, label, ha="center", va="center", fontsize=7.2, color=COLORS["gray"])
    ax.text(0.99, 0.02, "Fixed TP / FP / FN; only the loss ratio changes.", ha="right", va="bottom", fontsize=7.0, color=COLORS["gray"])
    return save_figure(fig, root, "figure_1_saber_pid_overview_v10", audits)


def build_quality(
    root: Path,
    report: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    direct = [find_quality(report, quality, "correct_minus_shuffled") for quality in QUALITY_ORDER]
    degraded = [quality for quality in QUALITY_ORDER if quality != "clean"]
    changes = [
        find_quality(report, quality, "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1")
        for quality in degraded
    ]
    internvl = find_internvl(report, "correct_minus_shuffled")

    fig = plt.figure(figsize=(WIDTH, 4.55))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.86],
        left=0.225,
        right=0.985,
        top=0.96,
        bottom=0.13,
        hspace=0.57,
        wspace=0.47,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    palette = [COLORS["blue"], COLORS["orange"], COLORS["magenta"], COLORS["green"]]
    y = np.arange(len(direct))[::-1]
    for position, row, color in zip(y, direct, palette):
        point = float(row["value_f1_difference"])
        low, high = map(float, row["value_f1_source_bootstrap_ci95"])
        interval_mark(ax_a, position, point, low, high, color, annotate=True)
    ax_a.set_yticks(y, [QUALITY_LABELS[key] for key in QUALITY_ORDER])
    ax_a.set_xlim(-0.02, 0.67)
    ax_a.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_a.set_ylim(-0.5, len(y) - 0.5)
    ax_a.set_xlabel("Strict tag F1: correct - shuffled")
    panel_title(ax_a, "a", "Requested-drawing effect")
    clean_axis(ax_a, zero=True)

    y = np.arange(len(changes))[::-1]
    for position, row, color in zip(y, changes, palette[1:]):
        point = float(row["value_f1_difference_in_differences"])
        low, high = map(float, row["source_bootstrap_ci95"])
        interval_mark(ax_b, position, point, low, high, color, annotate=True)
    ax_b.set_yticks(y, [QUALITY_LABELS[key] for key in degraded])
    ax_b.set_xlim(-0.075, 0.06)
    ax_b.set_ylim(-0.5, len(y) - 0.5)
    ax_b.set_xlabel("Change in effect relative to clean")
    panel_title(ax_b, "b", "Paired degradation change")
    clean_axis(ax_b, zero=True)

    qwen = direct[0]
    rows = [
        (
            "Qwen3-VL-8B\n35.98M tensor elements",
            float(qwen["value_f1_difference"]),
            *map(float, qwen["value_f1_source_bootstrap_ci95"]),
            COLORS["blue"],
        ),
        (
            "InternVL3.5-8B\n32.51M tensor elements",
            float(internvl["value_f1_difference"]),
            *map(float, internvl["value_f1_source_bootstrap_ci95"]),
            COLORS["green"],
        ),
    ]
    for position, (_label, point, low, high, color) in zip([1, 0], rows):
        interval_mark(ax_c, position, point, low, high, color, annotate=True)
    ax_c.set_yticks([1, 0], [row[0] for row in rows])
    ax_c.set_xlim(-0.02, 0.67)
    ax_c.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_c.set_ylim(-0.45, 1.45)
    ax_c.set_xlabel("Strict tag F1: correct - shuffled")
    panel_title(ax_c, "c", "Closest safe visual-budget comparison")
    clean_axis(ax_c, zero=True)
    return save_figure(fig, root, "figure_2_quality_and_budget_v10", audits)


def build_dexpi(
    root: Path,
    report: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    conditions = ["correct", "shuffled", "text_only", "paddleocr_full_image"]
    labels = ["Qwen correct", "Qwen shuffled", "Qwen no image", "PaddleOCR"]
    condition_colors = [COLORS["blue"], COLORS["orange"], COLORS["magenta"], COLORS["green"]]
    metrics = report["metrics"]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 3.20),
        gridspec_kw={"width_ratios": [1.02, 1.05]},
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.84, bottom=0.18, wspace=0.48)

    metric_specs = [
        ("precision", "Precision", "o", 0.17),
        ("recall", "Recall", "s", 0.0),
        ("f1", "F1", "D", -0.17),
    ]
    y = np.arange(len(conditions))[::-1]
    for position, condition, color in zip(y, conditions, condition_colors):
        for metric, _label, marker, offset in metric_specs:
            value = float(metrics[condition][metric])
            ax_a.scatter(value, position + offset, marker=marker, s=32, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        f1 = float(metrics[condition]["f1"])
        ax_a.annotate(f"{f1:.2f}", (f1, position - 0.17), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.1)
    ax_a.set_yticks(y, labels)
    ax_a.set_xlim(-0.02, 1.08)
    ax_a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_ylim(-0.55, len(y) - 0.45)
    ax_a.set_xlabel("Tag-set metric")
    ax_a.set_title("a  Public DEXPI operating points", loc="left", pad=28, fontweight="bold")
    clean_axis(ax_a)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=COLORS["gray"], markeredgecolor="white", markersize=5.5, label=label)
            for _metric, label, marker, _offset in metric_specs
        ],
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.00),
        handletextpad=0.3,
        columnspacing=0.8,
    )

    comparison_order = [
        "correct_minus_shuffled",
        "correct_minus_text_only",
        "correct_minus_paddleocr_full_image",
        "ocr_minus_text_only",
    ]
    comparison_labels = ["Correct - shuffled", "Correct - no image", "Correct - OCR", "OCR - no image"]
    comparison_colors = [COLORS["orange"], COLORS["magenta"], COLORS["green"], COLORS["gray"]]
    y = np.arange(len(comparison_order))[::-1]
    for position, key, color in zip(y, comparison_order, comparison_colors):
        row = report["comparisons"][key]
        point = float(row["difference"])
        low, high = map(float, row["ci95"])
        interval_mark(ax_b, position, point, low, high, color, annotate=True)
    ax_b.set_yticks(y, comparison_labels)
    ax_b.set_xlim(-0.03, 1.05)
    ax_b.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_ylim(-0.55, len(y) - 0.45)
    ax_b.set_xlabel("Paired logical-case F1 difference")
    ax_b.set_title("b  Evidence contrasts", loc="left", pad=28, fontweight="bold")
    clean_axis(ax_b, zero=True)
    return save_figure(fig, root, "figure_3_dexpi_external_v10", audits)


def build_cost_aware(
    root: Path,
    revision: dict[str, Any],
    cost: dict[str, Any],
    grid: dict[str, dict[str, np.ndarray]],
    audits: list[dict[str, Any]],
) -> list[Path]:
    fig = plt.figure(figsize=(WIDTH, 5.05))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.94, 1.08], width_ratios=[0.86, 1.34])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.11, hspace=0.48, wspace=0.40)

    method_rows = revision["datasets"]["set_b"]["methods"]
    scatter = [
        ("qwen", "Qwen [3]", COLORS["gray"], "o", (-42, 0)),
        ("paddleocr_geometry", "OCR joined [2]", COLORS["green"], "s", (8, 10)),
        ("ocr_if_nonempty_else_qwen", "OCR-first [2]", COLORS["blue"], "D", (9, -14)),
        ("set_union", "Union [4]", COLORS["orange"], "^", (-45, 7)),
        ("set_intersection", "Intersection [1]", COLORS["magenta"], "v", (8, -3)),
    ]
    for method, label, color, marker, offset in scatter:
        values = method_rows[method]["micro_pooled"]
        ax_a.scatter(values["recall"], values["precision"], s=38, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax_a.annotate(label, (values["recall"], values["precision"]), xytext=offset, textcoords="offset points", fontsize=7.0, color=COLORS["black"])
    ax_a.set_xlim(0.18, 0.77)
    ax_a.set_ylim(0.52, 1.02)
    ax_a.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax_a.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_xlabel("Recall")
    ax_a.set_ylabel("Precision")
    panel_title(ax_a, "a", "Operating points [median candidates]")
    clean_axis(ax_a, grid_axis="both")

    decision = cost["exact_decision_intervals"]
    for method_id, (label, color) in METHODS.items():
        values = grid[method_id]
        ax_b.plot(values["ratio"], values["loss"], color=color, linewidth=0.9, alpha=0.38)
    for row in decision:
        method_id = row["method_id"]
        label, color = METHODS[method_id]
        values = grid[method_id]
        lower = max(float(row["lower_ratio_inclusive"]), float(values["ratio"][0]))
        upper_value = row["upper_ratio_exclusive"]
        upper = float(values["ratio"][-1]) if upper_value == "infinity" else float(upper_value)
        mask = (values["ratio"] >= lower) & (values["ratio"] <= upper)
        ax_b.plot(values["ratio"][mask], values["loss"][mask], color=color, linewidth=2.5, solid_capstyle="round", label=label)
    for row in decision[:-1]:
        ax_b.axvline(float(row["upper_ratio_exclusive"]), color=COLORS["mid_gray"], linewidth=0.75, linestyle="--")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.05, 20.0)
    ax_b.set_ylim(0.10, 60.0)
    ax_b.set_xticks([0.1, 1.0, 10.0])
    ax_b.set_yticks([0.1, 1.0, 10.0])
    ax_b.set_xlabel(r"Relative miss cost, $C_{FN}/C_{FP}$")
    ax_b.set_ylabel("Error cost per drawing")
    panel_title(ax_b, "b", "Exact lower-loss envelope")
    ax_b.grid(True, which="major", color=COLORS["light_gray"], linewidth=0.65)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)
    handles, labels = ax_b.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_b.legend(unique.values(), unique.keys(), frameon=False, ncol=2, loc="upper left", handlelength=2.2, columnspacing=0.9)

    for method_id, (label, color) in METHODS.items():
        values = grid[method_id]
        ax_c.plot(values["ratio"], values["probability"], color=color, linewidth=1.65, label=label)
    for row in decision[:-1]:
        ax_c.axvline(float(row["upper_ratio_exclusive"]), color=COLORS["mid_gray"], linewidth=0.75, linestyle="--")
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.05, 20.0)
    ax_c.set_xticks([0.1, 1.0, 10.0])
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlabel(r"Relative miss cost, $C_{FN}/C_{FP}$")
    ax_c.set_ylabel("Probability of minimum loss")
    panel_title(ax_c, "c", "Source-bootstrap decision stability (10,000 resamples)")
    ax_c.grid(True, which="major", color=COLORS["light_gray"], linewidth=0.65)
    for spine in ("top", "right"):
        ax_c.spines[spine].set_visible(False)
    # Panel b already carries the shared method key; repeating it here would
    # crowd the panel heading at final print size.
    return save_figure(fig, root, "figure_4_cost_aware_operation_v10", audits)


def build_boundary_controls(
    root: Path,
    analysis: dict[str, Any],
    e3: dict[str, Any],
    e4: dict[str, Any],
    e8: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    fig = plt.figure(figsize=(WIDTH, 7.15))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.05, 0.96], hspace=0.50)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.91, bottom=0.055)

    ax_a = fig.add_subplot(grid[0])
    structural: list[tuple[str, dict[str, Any], str, str]] = []
    for task in ("connectivity", "count", "spatial_count"):
        label = task.replace("_", " ").title()
        structural.append((f"{label}: correct - no image", find_comparison(e8, "e8_correct_image_minus_text_only_3072", "strict_correct", task), COLORS["magenta"], "o"))
        structural.append((f"{label}: correct - shuffled", reverse(find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_correct", task)), COLORS["orange"], "s"))
    y = np.arange(len(structural))[::-1]
    for position, (_label, row, color, marker) in zip(y, structural):
        interval_mark(
            ax_a,
            position,
            float(row["difference_condition_minus_baseline"]),
            float(row["source_bootstrap_ci95_low"]),
            float(row["source_bootstrap_ci95_high"]),
            color,
            marker=marker,
        )
    ax_a.set_yticks(y, [row[0] for row in structural])
    ax_a.set_xlim(-0.37, 0.17)
    ax_a.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1])
    ax_a.set_ylim(-0.55, len(y) - 0.45)
    ax_a.set_xlabel("Strict accuracy difference")
    ax_a.set_title("a  Structural correct-image controls", loc="left", pad=28, fontweight="bold")
    clean_axis(ax_a, zero=True)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["magenta"], markeredgecolor="white", label="No-image control"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["orange"], markeredgecolor="white", label="Shuffled-image control"),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.01),
        handletextpad=0.3,
        columnspacing=1.0,
    )

    ax_b = fig.add_subplot(grid[1])
    internvl_rows = [
        ("Strict overall", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "strict_correct", "overall")),
        ("Semantic overall", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "overall")),
        ("Connectivity (semantic)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "connectivity")),
        ("Count (semantic)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "count")),
        ("Spatial count (semantic)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "spatial_count")),
        ("Value-tag F1 (strict)", find_comparison(e4, "e4_high_minus_low_actual_tile_budget", "strict_value_tag_f1", "value")),
    ]
    y = np.arange(len(internvl_rows))[::-1]
    for position, (_label, row) in zip(y, internvl_rows):
        interval_mark(
            ax_b,
            position,
            float(row["difference_condition_minus_baseline"]),
            float(row["source_bootstrap_ci95_low"]),
            float(row["source_bootstrap_ci95_high"]),
            COLORS["blue"],
        )
    ax_b.set_yticks(y, [row[0] for row in internvl_rows])
    ax_b.set_xlim(-0.25, 0.15)
    ax_b.set_ylim(-0.55, len(y) - 0.45)
    ax_b.set_xlabel("InternVL: 7 tiles - 1 tile")
    panel_title(ax_b, "b", "Corrected tile-budget boundary (n=400)")
    clean_axis(ax_b, zero=True)

    ax_c = fig.add_subplot(grid[2])
    ax_c.set_axis_off()
    panel_title(ax_c, "c", "Recorded operating quantities")
    rows = analysis["operating_rows"]
    columns = ["Condition", "Actual input", "Tokens\nmean / p95", "Cap rate", "Latency (s)\nmean / p95", "Peak GiB"]
    table_rows = []
    for row in rows:
        tokens = f"{row['output_token_mean']:.1f} / {row['output_token_p95']:.0f}" if row["output_token_mean"] is not None else "NR"
        cap = f"{100 * row['token_cap_rate']:.1f}%" if row["token_cap_rate"] is not None else "NR"
        latency = f"{row['latency_seconds_mean']:.2f} / {row['latency_seconds_p95']:.2f}" if row["latency_seconds_mean"] is not None else "NR"
        peak = f"{row['peak_allocated_gib']:.2f}" if row["peak_allocated_gib"] is not None else "NR"
        input_value = row["input_value"]
        if isinstance(input_value, int):
            input_text = f"{input_value:,}"
        else:
            parts = [part.strip() for part in str(input_value).split("/")]
            input_text = " / ".join(f"{int(part):,}" if part.isdigit() else part for part in parts)
        table_rows.append([row["label"], input_text, tokens, cap, latency, peak])
    table = ax_c.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.31, 0.17, 0.16, 0.11, 0.17, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    table.scale(1.0, 1.42)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor(COLORS["mid_gray"])
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor(COLORS["pale_blue"])
            cell.set_text_props(fontweight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F7F7F7")
        if column_index == 0 and row_index > 0:
            cell.set_text_props(ha="left")
    return save_figure(fig, root, "figure_s1_boundary_controls_v10", audits)


def build_qualification_effects(
    root: Path,
    e2: dict[str, Any],
    e3: dict[str, Any],
    e6: dict[str, Any],
    e8: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 3.30), gridspec_kw={"width_ratios": [1.02, 1.10]})
    fig.subplots_adjust(left=0.17, right=0.985, top=0.92, bottom=0.19, wspace=0.54)
    rows = [
        ("3072 - 768\n(common 512 cap)", find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value"), COLORS["blue"]),
        ("Correct - shuffled\n(3072 / 192)", reverse(find_comparison(e3, "e3_shuffled_minus_correct_3072", "strict_value_tag_f1", "value")), COLORS["orange"]),
        ("Correct - no image\n(3072 / 192)", find_comparison(e8, "e8_correct_image_minus_text_only_3072", "strict_value_tag_f1", "value"), COLORS["magenta"]),
    ]
    for position, (_label, row, color) in zip([2, 1, 0], rows):
        interval_mark(ax_a, position, float(row["difference_condition_minus_baseline"]), float(row["source_bootstrap_ci95_low"]), float(row["source_bootstrap_ci95_high"]), color, annotate=True)
    ax_a.set_yticks([2, 1, 0], [row[0] for row in rows])
    ax_a.set_xlim(-0.05, 0.74)
    ax_a.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_a.set_ylim(-0.5, 2.5)
    ax_a.set_xlabel("Strict value-tag F1 difference")
    panel_title(ax_a, "a", "Qualification contrasts")
    clean_axis(ax_a, zero=True)

    stability = [
        ("Set B / 192", find_comparison(e2, "e2_192_3072_minus_768_reference", "strict_value_tag_f1", "value"), COLORS["blue"]),
        ("Set B / 512", find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value"), COLORS["blue"]),
        ("Seed 29", find_comparison(e6, "e6_seed29_3072_minus_768", "strict_value_tag_f1", "value"), COLORS["green"]),
        ("Seed 31", find_comparison(e6, "e6_seed31_3072_minus_768", "strict_value_tag_f1", "value"), COLORS["green"]),
    ]
    for position, (_label, row, color) in zip([3, 2, 1, 0], stability):
        interval_mark(ax_b, position, float(row["difference_condition_minus_baseline"]), float(row["source_bootstrap_ci95_low"]), float(row["source_bootstrap_ci95_high"]), color, annotate=True)
    ax_b.set_yticks([3, 2, 1, 0], [row[0] for row in stability])
    ax_b.set_xlim(-0.05, 0.86)
    ax_b.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_b.set_ylim(-0.5, 3.5)
    ax_b.set_xlabel("Strict tag F1: 3072 - 768")
    panel_title(ax_b, "b", "Caps and PIDQA partitions")
    clean_axis(ax_b, zero=True)
    return save_figure(fig, root, "figure_s2_qualification_effects_v10", audits)


def build_operating_modes(
    root: Path,
    revision: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 3.35), gridspec_kw={"width_ratios": [0.93, 1.13]})
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.18, wspace=0.55)
    set_b = revision["datasets"]["set_b"]
    methods = [
        ("qwen", "Qwen", COLORS["gray"], "o", (-38, 0)),
        ("paddleocr_geometry", "OCR joined", COLORS["green"], "s", (7, 9)),
        ("ocr_if_nonempty_else_qwen", "OCR-first", COLORS["blue"], "D", (8, -12)),
        ("set_union", "Union", COLORS["orange"], "^", (-31, 7)),
        ("set_intersection", "Intersection", COLORS["magenta"], "v", (7, -3)),
    ]
    for key, label, color, marker, offset in methods:
        values = set_b["methods"][key]["micro_pooled"]
        ax_a.scatter(values["recall"], values["precision"], s=38, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax_a.annotate(label, (values["recall"], values["precision"]), xytext=offset, textcoords="offset points", fontsize=7.0)
    ax_a.set_xlim(0.18, 0.77)
    ax_a.set_ylim(0.52, 1.02)
    ax_a.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax_a.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_xlabel("Recall")
    ax_a.set_ylabel("Precision")
    panel_title(ax_a, "a", "Set B precision-coverage choices")
    clean_axis(ax_a, grid_axis="both")

    comparisons = [
        ("Set B post-hoc\n(n=100)", "set_b", COLORS["green"]),
        ("Seed 29 excl. Set B\n(n=83)", "seed29_excluding_set_b", COLORS["blue"]),
        ("Seed 31 excl. Set B\n(n=83)", "seed31_excluding_set_b", COLORS["blue"]),
        ("Seed 29 strict disjoint\n(n=65)", "seed29_strictly_disjoint", COLORS["orange"]),
        ("Seed 31 strict disjoint\n(n=65)", "seed31_strictly_disjoint", COLORS["orange"]),
    ]
    for position, (_label, dataset, color) in zip([4, 3, 2, 1, 0], comparisons):
        row = revision_comparison(revision, dataset, "union_minus_qwen", "micro_pooled", "f1")
        interval_mark(ax_b, position, float(row["difference_condition_minus_baseline"]), float(row["source_bootstrap_ci95_low"]), float(row["source_bootstrap_ci95_high"]), color, annotate=True)
    ax_b.set_yticks([4, 3, 2, 1, 0], [row[0] for row in comparisons])
    ax_b.set_xlim(-0.03, 0.15)
    ax_b.set_xticks([-0.02, 0.0, 0.04, 0.08, 0.12])
    ax_b.set_ylim(-0.55, 4.55)
    ax_b.set_xlabel("Union - Qwen pooled F1")
    panel_title(ax_b, "b", "Rule-frozen source exclusions")
    clean_axis(ax_b, zero=True)
    return save_figure(fig, root, "figure_s3_operating_modes_v10", audits)


def build_cross_model(
    root: Path,
    score: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    comparisons = comparison_index(score)
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.10), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.88, bottom=0.09, hspace=0.30, wspace=0.16)
    ordered = [(dataset, prompt) for dataset in DATASET_ORDER for prompt in ("p0", "p1")]
    y = np.arange(len(ordered))[::-1]
    y_labels = [f"{DATASET_LABELS[dataset]} / {prompt.upper()}" for dataset, prompt in ordered]
    for model_index, model in enumerate(MODEL_ORDER):
        for control_index, control in enumerate(("shuffled", "text_only")):
            ax = axes[model_index, control_index]
            for position, (dataset, prompt) in zip(y, ordered):
                row = comparisons[(model, dataset, prompt, f"correct_minus_{control}")]
                point = float(row["value_f1_difference"])
                low, high = map(float, row["value_f1_source_bootstrap_ci95"])
                interval_mark(ax, position, point, low, high, DATASET_COLORS[dataset], marker=PROMPT_MARKERS[prompt])
            ax.set_xlim(-0.04, 0.84)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6])
            ax.set_ylim(-0.55, len(y) - 0.45)
            ax.set_yticks(y)
            if control_index == 0:
                ax.set_yticklabels(y_labels)
                ax.set_ylabel(MODEL_LABELS[model], fontweight="bold", labelpad=13)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            clean_axis(ax, zero=True)
            if model_index == 0:
                panel_title(ax, "a" if control_index == 0 else "b", "Correct - shuffled" if control == "shuffled" else "Correct - no image")
            if model_index == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Strict value-tag F1 difference")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=DATASET_COLORS[dataset], markeredgecolor="white", markersize=5.7, label=DATASET_LABELS[dataset])
        for dataset in DATASET_ORDER
    ] + [
        Line2D([0], [0], marker=PROMPT_MARKERS[prompt], linestyle="none", color=COLORS["gray"], markersize=5.3, label=prompt.upper())
        for prompt in ("p0", "p1")
    ]
    fig.legend(handles=legend, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.61, 0.985), handletextpad=0.35, columnspacing=1.0)
    return save_figure(fig, root, "figure_s4_cross_model_replication_v10", audits)


def build_prompt_sensitivity(
    root: Path,
    score: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[Path]:
    rows = sorted(score["prompt_sensitivity"], key=lambda row: (MODEL_ORDER.index(row["model"]), DATASET_ORDER.index(row["dataset"])))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.85))
    fig.subplots_adjust(left=0.31, right=0.985, top=0.91, bottom=0.16)
    colors = {"qwen3vl8b": COLORS["blue"], "qwen3vl32b": COLORS["orange"], "internvl35_8b": COLORS["green"]}
    y = np.arange(len(rows))[::-1]
    labels = []
    for position, row in zip(y, rows):
        point = float(row["difference"])
        low, high = map(float, row["source_bootstrap_ci95"])
        interval_mark(ax, position, point, low, high, colors[row["model"]])
        labels.append(f"{MODEL_LABELS[row['model']]} / {DATASET_LABELS[row['dataset']]}")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.55, len(y) - 0.45)
    ax.set_xlim(-0.12, 0.17)
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Strict value-tag F1: P1 - P0")
    panel_title(ax, "a", "Frozen-prompt sensitivity")
    clean_axis(ax, zero=True)
    for boundary in (5.5, 2.5):
        ax.axhline(boundary, color=COLORS["light_gray"], linewidth=0.7)
    return save_figure(fig, root, "figure_s5_prompt_sensitivity_v10", audits)


def ensure_passing(paths: Iterable[tuple[str, dict[str, Any]]]) -> None:
    failed = [name for name, document in paths if document.get("status") != "pass"]
    if failed:
        raise ValueError(f"All source reports must pass before plotting: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--metadata", default="paper/figures/figure_metadata_v10.json")
    parser.add_argument("--report", default="reports/generated/rineng_v10_publication_figures.json")
    parser.add_argument("--layout-audit", default="reports/generated/rineng_figure_layout_audit_v10.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    configure_style()

    source_names = [
        "reports/generated/editorial_revision_evidence_v4.json",
        "reports/generated/image_dependence_control_v1.json",
        "reports/generated/internvl_tile_budget_v1.json",
        "reports/generated/text_only_image_grounding_control_v1.json",
        "reports/generated/qwen8_value_budget_sensitivity_v1.json",
        "reports/generated/source_seed_resolution_sensitivity_v1.json",
        "reports/generated/rineng_revision_analysis_v6.json",
        "reports/generated/rineng_overnight_v7_score.json",
        "reports/generated/rineng_v8_extension_score.json",
        "reports/generated/rineng_v8_dexpi_external_score.json",
        "reports/generated/rineng_v8_paper_summary.json",
        "reports/generated/rineng_cost_sensitive_operating_modes_v8.json",
        "reports/generated/rineng_cost_sensitive_operating_modes_v8.csv",
    ]
    documents = {name: read_json(root / name) for name in source_names if name.endswith(".json")}
    ensure_passing(documents.items())
    grid = read_cost_grid(root / "reports/generated/rineng_cost_sensitive_operating_modes_v8.csv")

    audits: list[dict[str, Any]] = []
    outputs: list[Path] = []
    outputs += build_overview(root, documents["reports/generated/rineng_v8_paper_summary.json"], documents["reports/generated/rineng_cost_sensitive_operating_modes_v8.json"], audits)
    outputs += build_quality(root, documents["reports/generated/rineng_v8_extension_score.json"], audits)
    outputs += build_dexpi(root, documents["reports/generated/rineng_v8_dexpi_external_score.json"], audits)
    outputs += build_cost_aware(root, documents["reports/generated/rineng_revision_analysis_v6.json"], documents["reports/generated/rineng_cost_sensitive_operating_modes_v8.json"], grid, audits)
    outputs += build_boundary_controls(
        root,
        documents["reports/generated/editorial_revision_evidence_v4.json"],
        documents["reports/generated/image_dependence_control_v1.json"],
        documents["reports/generated/internvl_tile_budget_v1.json"],
        documents["reports/generated/text_only_image_grounding_control_v1.json"],
        audits,
    )
    outputs += build_qualification_effects(
        root,
        documents["reports/generated/qwen8_value_budget_sensitivity_v1.json"],
        documents["reports/generated/image_dependence_control_v1.json"],
        documents["reports/generated/source_seed_resolution_sensitivity_v1.json"],
        documents["reports/generated/text_only_image_grounding_control_v1.json"],
        audits,
    )
    outputs += build_operating_modes(root, documents["reports/generated/rineng_revision_analysis_v6.json"], audits)
    outputs += build_cross_model(root, documents["reports/generated/rineng_overnight_v7_score.json"], audits)
    outputs += build_prompt_sensitivity(root, documents["reports/generated/rineng_overnight_v7_score.json"], audits)

    layout_report = {
        "version": "rineng-figure-layout-audit-v10",
        "status": "pass",
        "audit_scope": "all nine active manuscript and supplementary figures",
        "figure_count": len(audits),
        "figures": audits,
    }
    layout_path = root / args.layout_audit
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(json.dumps(layout_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = {
        "version": "rineng-publication-figures-v10",
        "status": "pass",
        "policy": "deterministic vector-first rendering of validated reports; no inference or generative imagery",
        "design": {
            "target_width_inches": WIDTH,
            "font_family": "Arial/Helvetica/DejaVu Sans fallback",
            "normal_text_points": 8.0,
            "panel_title_points": 8.5,
            "palette": "Okabe-Ito-derived color-blind-safe",
            "figure_titles_inside_artwork": False,
            "text_collision_audit": "pass",
        },
        "sources": {name: sha256(root / name) for name in source_names},
        "figures": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    metadata_path = root / args.metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "version": "rineng-v10-publication-figure-build",
        "status": "pass",
        "figure_count": len(outputs) // 2,
        "asset_count": len(outputs),
        "metadata": args.metadata,
        "layout_audit": args.layout_audit,
        "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
    }
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figures": len(outputs) // 2, "report": args.report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
