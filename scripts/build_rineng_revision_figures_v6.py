"""Render the RINENG-focused v6 main figures from frozen and rescored evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from build_paper_figures_v4 import COLORS, draw_interval, find_comparison, read_json, reverse, save, sha256, tidy_axis


def revision_comparison(
    report: dict[str, Any], dataset: str, name: str, estimator: str, metric: str
) -> dict[str, Any]:
    for row in report["datasets"][dataset]["comparisons"]:
        if (
            row["comparison"] == name
            and row["estimator"] == estimator
            and row["metric"] == metric
        ):
            return {
                "difference_condition_minus_baseline": row["difference_condition_minus_baseline"],
                "source_bootstrap_ci95_low": row["source_bootstrap_ci95"][0],
                "source_bootstrap_ci95_high": row["source_bootstrap_ci95"][1],
            }
    raise KeyError(f"Missing {dataset}/{name}/{estimator}/{metric}")


def rounded_box(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    return box


def figure_1(directory: Path, editorial: dict[str, Any]) -> list[Path]:
    fig = plt.figure(figsize=(7.15, 5.25))
    fig.subplots_adjust(0, 0, 1, 1)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.text(
        0.5,
        0.965,
        "A qualification decision for source-conditioned P&ID tag retrieval",
        ha="center",
        va="top",
        fontsize=11.4,
        fontweight="bold",
        color=COLORS["dark"],
    )
    canvas.text(
        0.5,
        0.925,
        "Aggregate accuracy is a diagnostic boundary; a retrieval claim must pass source, evidence, budget, and scope gates.",
        ha="center",
        va="top",
        fontsize=6.9,
        color=COLORS["gray"],
    )

    # Aggregate-score boundary panel.
    rounded_box(canvas, (0.035, 0.565), 0.29, 0.315, facecolor="#F6F6F6", edgecolor="#C9C9C9")
    canvas.text(
        0.055,
        0.842,
        "Why aggregate accuracy\ncannot qualify reading",
        fontsize=7.4,
        fontweight="bold",
        linespacing=1.05,
    )
    aggregate = next(row for row in editorial["heatmap_strict_accuracy"] if row["task"] == "overall")
    labels = ["Task prior", "No image", "Correct image"]
    values = [aggregate["task_prior"], aggregate["text_only"], aggregate["correct"]]
    colors = [COLORS["gray"], COLORS["purple"], COLORS["blue"]]
    bar_ax = fig.add_axes([0.075, 0.655, 0.22, 0.125])
    positions = np.arange(3)
    bar_ax.bar(positions, values, color=colors, width=0.62)
    bar_ax.set_ylim(0, 0.40)
    bar_ax.set_xticks(positions, ["Task\nprior", "No\nimage", "Correct\nimage"], fontsize=6.3)
    bar_ax.set_ylabel("Aggregate strict accuracy", fontsize=6.3)
    bar_ax.tick_params(axis="y", labelsize=6.0)
    bar_ax.grid(axis="y", color=COLORS["light"], linewidth=0.7)
    for spine in ("top", "right", "left"):
        bar_ax.spines[spine].set_visible(False)
    for position, value in zip(positions, values):
        bar_ax.text(position, value + 0.012, f"{value:.3f}", ha="center", fontsize=6.4)
    # Qualification workflow.
    rounded_box(canvas, (0.355, 0.565), 0.61, 0.315, facecolor="#FBFCFD", edgecolor="#B8C8D2")
    canvas.text(0.375, 0.842, "SABER-PID qualification workflow", fontsize=7.8, fontweight="bold")
    steps = [
        ("1  Source", "hold out\ndrawings"),
        ("2  Evidence", "correct / shuffled\n/ no image"),
        ("3  Budget", "common output cap;\nrecord input budget"),
        ("4  Decision", "strict sets +\nallowed claim"),
    ]
    x_positions = [0.38, 0.525, 0.67, 0.815]
    for index, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):
        rounded_box(
            canvas,
            (x, 0.665),
            0.115,
            0.115,
            facecolor=COLORS["green_light"] if index == 3 else "white",
            edgecolor=COLORS["green"] if index == 3 else COLORS["blue"],
            linewidth=1.1,
        )
        canvas.text(x + 0.0575, 0.745, title, ha="center", fontsize=6.8, fontweight="bold")
        canvas.text(x + 0.0575, 0.705, subtitle, ha="center", va="center", fontsize=5.8)
        if index < len(steps) - 1:
            canvas.add_patch(
                FancyArrowPatch(
                    (x + 0.119, 0.722),
                    (x_positions[index + 1] - 0.006, 0.722),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=1.0,
                    color=COLORS["gray"],
                )
            )
    canvas.text(
        0.66,
        0.61,
        "Pass only when requested-drawing and matched-budget contrasts\nsupport the same task-specific interpretation.",
        ha="center",
        fontsize=5.9,
        color=COLORS["gray"],
    )

    # Three evidence/claim cards.
    cards = [
        (
            0.035,
            "Requested-drawing gate  PASS",
            "Correct tag F1       0.5549\nSource-shuffled F1  0.0062\nNo-image F1          0.0000",
            "Correct-minus-controls: +0.5487 / +0.5549",
            COLORS["orange"],
        ),
        (
            0.355,
            "Matched-budget gate  PASS",
            "3072-side F1    0.5253\n768-side F1      0.0003\nCommon cap       512 tokens",
            "High-minus-low: +0.5250 [0.4224, 0.6293]",
            COLORS["blue"],
        ),
        (
            0.675,
            "Scope gate  BOUNDED",
            "Qualified\n  Qwen value-tag candidates within PIDQA\nNot qualified\n  topology, cross-model, or real-plant claims",
            "Qualification is task × model × budget × family",
            COLORS["green"],
        ),
    ]
    for x, title, body, footer, color in cards:
        rounded_box(canvas, (x, 0.175), 0.29, 0.31, facecolor="white", edgecolor=color, linewidth=1.35)
        canvas.text(x + 0.018, 0.448, title, fontsize=7.4, fontweight="bold", color=color)
        canvas.text(x + 0.018, 0.375, body, fontsize=6.5, va="top", linespacing=1.42)
        canvas.text(x + 0.018, 0.202, footer, fontsize=5.9, color=COLORS["gray"])
    canvas.text(
        0.5,
        0.075,
        "Supported output: human- or downstream-verified candidate-tag retrieval—not general P&ID understanding or autonomous acceptance.",
        ha="center",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["dark"],
    )
    return save(fig, directory, "figure_1_qualification_decision_v6")


def figure_2(
    directory: Path,
    e2: dict[str, Any],
    e3: dict[str, Any],
    e6: dict[str, Any],
    e8: dict[str, Any],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.25), gridspec_kw={"width_ratios": [1.05, 1.20]})
    fig.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.20, wspace=0.62)
    fig.suptitle(
        "Requested-drawing dependence and visual-input-budget sensitivity",
        fontsize=11.0,
        fontweight="bold",
    )
    rows = [
        (
            "3072 - 768\n(common 512 cap)",
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
    axes[0].set_title("A. Qualification contrasts", loc="left", fontsize=9.2, fontweight="bold")
    tidy_axis(axes[0])

    stability = [
        ("Set B / 192", find_comparison(e2, "e2_192_3072_minus_768_reference", "strict_value_tag_f1", "value")),
        ("Set B / 512", find_comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value")),
        ("Seed 29", find_comparison(e6, "e6_seed29_3072_minus_768", "strict_value_tag_f1", "value")),
        ("Seed 31", find_comparison(e6, "e6_seed31_3072_minus_768", "strict_value_tag_f1", "value")),
    ]
    for y, (label, row) in zip([3, 2, 1, 0], stability):
        draw_interval(
            axes[1],
            row,
            y,
            COLORS["blue"] if label.startswith("Set B") else COLORS["green"],
            fontsize=6.2,
        )
    axes[1].axvline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_yticks([3, 2, 1, 0], [label for label, _ in stability])
    axes[1].set_xlim(-0.05, 0.86)
    axes[1].set_ylim(-0.55, 3.55)
    axes[1].set_xlabel("3072 - 768 strict tag F1")
    axes[1].set_title("B. Caps and PIDQA partitions", loc="left", fontsize=9.2, fontweight="bold")
    tidy_axis(axes[1])
    fig.text(
        0.18,
        0.055,
        "95% percentile intervals use 10,000 paired source bootstrap replicates. Seed partitions are overlapping within-family sensitivities.",
        fontsize=6.6,
        color=COLORS["gray"],
    )
    return save(fig, directory, "figure_2_qualification_effects_v6")


def figure_3(directory: Path, revision: dict[str, Any]) -> list[Path]:
    methods = [
        ("Qwen", "qwen", COLORS["blue"]),
        ("OCR joined", "paddleocr_geometry", COLORS["orange"]),
        ("Union", "set_union", COLORS["green"]),
        ("Intersection", "set_intersection", COLORS["purple"]),
        ("OCR-first", "ocr_if_nonempty_else_qwen", COLORS["gray"]),
    ]
    set_b = revision["datasets"]["set_b"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.65), gridspec_kw={"width_ratios": [1.02, 1.18]})
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.25, wspace=0.55)
    fig.suptitle(
        "OCR--VLM operating modes and source-disjoint robustness",
        fontsize=10.9,
        fontweight="bold",
    )

    for label, key, color in methods:
        values = set_b["methods"][key]["micro_pooled"]
        axes[0].scatter(values["recall"], values["precision"], s=54, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        offsets = {
            "Qwen": (0.012, -0.035),
            "OCR joined": (0.012, 0.015),
            "Union": (0.012, -0.025),
            "Intersection": (-0.095, 0.015),
            "OCR-first": (0.012, 0.015),
        }
        dx, dy = offsets[label]
        axes[0].text(values["recall"] + dx, values["precision"] + dy, label, fontsize=6.5, color=color)
    axes[0].set_xlim(0.19, 0.76)
    axes[0].set_ylim(0.50, 1.03)
    axes[0].set_xlabel("Micro/pooled recall")
    axes[0].set_ylabel("Micro/pooled precision")
    axes[0].set_title("A. Set B precision--coverage choices", loc="left", fontsize=8.8, fontweight="bold")
    axes[0].grid(color=COLORS["light"], linewidth=0.8)
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)

    comparisons = [
        ("Set B post-hoc\n(n=100)", "set_b", COLORS["green"]),
        ("Seed 29 excl. Set B\n(n=83)", "seed29_excluding_set_b", COLORS["blue"]),
        ("Seed 31 excl. Set B\n(n=83)", "seed31_excluding_set_b", COLORS["blue"]),
        ("Seed 29 strict disjoint\n(n=65)", "seed29_strictly_disjoint", COLORS["orange"]),
        ("Seed 31 strict disjoint\n(n=65)", "seed31_strictly_disjoint", COLORS["orange"]),
    ]
    for y, (_, dataset, color) in zip([4, 3, 2, 1, 0], comparisons):
        row = revision_comparison(revision, dataset, "union_minus_qwen", "micro_pooled", "f1")
        draw_interval(axes[1], row, y, color, fontsize=5.5)
    axes[1].axvline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_yticks([4, 3, 2, 1, 0], [label for label, _, _ in comparisons], fontsize=6.4)
    axes[1].set_xlim(-0.03, 0.15)
    axes[1].set_ylim(-0.6, 4.6)
    axes[1].set_xlabel("Union - Qwen micro/pooled F1")
    axes[1].set_title("B. Rule-frozen source exclusions", loc="left", fontsize=8.8, fontweight="bold")
    tidy_axis(axes[1])

    union_workload = set_b["methods"]["set_union"]["workload_per_source"]
    intersection_workload = set_b["methods"]["set_intersection"]["workload_per_source"]
    fig.text(
        0.11,
        0.085,
        (
            "Set B workload per drawing (median [IQR]): union candidates "
            f"{union_workload['candidate_count']['median']:.0f} "
            f"[{union_workload['candidate_count']['q1']:.0f}, {union_workload['candidate_count']['q3']:.0f}], "
            f"false candidates {union_workload['false_candidate_count']['median']:.0f} "
            f"[{union_workload['false_candidate_count']['q1']:.0f}, {union_workload['false_candidate_count']['q3']:.0f}]; "
            f"intersection candidates {intersection_workload['candidate_count']['median']:.0f} "
            f"[{intersection_workload['candidate_count']['q1']:.0f}, {intersection_workload['candidate_count']['q3']:.0f}]."
        ),
        fontsize=6.3,
        color=COLORS["gray"],
    )
    fig.text(
        0.11,
        0.045,
        "All rules are answer-independent. The 83-source checks retain 18 mutual overlaps; the two 65-source subsets share no sources with Set B or each other.",
        fontsize=6.2,
        color=COLORS["gray"],
    )
    return save(fig, directory, "figure_3_operating_modes_v6")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    directory = root / args.output_dir
    generated = root / "reports/generated"
    editorial = read_json(generated / "editorial_revision_evidence_v4.json")
    e2 = read_json(generated / "qwen8_value_budget_sensitivity_v1.json")
    e3 = read_json(generated / "image_dependence_control_v1.json")
    e6 = read_json(generated / "source_seed_resolution_sensitivity_v1.json")
    e8 = read_json(generated / "text_only_image_grounding_control_v1.json")
    revision = read_json(generated / "rineng_revision_analysis_v6.json")

    outputs: list[Path] = []
    outputs.extend(figure_1(directory, editorial))
    outputs.extend(figure_2(directory, e2, e3, e6, e8))
    outputs.extend(figure_3(directory, revision))
    source_paths = [
        "reports/generated/editorial_revision_evidence_v4.json",
        "reports/generated/qwen8_value_budget_sensitivity_v1.json",
        "reports/generated/image_dependence_control_v1.json",
        "reports/generated/source_seed_resolution_sensitivity_v1.json",
        "reports/generated/text_only_image_grounding_control_v1.json",
        "reports/generated/rineng_revision_analysis_v6.json",
    ]
    metadata = {
        "status": "pass",
        "version": "rineng-revision-figures-v6",
        "generator": "scripts/build_rineng_revision_figures_v6.py",
        "source_artifacts": [
            {"path": path, "sha256": sha256(root / path)} for path in source_paths
        ],
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "image_policy": (
            "deterministic Matplotlib composition of frozen machine-readable evidence; "
            "no generative image model and no outcome-selected drawing"
        ),
    }
    metadata_path = directory / "figure_metadata_v6.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "pass", "figure_count": len(outputs), "metadata": str(metadata_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
