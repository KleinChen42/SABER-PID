"""Build manuscript figures directly from frozen machine-readable evidence.

The script creates code-drawn, publication-resolution PNG and vector PDF
figures. It does not use generative image synthesis and does not alter any
experiment artifact.
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
    "gray": "#595959",
    "lightgray": "#E8E8E8",
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


def comparison(report: dict[str, Any], label: str, metric: str, task: str) -> dict[str, Any]:
    for row in report["comparisons"]:
        if row["comparison"] == label and row["metric"] == metric and row["task"] == task:
            return row
    raise KeyError(f"Missing {label}/{metric}/{task}")


def save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [directory / f"{stem}.pdf", directory / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=400, bbox_inches="tight")
    plt.close(fig)
    return outputs


def box(ax: Any, xy: tuple[float, float], text: str, *, color: str, width: float = 0.22) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, 0.16,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white", edgecolor=color, linewidth=1.7,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.08, text, ha="center", va="center", fontsize=10, wrap=True)


def arrow(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.35, color=COLORS["gray"]))


def figure_protocol(directory: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box(ax, (0.04, 0.66), "PIDQA sources\n500 drawings", color=COLORS["blue"])
    box(ax, (0.31, 0.66), "Source-disjoint split\n+ answer-isolated inputs", color=COLORS["green"])
    box(ax, (0.58, 0.66), "Frozen VLM inference\nactual input/output budgets", color=COLORS["orange"])
    box(ax, (0.81, 0.66), "Immutable raw\ngenerations", color=COLORS["purple"], width=0.15)
    arrow(ax, (0.26, 0.74), (0.31, 0.74))
    arrow(ax, (0.53, 0.74), (0.58, 0.74))
    arrow(ax, (0.80, 0.74), (0.81, 0.74))
    box(ax, (0.16, 0.28), "Controls\nimage shuffle; ontology visibility", color=COLORS["orange"], width=0.25)
    box(ax, (0.47, 0.28), "Deterministic scoring\nstrict + semantic", color=COLORS["blue"], width=0.22)
    box(ax, (0.75, 0.28), "Source-cluster bootstrap\nclaims + limits", color=COLORS["green"], width=0.20)
    arrow(ax, (0.51, 0.66), (0.29, 0.44))
    arrow(ax, (0.69, 0.66), (0.58, 0.44))
    arrow(ax, (0.41, 0.36), (0.47, 0.36))
    arrow(ax, (0.69, 0.36), (0.75, 0.36))
    ax.text(0.5, 0.95, "Evidence path for source-isolated P&ID VLM evaluation", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.08, "Controls diagnose what a score can mean; they do not turn a task-specific result into a universal reasoning claim.", ha="center", va="center", fontsize=9.5, color=COLORS["gray"])
    return save(fig, directory, "figure_1_evaluation_protocol")


def point_with_ci(ax: Any, value: float, low: float, high: float, label: str, color: str) -> None:
    ax.errorbar(value, 0, xerr=[[value - low], [high - value]], fmt="o", markersize=8,
                capsize=4, linewidth=2, color=color, mec="white", mew=0.8)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.25, 0.25)
    ax.set_title(label, fontsize=10, loc="left", fontweight="bold")
    ax.text(value, 0.12, f"{value:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="x", color=COLORS["lightgray"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)


def figure_effects(directory: Path, e2: dict[str, Any], e3: dict[str, Any], e4: dict[str, Any], e5: dict[str, Any]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), constrained_layout=True)
    panels = [
        (axes[0, 0], comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value"), "E2: value tag F1\n3072 − 768 at 512 tokens", COLORS["blue"]),
        (axes[0, 1], comparison(e3, "e3_shuffled_minus_correct_3072", "strict_value_tag_f1", "value"), "E3: value tag F1\nshuffled − correct image", COLORS["orange"]),
        (axes[1, 0], comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "overall"), "E4: semantic overall\n7 tiles − 1 tile", COLORS["purple"]),
    ]
    for ax, row, label, color in panels:
        point_with_ci(ax, row["difference_condition_minus_baseline"], row["source_bootstrap_ci95_low"], row["source_bootstrap_ci95_high"], label, color)
        ax.set_xlabel("Paired source-level difference")
    ax = axes[1, 1]
    rows = [
        comparison(e5, "e5_ontology_visible_minus_raw_768", "semantic_correct", "spatial_count"),
        comparison(e5, "e5_ontology_visible_minus_raw_3072", "semantic_correct", "spatial_count"),
    ]
    labels = ["768", "3072"]
    values = [row["difference_condition_minus_baseline"] for row in rows]
    lower = [value - row["source_bootstrap_ci95_low"] for value, row in zip(values, rows)]
    upper = [row["source_bootstrap_ci95_high"] - value for value, row in zip(values, rows)]
    ax.bar(labels, values, color=COLORS["green"], width=0.58)
    ax.errorbar(labels, values, yerr=[lower, upper], fmt="none", ecolor=COLORS["gray"], capsize=4, linewidth=1.6)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    for x, value in enumerate(values):
        ax.text(x, value + 0.024, f"{value:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("E5: semantic spatial-count\nontology-visible − raw", fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel("Paired source-level difference")
    ax.grid(axis="y", color=COLORS["lightgray"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Controls separate image dependence, ontology visibility, and model-specific budget effects", fontsize=14, fontweight="bold")
    return save(fig, directory, "figure_2_controlled_effects")


def figure_seed_sensitivity(directory: Path, e6: dict[str, Any]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    tasks = ["connectivity", "count", "spatial_count", "value"]
    display = ["Connectivity", "Count", "Spatial count", "Value"]
    positions = list(range(len(tasks)))
    offsets = [-0.16, 0.16]
    for seed, offset, color in [(29, offsets[0], COLORS["blue"]), (31, offsets[1], COLORS["orange"])]:
        values, lows, highs = [], [], []
        for task in tasks:
            row = comparison(e6, f"e6_seed{seed}_3072_minus_768", "strict_correct", task)
            values.append(row["difference_condition_minus_baseline"])
            lows.append(row["source_bootstrap_ci95_low"])
            highs.append(row["source_bootstrap_ci95_high"])
        axes[0].errorbar(
            [pos + offset for pos in positions], values,
            yerr=[[value - low for value, low in zip(values, lows)], [high - value for value, high in zip(values, highs)]],
            fmt="o", markersize=7, capsize=4, linewidth=1.8, color=color, label=f"Seed {seed}",
        )
    axes[0].axhline(0, color=COLORS["gray"], linewidth=0.8)
    axes[0].set_xticks(positions, display, rotation=18, ha="right")
    axes[0].set_ylabel("Strict accuracy: 3072 − 768")
    axes[0].set_title("Task-level source-split sensitivity", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=COLORS["lightgray"], linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)

    values, lows, highs = [], [], []
    for seed in (29, 31):
        row = comparison(e6, f"e6_seed{seed}_3072_minus_768", "strict_value_tag_f1", "value")
        values.append(row["difference_condition_minus_baseline"])
        lows.append(row["source_bootstrap_ci95_low"])
        highs.append(row["source_bootstrap_ci95_high"])
    axes[1].bar(["Seed 29", "Seed 31"], values, color=[COLORS["blue"], COLORS["orange"]], width=0.58)
    axes[1].errorbar(["Seed 29", "Seed 31"], values, yerr=[[value - low for value, low in zip(values, lows)], [high - value for value, high in zip(values, highs)]], fmt="none", ecolor=COLORS["gray"], capsize=4, linewidth=1.6)
    axes[1].axhline(0, color=COLORS["gray"], linewidth=0.8)
    for x, value in enumerate(values):
        axes[1].text(x, value + 0.026, f"{value:+.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_ylabel("Strict value tag F1: 3072 − 768")
    axes[1].set_title("Value/tag effect remains directional", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color=COLORS["lightgray"], linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pre-specified source-split sensitivity is reported separately, not pooled", fontsize=14, fontweight="bold")
    return save(fig, directory, "figure_3_source_split_sensitivity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    generated = root / "reports" / "generated"
    directory = root / args.output_dir
    e2 = read_json(generated / "qwen8_value_budget_sensitivity_v1.json")
    e3 = read_json(generated / "image_dependence_control_v1.json")
    e4 = read_json(generated / "internvl_tile_budget_v1.json")
    e5 = read_json(generated / "ontology_visibility_effect_v1.json")
    e6 = read_json(generated / "source_seed_resolution_sensitivity_v1.json")
    outputs = []
    outputs.extend(figure_protocol(directory))
    outputs.extend(figure_effects(directory, e2, e3, e4, e5))
    outputs.extend(figure_seed_sensitivity(directory, e6))
    metadata = {
        "status": "pass",
        "generator": "scripts/build_paper_figures_v2.py",
        "source_artifacts": [
            "reports/generated/qwen8_value_budget_sensitivity_v1.json",
            "reports/generated/image_dependence_control_v1.json",
            "reports/generated/internvl_tile_budget_v1.json",
            "reports/generated/ontology_visibility_effect_v1.json",
            "reports/generated/source_seed_resolution_sensitivity_v1.json",
        ],
        "files": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "image_generation": "deterministic matplotlib rendering from frozen numerical artifacts; no generative image model used",
    }
    output = directory / "figure_metadata_v2.json"
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figure_count": len(outputs), "metadata": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
