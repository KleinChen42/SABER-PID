"""Build paper-ready V7 tables, figures, and a machine-readable result digest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#5C5C5C",
    "light": "#E8E8E8",
    "dark": "#222222",
}
MODEL_ORDER = ("qwen3vl8b", "qwen3vl32b", "internvl35_8b")
MODEL_LABELS = {
    "qwen3vl8b": "Qwen3-VL-8B",
    "qwen3vl32b": "Qwen3-VL-32B",
    "internvl35_8b": "InternVL3.5-8B",
}
DATASET_ORDER = ("set_b100", "seed29_strict65", "seed31_strict65")
DATASET_LABELS = {
    "set_b100": "Set B (100)",
    "seed29_strict65": "Seed 29 (65)",
    "seed31_strict65": "Seed 31 (65)",
}
DATASET_COLORS = {
    "set_b100": COLORS["blue"],
    "seed29_strict65": COLORS["orange"],
    "seed31_strict65": COLORS["green"],
}
PROMPT_MARKERS = {"p0": "o", "p1": "s"}
CONTROLS = ("shuffled", "text_only")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.pdf", directory / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=500, bbox_inches="tight")
    plt.close(fig)
    return paths


def cell_index(score: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for cell_id, cell in score["cells"].items():
        result[tuple(cell_id.split("|"))] = cell
    return result


def comparison_index(score: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["model"], row["dataset"], row["prompt"], row["contrast"]): row
        for row in score["counterfactual_comparisons"]
    }


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def interval_text(point: float, interval: list[float]) -> str:
    return f"{point:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]"


def build_flat_rows(
    cells: dict[tuple[str, str, str, str], dict[str, Any]],
    comparisons: dict[tuple[str, str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counterfactual_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for prompt in ("p0", "p1"):
                correct = cells[(model, dataset, prompt, "correct")]["metrics"]
                task_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "prompt": prompt,
                        "overall_strict_accuracy": correct["strict_accuracy"],
                        "connectivity_accuracy": correct["task"]["connectivity"]["strict_accuracy"],
                        "count_accuracy": correct["task"]["count"]["strict_accuracy"],
                        "spatial_count_accuracy": correct["task"]["spatial_count"]["strict_accuracy"],
                        "value_exact_accuracy": correct["task"]["value"]["strict_accuracy"],
                        "value_tag_f1": correct["strict_value_tags"]["f1"],
                    }
                )
                for control in CONTROLS:
                    control_metrics = cells[(model, dataset, prompt, control)]["metrics"]
                    comparison = comparisons[(model, dataset, prompt, f"correct_minus_{control}")]
                    counterfactual_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "prompt": prompt,
                            "control": control,
                            "correct_value_f1": correct["strict_value_tags"]["f1"],
                            "control_value_f1": control_metrics["strict_value_tags"]["f1"],
                            "correct_minus_control_value_f1": comparison["value_f1_difference"],
                            "ci95_low": comparison["value_f1_source_bootstrap_ci95"][0],
                            "ci95_high": comparison["value_f1_source_bootstrap_ci95"][1],
                            "correct_minus_control_source_macro_accuracy": comparison[
                                "source_macro_strict_accuracy_difference"
                            ],
                            "source_macro_accuracy_ci95_low": comparison["source_bootstrap_ci95"][0],
                            "source_macro_accuracy_ci95_high": comparison["source_bootstrap_ci95"][1],
                        }
                    )
    return counterfactual_rows, task_rows


def build_counterfactual_tex(
    path: Path,
    cells: dict[tuple[str, str, str, str], dict[str, Any]],
    comparisons: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Frozen V7 value-tag counterfactual matrix. Effects are correct-image minus control pooled strict tag F1; brackets are 95\% percentile intervals from 10,000 paired source bootstrap replicates.}",
        r"\label{tab:v7_counterfactual}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Model & Subset & Prompt & Correct F1 & Shuffled F1 / effect [CI] & No-image F1 / effect [CI] \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for prompt in ("p0", "p1"):
                correct = cells[(model, dataset, prompt, "correct")]["metrics"]["strict_value_tags"]["f1"]
                shuffled = cells[(model, dataset, prompt, "shuffled")]["metrics"]["strict_value_tags"]["f1"]
                text_only = cells[(model, dataset, prompt, "text_only")]["metrics"]["strict_value_tags"]["f1"]
                shuffled_row = comparisons[(model, dataset, prompt, "correct_minus_shuffled")]
                text_row = comparisons[(model, dataset, prompt, "correct_minus_text_only")]
                lines.append(
                    " & ".join(
                        [
                            MODEL_LABELS[model],
                            DATASET_LABELS[dataset],
                            prompt.upper(),
                            f"{correct:.3f}",
                            f"{shuffled:.3f} / {interval_text(shuffled_row['value_f1_difference'], shuffled_row['value_f1_source_bootstrap_ci95'])}",
                            f"{text_only:.3f} / {interval_text(text_row['value_f1_difference'], text_row['value_f1_source_bootstrap_ci95'])}",
                        ]
                    )
                    + r" \\"
                )
        if model != MODEL_ORDER[-1]:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_task_tex(path: Path, task_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Correct-image strict accuracy by task under the frozen V7 configurations. Value F1 is pooled strict value-tag F1 and is not interchangeable with exact-set accuracy.}",
        r"\label{tab:v7_tasks}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Model & Subset & Prompt & Overall & Connectivity & Count & Spatial count & Value exact & Value F1 \\",
        r"\midrule",
    ]
    for row in task_rows:
        lines.append(
            " & ".join(
                [
                    MODEL_LABELS[row["model"]],
                    DATASET_LABELS[row["dataset"]],
                    row["prompt"].upper(),
                    f"{row['overall_strict_accuracy']:.3f}",
                    f"{row['connectivity_accuracy']:.3f}",
                    f"{row['count_accuracy']:.3f}",
                    f"{row['spatial_count_accuracy']:.3f}",
                    f"{row['value_exact_accuracy']:.3f}",
                    f"{row['value_tag_f1']:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def figure_counterfactual(
    directory: Path,
    comparisons: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 7.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.88, bottom=0.13, hspace=0.34, wspace=0.14)
    fig.suptitle(
        "Requested-drawing value-tag signal across models, subsets, and frozen prompts",
        fontsize=10.8,
        fontweight="bold",
    )
    ordered_cells = [(dataset, prompt) for dataset in DATASET_ORDER for prompt in ("p0", "p1")]
    y_positions = list(reversed(range(len(ordered_cells))))
    y_labels = [f"{DATASET_LABELS[dataset]} / {prompt.upper()}" for dataset, prompt in ordered_cells]
    for model_index, model in enumerate(MODEL_ORDER):
        for control_index, control in enumerate(CONTROLS):
            ax = axes[model_index, control_index]
            for y, (dataset, prompt) in zip(y_positions, ordered_cells):
                row = comparisons[(model, dataset, prompt, f"correct_minus_{control}")]
                point = float(row["value_f1_difference"])
                low, high = (float(value) for value in row["value_f1_source_bootstrap_ci95"])
                ax.errorbar(
                    point,
                    y,
                    xerr=[[point - low], [high - point]],
                    fmt=PROMPT_MARKERS[prompt],
                    markersize=5.7,
                    markeredgewidth=0.7,
                    markeredgecolor="white",
                    color=DATASET_COLORS[dataset],
                    ecolor=DATASET_COLORS[dataset],
                    elinewidth=1.35,
                    capsize=2.5,
                    zorder=3,
                )
            ax.axvline(0, color=COLORS["gray"], linewidth=0.9, zorder=1)
            ax.grid(axis="x", color=COLORS["light"], linewidth=0.75, zorder=0)
            ax.set_xlim(-0.035, 0.84)
            ax.set_ylim(-0.6, 5.6)
            ax.set_yticks(y_positions)
            if control_index == 0:
                ax.set_yticklabels(y_labels)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            ax.tick_params(axis="both", labelsize=6.4)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            if control_index == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=7.5, fontweight="bold", labelpad=12)
            if model_index == 0:
                title = "Correct minus shuffled" if control == "shuffled" else "Correct minus no image"
                ax.set_title(title, fontsize=8.4, fontweight="bold")
            if model_index == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Strict value-tag F1 difference", fontsize=7.2)
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DATASET_COLORS[dataset], markeredgecolor="white", markersize=6, label=DATASET_LABELS[dataset])
        for dataset in DATASET_ORDER
    ] + [
        Line2D([0], [0], marker=PROMPT_MARKERS[prompt], color=COLORS["gray"], linestyle="none", markersize=5.5, label=prompt.upper())
        for prompt in ("p0", "p1")
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.60, 0.925), ncol=5, frameon=False, fontsize=6.5)
    fig.text(
        0.23,
        0.047,
        "All 36 pre-specified intervals have positive lower bounds. Whiskers: 95% percentile intervals from 10,000 paired source bootstrap replicates.",
        fontsize=6.4,
        color=COLORS["gray"],
    )
    return save_figure(fig, directory, "figure_v7_cross_model_counterfactual_replication")


def figure_prompt_sensitivity(directory: Path, score: dict[str, Any]) -> list[Path]:
    rows = sorted(
        score["prompt_sensitivity"],
        key=lambda row: (MODEL_ORDER.index(row["model"]), DATASET_ORDER.index(row["dataset"])),
    )
    fig, ax = plt.subplots(figsize=(7.15, 4.65))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.84, bottom=0.18)
    fig.suptitle("Frozen-prompt sensitivity of correct-image value-tag F1", fontsize=10.5, fontweight="bold")
    y_positions = list(reversed(range(len(rows))))
    labels: list[str] = []
    model_colors = {
        "qwen3vl8b": COLORS["blue"],
        "qwen3vl32b": COLORS["orange"],
        "internvl35_8b": COLORS["green"],
    }
    for y, row in zip(y_positions, rows):
        point = float(row["difference"])
        low, high = (float(value) for value in row["source_bootstrap_ci95"])
        ax.errorbar(
            point,
            y,
            xerr=[[point - low], [high - point]],
            fmt="o",
            color=model_colors[row["model"]],
            ecolor=model_colors[row["model"]],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6,
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
        labels.append(f"{MODEL_LABELS[row['model']]} / {DATASET_LABELS[row['dataset']]}")
    ax.axvline(0, color=COLORS["gray"], linewidth=0.9)
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.75)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("P1 minus P0 strict value-tag F1", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=6.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.30,
        0.065,
        "Eight of nine 95% source-bootstrap intervals include zero; both prompts are retained without best-prompt selection.",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    return save_figure(fig, directory, "figure_s_v7_prompt_sensitivity")


def build_summary(
    counterfactual_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    score: dict[str, Any],
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model in MODEL_ORDER:
        correct_values = [row["value_tag_f1"] for row in task_rows if row["model"] == model]
        effects = [row for row in counterfactual_rows if row["model"] == model]
        model_summary: dict[str, Any] = {
            "correct_value_f1": {
                "minimum": min(correct_values),
                "median": statistics.median(correct_values),
                "maximum": max(correct_values),
            }
        }
        for control in CONTROLS:
            subset = [row for row in effects if row["control"] == control]
            points = [row["correct_minus_control_value_f1"] for row in subset]
            model_summary[f"correct_minus_{control}"] = {
                "comparison_count": len(subset),
                "positive_interval_count": sum(row["ci95_low"] > 0 for row in subset),
                "minimum_effect": min(points),
                "median_effect": statistics.median(points),
                "maximum_effect": max(points),
                "minimum_ci95_lower_bound": min(row["ci95_low"] for row in subset),
            }
        by_model[model] = model_summary
    sensitivities = score["prompt_sensitivity"]
    return {
        "version": "rineng-overnight-v7-paper-summary",
        "status": "pass",
        "scope": {
            "models": 3,
            "datasets": 3,
            "prompts": 2,
            "conditions": 3,
            "cells": 54,
            "prediction_rows": 16_560,
            "counterfactual_comparisons": 36,
        },
        "primary_result": {
            "positive_interval_count": sum(row["ci95_low"] > 0 for row in counterfactual_rows),
            "comparison_count": len(counterfactual_rows),
            "interpretation": "Requested-drawing dependence is directionally replicated, with useful magnitude concentrated in Qwen3-VL.",
        },
        "prompt_sensitivity": {
            "intervals_including_zero": sum(
                row["source_bootstrap_ci95"][0] <= 0 <= row["source_bootstrap_ci95"][1]
                for row in sensitivities
            ),
            "comparison_count": len(sensitivities),
            "selection": "Both frozen prompts reported; no best-prompt selection.",
        },
        "models": by_model,
        "boundary": score["boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--score", default="reports/generated/rineng_overnight_v7_score.json")
    parser.add_argument("--validation", default="reports/generated/rineng_overnight_v7_validation.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    score_path = root / args.score
    validation_path = root / args.validation
    score = read_json(score_path)
    validation = read_json(validation_path)
    if score.get("status") != "pass" or validation.get("status") != "pass":
        raise SystemExit("Refusing to build paper artifacts from an unvalidated score report")

    cells = cell_index(score)
    comparisons = comparison_index(score)
    counterfactual_rows, task_rows = build_flat_rows(cells, comparisons)
    generated = root / "reports/generated"
    figures = root / "paper/figures"
    tables = root / "paper/tables"
    write_csv(generated / "rineng_overnight_v7_counterfactual_table.csv", counterfactual_rows)
    write_csv(generated / "rineng_overnight_v7_task_table.csv", task_rows)
    write_csv(generated / "rineng_overnight_v7_prompt_sensitivity.csv", score["prompt_sensitivity"])
    build_counterfactual_tex(tables / "table_rineng_overnight_v7_counterfactual.tex", cells, comparisons)
    build_task_tex(tables / "table_rineng_overnight_v7_task_accuracy.tex", task_rows)
    figure_paths = figure_counterfactual(figures, comparisons)
    figure_paths += figure_prompt_sensitivity(figures, score)
    summary = build_summary(counterfactual_rows, task_rows, score)
    summary["inputs"] = {
        "score": {"path": args.score, "sha256": sha256(score_path)},
        "validation": {"path": args.validation, "sha256": sha256(validation_path)},
    }
    summary_path = generated / "rineng_overnight_v7_paper_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "version": "rineng-overnight-v7-figure-metadata",
        "status": "pass",
        "source_score_sha256": sha256(score_path),
        "source_validation_sha256": sha256(validation_path),
        "figures": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in figure_paths
        ],
    }
    metadata_path = figures / "figure_metadata_v7.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "figures": len(figure_paths),
                "tables": 2,
                "summary": summary_path.relative_to(root).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
