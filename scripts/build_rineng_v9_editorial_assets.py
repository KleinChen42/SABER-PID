"""Build acceptance-oriented RINENG V9 figures and compact main tables.

The assets are deterministic compositions of already validated V8 score
reports.  This script performs no model inference and does not change a
prediction.  It promotes the qualification-to-decision story while retaining
all boundary analyses in the supplementary material and public release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#666666",
    "light": "#E6E6E6",
    "dark": "#222222",
}

METHOD_STYLE = {
    "set_intersection": ("Intersection", "#CC79A7"),
    "paddleocr_geometry": ("OCR joined", "#009E73"),
    "ocr_if_nonempty_else_qwen": ("OCR-first", "#0072B2"),
    "set_union": ("Union", "#D55E00"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=500, bbox_inches="tight")
    plt.close(fig)
    return outputs


def rounded_box(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = "#B8C8D2",
    linewidth: float = 1.0,
    radius: float = 0.018,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def pooled_internvl(extension: dict[str, Any], contrast: str) -> dict[str, Any]:
    return next(
        row
        for row in extension["internvl_comparisons"]
        if row["dataset"] == "pooled_three_source_disjoint_subsets"
        and row["contrast"] == contrast
    )


def build_overview(
    output_dir: Path,
    paper_summary: dict[str, Any],
    cost: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    quality = {row["quality"]: row for row in paper_summary["quality"]}
    internvl = next(
        row
        for row in paper_summary["internvl_budget54"]
        if row["dataset"] == "pooled_three_source_disjoint_subsets"
    )
    dexpi = {
        row["condition"]: row for row in paper_summary["dexpi_external"]["conditions"]
    }
    decision = cost["exact_decision_intervals"]

    fig = plt.figure(figsize=(7.15, 4.55))
    fig.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.965,
        "SABER-PID: from requested-drawing evidence to an engineering operating decision",
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.5,
        0.918,
        "Qualify the task and budget first; choose the precision--coverage mode second.",
        ha="center",
        va="top",
        fontsize=7.0,
        color=COLORS["gray"],
    )

    steps = [
        ("1  Isolate", "unseen source\ndrawings"),
        ("2  Intervene", "correct / shuffled\n/ no image"),
        ("3  Match", "recorded visual\nand output budgets"),
        ("4  Transfer", "quality, model,\nand drawing family"),
        ("5  Operate", "cost-aware\ntransparent mode"),
    ]
    x_positions = [0.045, 0.235, 0.425, 0.615, 0.805]
    for index, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):
        color = COLORS["green"] if index == 4 else COLORS["blue"]
        rounded_box(
            ax,
            (x, 0.695),
            0.15,
            0.135,
            facecolor="#EFF8F4" if index == 4 else "white",
            edgecolor=color,
            linewidth=1.2,
        )
        ax.text(x + 0.075, 0.788, title, ha="center", fontsize=7.0, fontweight="bold", color=color)
        ax.text(x + 0.075, 0.738, subtitle, ha="center", va="center", fontsize=5.9, linespacing=1.15)
        if index < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.154, 0.762),
                    (x_positions[index + 1] - 0.006, 0.762),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=1.0,
                    color=COLORS["gray"],
                )
            )

    quality_effects = [quality[key]["correct_minus_shuffled"]["value_f1_difference"] for key in ("clean", "jpeg_q70", "blur_r1", "downsample_s075")]
    cards = [
        (
            "PIDQA requested drawing",
            "F1  0.555 correct\n      0.006 shuffled\n      0.000 no image",
            "+0.549 correct - shuffled",
            COLORS["blue"],
        ),
        (
            "Mild quality shifts",
            "Clean / JPEG / blur /\n0.75x restore; 230 sources",
            f"effect {min(quality_effects):.3f}-{max(quality_effects):.3f}",
            COLORS["orange"],
        ),
        (
            "Matched-budget InternVL",
            f"F1  {internvl['correct_f1']:.3f} correct\n      {internvl['shuffled_f1']:.3f} shuffled\n      0.000 no image",
            f"+{internvl['correct_minus_shuffled']['value_f1_difference']:.3f} requested-drawing effect",
            COLORS["green"],
        ),
        (
            "Public DEXPI transfer",
            f"F1  {dexpi['correct']['f1']:.3f} correct\n      {dexpi['shuffled']['f1']:.3f} shuffled\n      {dexpi['text_only']['f1']:.3f} no image",
            "35 images / 26 logical cases",
            COLORS["purple"],
        ),
    ]
    for index, (title, body, footer, color) in enumerate(cards):
        x = 0.045 + index * 0.2375
        rounded_box(ax, (x, 0.31), 0.205, 0.29, facecolor="white", edgecolor=color, linewidth=1.25)
        ax.text(x + 0.015, 0.557, title, fontsize=6.8, fontweight="bold", color=color)
        ax.text(x + 0.015, 0.485, body, fontsize=6.15, va="top", linespacing=1.38)
        ax.text(x + 0.015, 0.338, footer, fontsize=5.65, color=COLORS["gray"])

    labels = [row["recommended_mode"] for row in decision]
    bounds = [float(row["lower_ratio_inclusive"]) for row in decision]
    upper = [row["upper_ratio_exclusive"] for row in decision]
    rule_parts = []
    for label, lower, high in zip(labels, bounds, upper):
        if high == "infinity":
            interval = f"r >= {lower:.3f}"
        elif lower == 0:
            interval = f"r < {float(high):.3f}"
        else:
            interval = f"{lower:.3f}-{float(high):.3f}"
        rule_parts.append(f"{label}: {interval}")
    rounded_box(ax, (0.045, 0.115), 0.91, 0.105, facecolor="#F7FAFC", edgecolor="#AFC3CF", linewidth=1.0)
    ax.text(0.065, 0.183, r"Cost-aware mode selection ($r=C_{FN}/C_{FP}$)", fontsize=6.7, fontweight="bold")
    ax.text(0.065, 0.143, "   |   ".join(rule_parts), fontsize=5.75, color=COLORS["dark"])
    ax.text(
        0.5,
        0.052,
        "Qualified output: configurable candidate-tag retrieval under the evaluated PIDQA and public DEXPI conditions.",
        ha="center",
        fontsize=6.5,
        fontweight="bold",
        color=COLORS["dark"],
    )
    outputs = save_figure(fig, output_dir / "figure_1_saber_pid_overview_v9")
    return outputs, {
        "quality_effect_range": [min(quality_effects), max(quality_effects)],
        "internvl_correct_minus_shuffled": internvl["correct_minus_shuffled"],
        "dexpi_correct_f1": dexpi["correct"]["f1"],
        "cost_intervals": decision,
    }


def read_cost_grid(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["method_id"], []).append(row)
    return {
        method: {
            "ratio": np.asarray([float(row["cost_ratio_fn_over_fp"]) for row in values]),
            "loss": np.asarray([float(row["loss_per_source"]) for row in values]),
            "probability": np.asarray([float(row["bootstrap_probability_optimal"]) for row in values]),
        }
        for method, values in rows.items()
    }


def build_operating_figure(
    output_dir: Path,
    revision: dict[str, Any],
    cost: dict[str, Any],
    grid: dict[str, dict[str, np.ndarray]],
) -> tuple[list[Path], dict[str, Any]]:
    plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 8.2, "axes.labelsize": 7.3})
    fig = plt.figure(figsize=(7.15, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[0.86, 1.35])
    ax_pr = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.11, hspace=0.42, wspace=0.38)
    fig.suptitle(
        "Transparent OCR--VLM modes convert retrieval trade-offs into an engineering decision",
        fontsize=10.6,
        fontweight="bold",
    )

    method_rows = revision["datasets"]["set_b"]["methods"]
    scatter_methods = [
        ("qwen", "Qwen", COLORS["gray"]),
        ("paddleocr_geometry", "OCR joined", COLORS["green"]),
        ("ocr_if_nonempty_else_qwen", "OCR-first", COLORS["blue"]),
        ("set_union", "Union", COLORS["orange"]),
        ("set_intersection", "Intersection", COLORS["purple"]),
    ]
    offsets = {
        "Qwen": (0.012, -0.035),
        "OCR joined": (0.010, 0.012),
        "OCR-first": (0.010, 0.010),
        "Union": (0.010, -0.025),
        "Intersection": (-0.105, 0.015),
    }
    for method, label, color in scatter_methods:
        values = method_rows[method]["micro_pooled"]
        workload = method_rows[method]["workload_per_source"]["candidate_count"]["median"]
        ax_pr.scatter(values["recall"], values["precision"], s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy = offsets[label]
        ax_pr.text(values["recall"] + dx, values["precision"] + dy, f"{label}\n({workload:.0f} cand.)", fontsize=5.2, color=color, linespacing=1.0)
    ax_pr.set_xlim(0.19, 0.76)
    ax_pr.set_ylim(0.50, 1.03)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("A. Operating points and median workload", loc="left", fontweight="bold")
    ax_pr.grid(color=COLORS["light"], linewidth=0.6)

    decision = cost["exact_decision_intervals"]
    for method_id, (label, color) in METHOD_STYLE.items():
        values = grid[method_id]
        ax_loss.plot(values["ratio"], values["loss"], color=color, linewidth=1.0, alpha=0.48)
    for row in decision:
        method_id = row["method_id"]
        label, color = METHOD_STYLE[method_id]
        values = grid[method_id]
        lower = max(float(row["lower_ratio_inclusive"]), float(values["ratio"][0]))
        raw_upper = row["upper_ratio_exclusive"]
        upper = float(values["ratio"][-1]) if raw_upper == "infinity" else float(raw_upper)
        mask = (values["ratio"] >= lower) & (values["ratio"] <= upper)
        ax_loss.plot(values["ratio"][mask], values["loss"][mask], color=color, linewidth=3.4, solid_capstyle="round", label=label)
    for row in decision[:-1]:
        ax_loss.axvline(float(row["upper_ratio_exclusive"]), color="#888888", linewidth=0.65, linestyle="--")
    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel(r"Relative miss cost, $C_{FN}/C_{FP}$")
    ax_loss.set_ylabel("Error cost per drawing")
    ax_loss.set_title("B. Exact lower-loss envelope", loc="left", fontweight="bold")
    ax_loss.grid(True, which="both", color=COLORS["light"], linewidth=0.55)
    handles, labels = ax_loss.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_loss.legend(unique.values(), unique.keys(), frameon=False, ncol=2, loc="upper left", fontsize=5.7)

    for method_id, (label, color) in METHOD_STYLE.items():
        values = grid[method_id]
        ax_prob.plot(values["ratio"], values["probability"], color=color, linewidth=1.9, label=label)
    for row in decision[:-1]:
        ax_prob.axvline(float(row["upper_ratio_exclusive"]), color="#888888", linewidth=0.65, linestyle="--")
    ax_prob.set_xscale("log")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_xlabel(r"Relative miss cost, $C_{FN}/C_{FP}$")
    ax_prob.set_ylabel("Probability of minimum loss")
    ax_prob.set_title("C. Source-bootstrap decision stability (10,000 resamples)", loc="left", fontweight="bold")
    ax_prob.grid(True, which="both", color=COLORS["light"], linewidth=0.55)
    ax_prob.legend(frameon=False, ncol=4, loc="upper center", fontsize=6.1)
    fig.text(
        0.10,
        0.035,
        "Counts and predictions are fixed; the loss scorer changes only the selected transparent operating mode.",
        fontsize=6.0,
        color=COLORS["gray"],
    )
    for axis in (ax_pr, ax_loss, ax_prob):
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    outputs = save_figure(fig, output_dir / "figure_4_cost_aware_operation_v9")
    return outputs, {
        "decision_intervals": decision,
        "operating_points": {
            method: method_rows[method]["micro_pooled"] for method, _label, _color in scatter_methods
        },
    }


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%")


def write_scorecard(path: Path, summary: dict[str, Any]) -> None:
    quality = {row["quality"]: row for row in summary["quality"]}
    quality_effects = [quality[key]["correct_minus_shuffled"]["value_f1_difference"] for key in ("clean", "jpeg_q70", "blur_r1", "downsample_s075")]
    quality_correct = [quality[key]["correct"]["f1"] for key in ("clean", "jpeg_q70", "blur_r1", "downsample_s075")]
    quality_shuffled = [quality[key]["shuffled"]["f1"] for key in ("clean", "jpeg_q70", "blur_r1", "downsample_s075")]
    internvl = next(row for row in summary["internvl_budget54"] if row["dataset"] == "pooled_three_source_disjoint_subsets")
    dexpi = {row["condition"]: row for row in summary["dexpi_external"]["conditions"]}
    dexpi_effect = summary["dexpi_external"]["comparisons"]["correct_minus_shuffled"]
    rows = [
        ("PIDQA, Qwen primary", "100 sources", "0.5549", "0.0062", "+0.5487 [0.4729, 0.6239]"),
        (
            "PIDQA mild-quality matrix",
            "230 disjoint sources",
            f"{min(quality_correct):.4f}--{max(quality_correct):.4f}",
            f"{min(quality_shuffled):.4f}--{max(quality_shuffled):.4f}",
            f"+{min(quality_effects):.4f}--+{max(quality_effects):.4f}",
        ),
        (
            "PIDQA, InternVL 54-tile",
            "230 disjoint sources",
            f"{internvl['correct_f1']:.4f}",
            f"{internvl['shuffled_f1']:.4f}",
            f"+{internvl['correct_minus_shuffled']['value_f1_difference']:.4f} "
            f"[{internvl['correct_minus_shuffled']['value_f1_source_bootstrap_ci95'][0]:.4f}, "
            f"{internvl['correct_minus_shuffled']['value_f1_source_bootstrap_ci95'][1]:.4f}]",
        ),
        (
            "Public DEXPI, Qwen",
            "26 logical cases / 35 images",
            f"{dexpi['correct']['f1']:.4f}",
            f"{dexpi['shuffled']['f1']:.4f}",
            f"+{dexpi_effect['difference']:.4f} [{dexpi_effect['ci95'][0]:.4f}, {dexpi_effect['ci95'][1]:.4f}]",
        ),
    ]
    content = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Qualification scorecard across drawing, quality, and model families. Effects are correct-image minus source-shuffled strict tag F1. Intervals use 10,000 grouped bootstrap replicates; the mild-quality row reports ranges over the four frozen conditions.}",
        r"\label{tab:scorecard}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Setting & Evaluation unit & Correct F1 & Shuffled F1 & Requested-drawing effect [95\% interval] \\",
        r"\midrule",
    ]
    content.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    content.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def format_iqr(values: dict[str, Any]) -> str:
    return f"{values['median']:.0f} [{values['q1']:.0f}, {values['q3']:.0f}]"


def write_operating_table(path: Path, revision: dict[str, Any]) -> None:
    methods = revision["datasets"]["set_b"]["methods"]
    order = [
        ("qwen", "Qwen", "single VLM"),
        ("paddleocr_geometry", "Joined OCR", "low-false-positive OCR"),
        ("set_intersection", "Intersection", "precision-first shortlist"),
        ("ocr_if_nonempty_else_qwen", "OCR-first", "balanced fallback"),
        ("set_union", "Union", "recall-first coverage"),
    ]
    rows = []
    for method, label, role in order:
        metric = methods[method]["micro_pooled"]
        workload = methods[method]["workload_per_source"]
        rows.append(
            (
                label,
                role,
                str(metric["tp"]),
                str(metric["fp"]),
                str(metric["fn"]),
                f"{metric['precision']:.4f}",
                f"{metric['recall']:.4f}",
                f"{metric['f1']:.4f}",
                format_iqr(workload["candidate_count"]),
                format_iqr(workload["false_candidate_count"]),
            )
        )
    content = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Transparent PIDQA operating modes on 100 unseen sources. Workload columns are per-drawing median [IQR].}",
        r"\label{tab:operating}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Mode & Engineering role & TP & FP & FN & Precision & Recall & F1 & Candidates & False candidates \\",
        r"\midrule",
    ]
    content.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    content.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--paper-summary", default="reports/generated/rineng_v8_paper_summary.json")
    parser.add_argument("--cost-report", default="reports/generated/rineng_cost_sensitive_operating_modes_v8.json")
    parser.add_argument("--cost-grid", default="reports/generated/rineng_cost_sensitive_operating_modes_v8.csv")
    parser.add_argument("--revision", default="reports/generated/rineng_revision_analysis_v6.json")
    parser.add_argument("--figure-dir", default="paper/figures")
    parser.add_argument("--table-dir", default="paper/tables")
    parser.add_argument("--metadata", default="paper/figures/figure_metadata_v9.json")
    parser.add_argument("--report", default="reports/generated/rineng_v9_editorial_assets.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    inputs = {
        args.paper_summary: read_json(root / args.paper_summary),
        args.cost_report: read_json(root / args.cost_report),
        args.revision: read_json(root / args.revision),
    }
    summary = inputs[args.paper_summary]
    cost = inputs[args.cost_report]
    revision = inputs[args.revision]
    for name, document in inputs.items():
        if document.get("status") != "pass":
            raise ValueError(f"Input report must pass before plotting: {name}")
    grid = read_cost_grid(root / args.cost_grid)

    figure_dir = root / args.figure_dir
    overview_outputs, overview_values = build_overview(figure_dir, summary, cost)
    operating_outputs, operating_values = build_operating_figure(figure_dir, revision, cost, grid)
    scorecard_path = root / args.table_dir / "table_rineng_v9_qualification_scorecard.tex"
    operating_table_path = root / args.table_dir / "table_rineng_v9_operating_modes.tex"
    write_scorecard(scorecard_path, summary)
    write_operating_table(operating_table_path, revision)

    figure_outputs = overview_outputs + operating_outputs
    metadata = {
        "version": "rineng-v9-editorial-figures",
        "status": "pass",
        "policy": "deterministic composition of validated score reports; no inference and no generative image model",
        "sources": {
            name: sha256(root / name) for name in (*inputs.keys(), args.cost_grid)
        },
        "figures": {
            "figure_1_saber_pid_overview_v9": {
                "files": [path.relative_to(root).as_posix() for path in overview_outputs],
                "values": overview_values,
            },
            "figure_4_cost_aware_operation_v9": {
                "files": [path.relative_to(root).as_posix() for path in operating_outputs],
                "values": operating_values,
            },
        },
    }
    metadata_path = root / args.metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "version": "rineng-v9-editorial-assets",
        "status": "pass",
        "figure_count": 2,
        "table_count": 2,
        "figures": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in figure_outputs
        ],
        "tables": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (scorecard_path, operating_table_path)
        ],
        "metadata": args.metadata,
    }
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figures": 2, "tables": 2, "report": args.report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
