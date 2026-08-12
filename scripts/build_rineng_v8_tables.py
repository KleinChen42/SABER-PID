"""Build manuscript-ready V8 tables and a machine-readable paper summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POOLED = "pooled_three_source_disjoint_subsets"
QUALITY_ORDER = ("clean", "jpeg_q70", "blur_r1", "downsample_s075")
QUALITY_LABEL = {
    "clean": "Clean",
    "jpeg_q70": "JPEG Q70",
    "blur_r1": "Blur $r=1$",
    "downsample_s075": "0.75$\\times$ restore",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def pooled_quality_metric(
    report: dict[str, Any], quality: str, condition: str
) -> dict[str, float | int]:
    matches = [
        cell["metrics"]["strict_value_tags"]
        for key, cell in report["cells"].items()
        if key.startswith("quality|qwen3vl8b|")
        and key.endswith(f"|{quality}|{condition}")
    ]
    if len(matches) != 3:
        raise ValueError(f"Expected three quality cells for {quality}/{condition}")
    return metric_from_counts(
        sum(int(row["tp"]) for row in matches),
        sum(int(row["fp"]) for row in matches),
        sum(int(row["fn"]) for row in matches),
    )


def find_quality(
    report: dict[str, Any], quality: str, contrast: str, dataset: str = POOLED
) -> dict[str, Any]:
    matches = [
        row
        for row in report["quality_comparisons"]
        if row["dataset"] == dataset
        and row["quality"] == quality
        and row["contrast"] == contrast
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one quality row: {dataset}/{quality}/{contrast}")
    return matches[0]


def find_internvl(
    report: dict[str, Any], dataset: str, contrast: str
) -> dict[str, Any]:
    matches = [
        row
        for row in report["internvl_comparisons"]
        if row["dataset"] == dataset and row["contrast"] == contrast
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one InternVL row: {dataset}/{contrast}")
    return matches[0]


def format_effect(point: float, interval: list[float]) -> str:
    return f"{point:+.4f} [{interval[0]:+.4f}, {interval[1]:+.4f}]"


def build_quality_table(report: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    tex_rows = []
    for quality in QUALITY_ORDER:
        correct = pooled_quality_metric(report, quality, "correct")
        shuffled = pooled_quality_metric(report, quality, "shuffled")
        effect = find_quality(report, quality, "correct_minus_shuffled")
        did = None
        if quality != "clean":
            did = find_quality(
                report,
                quality,
                "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1",
            )
        row = {
            "quality": quality,
            "source_count": int(effect.get("source_count", 230)),
            "correct": correct,
            "shuffled": shuffled,
            "correct_minus_shuffled": effect,
            "difference_in_differences": did,
        }
        rows.append(row)
        effect_text = format_effect(
            float(effect["value_f1_difference"]),
            list(effect["value_f1_source_bootstrap_ci95"]),
        )
        did_text = "--"
        if did is not None:
            did_text = format_effect(
                float(did["value_f1_difference_in_differences"]),
                list(did["source_bootstrap_ci95"]),
            )
        tex_rows.append(
            f"{QUALITY_LABEL[quality]} & {correct['f1']:.4f} & {shuffled['f1']:.4f} "
            f"& {effect_text} & {did_text} \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Paired quality robustness at the qualified Qwen 3072-side/512-token setting, pooled over 230 pairwise source-disjoint drawings. Intervals use 10,000 paired source-bootstrap replicates. The final column is the change in the correct-minus-shuffled effect relative to clean.}",
            "\\label{tab:v8quality}",
            "\\scriptsize",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Condition & Correct F1 & Shuffled F1 & Correct$-$shuffled [95\\% interval] & Change vs clean [95\\% interval] \\\\",
            "\\midrule",
            *tex_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
        ]
    )
    return tex, rows


def build_internvl_table(report: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    datasets = ("set_b100", "seed29_strict65", "seed31_strict65")
    rows = []
    tex_rows = []
    labels = {
        "set_b100": "Set B (100)",
        "seed29_strict65": "Seed 29 disjoint (65)",
        "seed31_strict65": "Seed 31 disjoint (65)",
        POOLED: "Pooled (230)",
    }

    def cell_f1(dataset: str, condition: str) -> float:
        if dataset != POOLED:
            return float(
                report["cells"][f"internvl_budget54|{dataset}|{condition}"]["metrics"][
                    "strict_value_tags"
                ]["f1"]
            )
        metrics = [
            report["cells"][f"internvl_budget54|{item}|{condition}"]["metrics"][
                "strict_value_tags"
            ]
            for item in datasets
        ]
        return float(
            metric_from_counts(
                sum(int(row["tp"]) for row in metrics),
                sum(int(row["fp"]) for row in metrics),
                sum(int(row["fn"]) for row in metrics),
            )["f1"]
        )

    for dataset in (*datasets, POOLED):
        correct = cell_f1(dataset, "correct")
        shuffled = cell_f1(dataset, "shuffled")
        text_only = cell_f1(dataset, "text_only")
        shuffled_effect = find_internvl(report, dataset, "correct_minus_shuffled")
        text_effect = find_internvl(report, dataset, "correct_minus_text_only")
        native_effect = find_internvl(
            report, dataset, "budget54_minus_native_tiles12_correct"
        )
        row = {
            "dataset": dataset,
            "correct_f1": correct,
            "shuffled_f1": shuffled,
            "text_only_f1": text_only,
            "correct_minus_shuffled": shuffled_effect,
            "correct_minus_text_only": text_effect,
            "budget54_minus_native12": native_effect,
        }
        rows.append(row)
        tex_rows.append(
            f"{labels[dataset]} & {correct:.4f} & {shuffled:.4f} & {text_only:.4f} & "
            f"{format_effect(float(shuffled_effect['value_f1_difference']), list(shuffled_effect['value_f1_source_bootstrap_ci95']))} & "
            f"{format_effect(float(native_effect['value_f1_difference']), list(native_effect['value_f1_source_bootstrap_ci95']))} \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Closest-safe InternVL3.5-8B visual-budget comparison. The 54-tile input has 32.51 million tensor elements versus 35.98 million for the qualified Qwen processor; this does not imply encoder equivalence. Intervals use 10,000 paired source-bootstrap replicates.}",
            "\\label{tab:v8internvl}",
            "\\scriptsize",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Subset & Correct F1 & Shuffled F1 & No-image F1 & Correct$-$shuffled [95\\% interval] & 54-tile$-$native-12 [95\\% interval] \\\\",
            "\\midrule",
            *tex_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
        ]
    )
    return tex, rows


def build_external_table(report: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    order = ("correct", "shuffled", "text_only", "paddleocr_full_image")
    labels = {
        "correct": "Qwen, correct image",
        "shuffled": "Qwen, cross-case shuffled",
        "text_only": "Qwen, no image",
        "paddleocr_full_image": "PaddleOCR, full image",
    }
    rows = []
    tex_rows = []
    for condition in order:
        metric = dict(report["metrics"][condition])
        rows.append({"condition": condition, **metric})
        tex_rows.append(
            f"{labels[condition]} & {metric['tp']} & {metric['fp']} & {metric['fn']} & "
            f"{metric['precision']:.4f} & {metric['recall']:.4f} & {metric['f1']:.4f} & "
            f"{metric['exact_set_accuracy']:.4f} \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{External-family tag retrieval on 65 questions from 35 DEXPI images in 26 logical test cases. Vendor variants are grouped by logical case for uncertainty estimates.}",
            "\\label{tab:v8dexpi}",
            "\\scriptsize",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "Condition & TP & FP & FN & Precision & Recall & F1 & Exact \\\\",
            "\\midrule",
            *tex_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
        ]
    )
    return tex, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--extension-score", default="reports/generated/rineng_v8_extension_score.json"
    )
    parser.add_argument(
        "--external-score", default="reports/generated/rineng_v8_dexpi_external_score.json"
    )
    parser.add_argument("--table-dir", default="paper/tables")
    parser.add_argument(
        "--summary", default="reports/generated/rineng_v8_paper_summary.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    extension = read_json(root / args.extension_score)
    external = read_json(root / args.external_score)
    if extension.get("status") != "pass" or external.get("status") != "pass":
        raise ValueError("Refusing to build paper tables from a failing score report")

    quality_tex, quality_rows = build_quality_table(extension)
    internvl_tex, internvl_rows = build_internvl_table(extension)
    external_tex, external_rows = build_external_table(external)
    table_dir = root / args.table_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "quality": table_dir / "table_rineng_v8_quality.tex",
        "internvl": table_dir / "table_rineng_v8_internvl_budget54.tex",
        "external": table_dir / "table_rineng_v8_dexpi_external.tex",
    }
    outputs["quality"].write_text(quality_tex, encoding="utf-8")
    outputs["internvl"].write_text(internvl_tex, encoding="utf-8")
    outputs["external"].write_text(external_tex, encoding="utf-8")
    summary = {
        "version": "rineng-v8-paper-summary",
        "status": "pass",
        "quality": quality_rows,
        "internvl_budget54": internvl_rows,
        "dexpi_external": {
            "selection": external["selection"],
            "conditions": external_rows,
            "comparisons": external["comparisons"],
        },
        "table_files": {
            key: path.relative_to(root).as_posix() for key, path in outputs.items()
        },
    }
    summary_path = root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "pass", "tables": len(outputs), "summary": args.summary},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
