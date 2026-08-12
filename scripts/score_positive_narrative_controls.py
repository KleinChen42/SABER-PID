"""Score E7 and E8 controls introduced by the positive-narrative revision.

The inference runners read only answer-isolated public manifests.  This module
is the scorer-side join: it reads the local hidden Set-B references, produces
strict and semantic task metrics, and uses the same paired source-cluster
bootstrap implementation as E2--E6.

E7 compares raw, correct-legend, and label-permuted legend conditions for
spatial-count records.  E8 compares correct-image, source-shuffled, and
text-only Qwen conditions for all Set-B tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_e1_evidence_audit import read_rows
from score_evidence_strengthening import (
    add_comparison,
    cell_table,
    predictions_for_tasks,
    records_for_tasks,
    score_cell,
    write_csv,
    write_json,
)


def serializable_cells(cells: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Drop instance-level scorer events from the compact JSON summary."""

    return {
        key: {field: value for field, value in cell.items() if field != "events"}
        for key, cell in cells.items()
    }


def e7(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    """Score the layout-matched numeric-label permutation control."""

    spatial_records = records_for_tasks(records, {"spatial_count"})
    base_dir = root / "outputs/final_replication"
    visible_dir = root / "outputs/evidence_strengthening/qwen8_ontology_visible_v1"
    permuted_dir = root / "outputs/evidence_strengthening/qwen8_ontology_permuted_v1"
    cells: dict[str, dict[str, Any]] = {}
    for side in (768, 3072):
        cells[f"qwen8_b_p0_raw_{side}"] = score_cell(
            label=f"qwen8_b_p0_raw_{side}",
            records=spatial_records,
            predictions=predictions_for_tasks(
                read_rows(base_dir / f"qwen8_b_p0_{side}.jsonl"), {"spatial_count"}
            ),
            metadata={
                "experiment": "E7",
                "condition": "raw_image_only",
                "side": side,
                "max_new_tokens": 192,
            },
        )
        cells[f"qwen8_b_p0_ontology_visible_{side}"] = score_cell(
            label=f"qwen8_b_p0_ontology_visible_{side}",
            records=spatial_records,
            predictions=predictions_for_tasks(
                read_rows(visible_dir / f"qwen8_ontology_visible_v1_{side}.jsonl"),
                {"spatial_count"},
            ),
            metadata={
                "experiment": "E7",
                "condition": "correct_numeric_class_legend",
                "side": side,
                "max_new_tokens": 192,
            },
        )
        cells[f"qwen8_b_p0_ontology_permuted_{side}"] = score_cell(
            label=f"qwen8_b_p0_ontology_permuted_{side}",
            records=spatial_records,
            predictions=read_rows(
                permuted_dir / f"qwen8_ontology_permuted_v1_{side}.jsonl"
            ),
            metadata={
                "experiment": "E7",
                "condition": "layout_matched_cyclic_label_permutation",
                "label_shift": 1,
                "side": side,
                "max_new_tokens": 192,
            },
        )

    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for side, base_seed in ((768, 2701), (3072, 2704)):
        add_comparison(
            comparisons,
            transitions,
            f"e7_correct_legend_minus_raw_{side}",
            cells[f"qwen8_b_p0_raw_{side}"],
            cells[f"qwen8_b_p0_ontology_visible_{side}"],
            reps,
            base_seed,
        )
        add_comparison(
            comparisons,
            transitions,
            f"e7_permuted_legend_minus_raw_{side}",
            cells[f"qwen8_b_p0_raw_{side}"],
            cells[f"qwen8_b_p0_ontology_permuted_{side}"],
            reps,
            base_seed + 1,
        )
        add_comparison(
            comparisons,
            transitions,
            f"e7_correct_legend_minus_permuted_{side}",
            cells[f"qwen8_b_p0_ontology_permuted_{side}"],
            cells[f"qwen8_b_p0_ontology_visible_{side}"],
            reps,
            base_seed + 2,
        )
    return "ontology_mapping_control_v1", {
        "experiment": "E7",
        "status": "pass",
        "records": {"task_filter": "spatial_count", "record_count": len(spatial_records)},
        "control": {
            "label_permutation": "cyclic +1 over numeric labels 1..32",
            "matched_factors": [
                "symbol prototypes",
                "grid layout",
                "font",
                "image dimensions",
                "second-image count",
                "frozen model/prompt/decoder",
            ],
            "changed_factor": "numeric label-to-prototype mapping",
        },
        "cells": serializable_cells(cells),
        "comparisons": comparisons,
        "paired_transitions": transitions,
        "bootstrap_reps": reps,
    }


def e8(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    """Score the same-model text-only image-grounding control."""

    base_dir = root / "outputs/final_replication"
    shuffled_dir = root / "outputs/evidence_strengthening/qwen8_image_shuffle_v1"
    text_only_dir = root / "outputs/evidence_strengthening/qwen8_text_only_v1"
    cells = {
        "qwen8_b_p0_correct_3072": score_cell(
            label="qwen8_b_p0_correct_3072",
            records=records,
            predictions=read_rows(base_dir / "qwen8_b_p0_3072.jsonl"),
            metadata={
                "experiment": "E8",
                "condition": "correct_image",
                "side": 3072,
                "max_new_tokens": 192,
            },
        ),
        "qwen8_b_p0_shuffled_3072": score_cell(
            label="qwen8_b_p0_shuffled_3072",
            records=records,
            predictions=read_rows(shuffled_dir / "qwen8_image_shuffle_v1_3072.jsonl"),
            metadata={
                "experiment": "E8",
                "condition": "source_shuffled_no_fixed_point",
                "side": 3072,
                "max_new_tokens": 192,
            },
        ),
        "qwen8_b_p0_text_only": score_cell(
            label="qwen8_b_p0_text_only",
            records=records,
            predictions=read_rows(text_only_dir / "qwen8_text_only_v1_3072.jsonl"),
            metadata={
                "experiment": "E8",
                "condition": "text_only_same_model_prompt_decoder",
                "side": None,
                "max_image_side_argument": 3072,
                "max_new_tokens": 192,
            },
        ),
    }
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    add_comparison(
        comparisons,
        transitions,
        "e8_correct_image_minus_text_only_3072",
        cells["qwen8_b_p0_text_only"],
        cells["qwen8_b_p0_correct_3072"],
        reps,
        2801,
    )
    add_comparison(
        comparisons,
        transitions,
        "e8_shuffled_image_minus_text_only_3072",
        cells["qwen8_b_p0_text_only"],
        cells["qwen8_b_p0_shuffled_3072"],
        reps,
        2802,
    )
    add_comparison(
        comparisons,
        transitions,
        "e8_correct_image_minus_shuffled_3072",
        cells["qwen8_b_p0_shuffled_3072"],
        cells["qwen8_b_p0_correct_3072"],
        reps,
        2803,
    )
    return "text_only_image_grounding_control_v1", {
        "experiment": "E8",
        "status": "pass",
        "records": {"task_filter": "all", "record_count": len(records)},
        "control": {
            "frozen_factors": ["model", "prompt", "decoder", "token ceiling", "question", "scorer"],
            "changed_factor": "image content absent",
            "text_only_file_side_suffix": 3072,
            "interpretation": "The suffix is a stable filename compatibility token; no image side is applied.",
        },
        "cells": serializable_cells(cells),
        "comparisons": comparisons,
        "paired_transitions": transitions,
        "bootstrap_reps": reps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--experiment", required=True, choices=("e7", "e8"))
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    root = Path(args.root).resolve()
    records = read_rows(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl")
    builders = {"e7": e7, "e8": e8}
    name, payload = builders[args.experiment](root, records, args.bootstrap_reps)
    outdir = root / args.output_dir
    write_json(outdir / f"{name}.json", payload)

    table_rows: list[dict[str, Any]] = []
    for cell in payload["cells"].values():
        table_rows.extend(cell_table({**cell, "events": []}))
    table_rows.extend({"row_type": "comparison", **row} for row in payload["comparisons"])
    table_rows.extend({"row_type": "paired_transition", **row} for row in payload["paired_transitions"])
    write_csv(outdir / f"{name}.csv", table_rows)
    print(
        json.dumps(
            {
                "status": "pass",
                "experiment": payload["experiment"],
                "report": str(outdir / f"{name}.json"),
                "cell_count": len(payload["cells"]),
                "comparison_count": len(payload["comparisons"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
