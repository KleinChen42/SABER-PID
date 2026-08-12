"""Build the truth-free OCR/VLM fusion analysis used by the v5 narrative.

The fusion rules operate only on frozen predicted tag sets.  All four simple
set/fallback rules are scored and retained so that the analysis does not hide
an unfavourable member of a searched rule family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from pidbench.pidqa_metrics import normalize_pidqa_answer
from run_e1_evidence_audit import read_rows
from score_editorial_extension_experiments_v4 import geometry_joined_ocr_prediction
from score_evidence_strengthening import add_comparison, score_cell


VERSION = "positive-narrative-hybrid-v5"
RULES = (
    "set_union",
    "set_intersection",
    "ocr_if_nonempty_else_qwen",
    "qwen_if_nonempty_else_ocr",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_set(row: dict[str, Any]) -> set[str]:
    value = normalize_pidqa_answer(row.get("answer"), "value")
    return set(value or ())


def render(tags: set[str]) -> str:
    return ", ".join(sorted(tags)) if tags else "[]"


def fuse(qwen: set[str], ocr: set[str], rule: str) -> set[str]:
    if rule == "set_union":
        return qwen | ocr
    if rule == "set_intersection":
        return qwen & ocr
    if rule == "ocr_if_nonempty_else_qwen":
        return ocr if ocr else qwen
    if rule == "qwen_if_nonempty_else_ocr":
        return qwen if qwen else ocr
    raise KeyError(rule)


def public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cell.items() if key != "events"}


def metric_row(label: str, cell: dict[str, Any]) -> dict[str, Any]:
    metrics = cell["metrics"]
    tags = metrics["strict_value_tags"]
    return {
        "row_type": "cell",
        "label": label,
        "precision": tags["precision"],
        "recall": tags["recall"],
        "f1": tags["f1"],
        "exact": metrics["task"]["value"]["strict_accuracy"],
        "tp": tags["tp"],
        "fp": tags["fp"],
        "fn": tags["fn"],
        "record_count": metrics["task"]["value"]["record_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--output", default="reports/generated/positive_narrative_hybrid_analysis_v5.json")
    parser.add_argument("--csv-output", default="reports/generated/positive_narrative_hybrid_analysis_v5.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    qwen_path = root / "outputs/final_replication/qwen8_b_p0_3072.jsonl"
    ocr_path = root / "outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl"
    records_path = root / "data/processed/pidqa_records.jsonl"

    qwen_rows = [row for row in read_rows(qwen_path) if str(row.get("task")) == "value"]
    ocr_raw = read_rows(ocr_path)
    ocr_rows = [geometry_joined_ocr_prediction(row) for row in ocr_raw]
    qwen_by_id = {str(row["instance_id"]): row for row in qwen_rows}
    ocr_by_id = {str(row["instance_id"]): row for row in ocr_rows}
    ids = set(qwen_by_id)
    failures: list[str] = []
    if len(qwen_rows) != 100 or len(qwen_by_id) != 100:
        failures.append("qwen_membership_not_100_unique")
    if len(ocr_rows) != 100 or len(ocr_by_id) != 100:
        failures.append("ocr_membership_not_100_unique")
    if ids != set(ocr_by_id):
        failures.append("qwen_ocr_membership_mismatch")
    if any(row.get("status", "ok") != "ok" for row in qwen_rows + ocr_raw):
        failures.append("non_ok_raw_output")
    forbidden = {"reference_answer", "truth", "cypher"}
    if any(forbidden & set(row) for row in qwen_rows + ocr_raw):
        failures.append("reference_field_in_raw_output")
    if any(row.get("test_answer_used") is True for row in qwen_rows + ocr_raw):
        failures.append("test_answer_used_true")

    records = [row for row in read_rows(records_path) if str(row.get("instance_id")) in ids]
    if len(records) != 100 or {str(row["instance_id"]) for row in records} != ids:
        failures.append("scorer_record_membership_mismatch")
    if failures:
        raise ValueError("; ".join(failures))

    base_predictions = {"qwen": qwen_rows, "paddleocr_geometry": ocr_rows}
    predictions: dict[str, list[dict[str, Any]]] = dict(base_predictions)
    for rule in RULES:
        rows: list[dict[str, Any]] = []
        for instance_id in sorted(ids):
            qwen = answer_set(qwen_by_id[instance_id])
            ocr = answer_set(ocr_by_id[instance_id])
            tags = fuse(qwen, ocr, rule)
            rows.append(
                {
                    "instance_id": instance_id,
                    "source_id": str(qwen_by_id[instance_id]["source_id"]),
                    "task": "value",
                    "action": "ANSWER",
                    "answer": render(tags),
                    "raw": render(tags),
                    "status": "ok",
                    "fusion_rule": rule,
                    "test_answer_used": False,
                }
            )
        predictions[rule] = rows

    cells = {
        label: score_cell(
            label=label,
            records=records,
            predictions=rows,
            metadata={
                "analysis": "truth-free predicted-tag fusion",
                "confirmatory_status": "descriptive post-hoc extension",
            },
        )
        for label, rows in predictions.items()
    }
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    requested = (
        ("union_minus_qwen", "qwen", "set_union"),
        ("union_minus_ocr", "paddleocr_geometry", "set_union"),
        ("intersection_minus_qwen", "qwen", "set_intersection"),
        ("intersection_minus_ocr", "paddleocr_geometry", "set_intersection"),
        ("ocr_fallback_minus_qwen", "qwen", "ocr_if_nonempty_else_qwen"),
        ("ocr_fallback_minus_ocr", "paddleocr_geometry", "ocr_if_nonempty_else_qwen"),
    )
    for offset, (label, baseline, condition) in enumerate(requested):
        add_comparison(
            comparisons,
            transitions,
            label,
            cells[baseline],
            cells[condition],
            args.bootstrap_reps,
            5100 + offset * 100,
        )

    payload = {
        "version": VERSION,
        "status": "pass",
        "analysis_role": "descriptive post-hoc extension; not independently validated",
        "fusion_boundary": {
            "reference_used_in_prediction_construction": False,
            "all_predeclared_simple_rules_reported": True,
            "rules": list(RULES),
            "selection_statement": "The union is discussed for coverage and the intersection for precision; no rule is hidden.",
        },
        "records": {"task": "value", "record_count": len(records), "source_count": len({str(row["source_id"]) for row in records})},
        "sources": [
            {"path": str(qwen_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(qwen_path)},
            {"path": str(ocr_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(ocr_path)},
            {"path": str(records_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(records_path)},
        ],
        "cells": {label: public_cell(cell) for label, cell in cells.items()},
        "comparisons": comparisons,
        "paired_transitions": transitions,
        "bootstrap_reps": args.bootstrap_reps,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_rows = [metric_row(label, cell) for label, cell in cells.items()]
    csv_rows.extend(
        {
            "row_type": "comparison",
            "label": row["comparison"],
            "precision": "",
            "recall": "",
            "f1": row["difference_condition_minus_baseline"],
            "exact": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "record_count": row["source_count"],
            "ci95_low": row["source_bootstrap_ci95_low"],
            "ci95_high": row["source_bootstrap_ci95_high"],
        }
        for row in comparisons
        if row["metric"] == "strict_value_tag_f1" and row["task"] == "value"
    )
    csv_output = root / args.csv_output
    fieldnames = sorted({key for row in csv_rows for key in row})
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps({"status": "pass", "output": str(output), "cells": len(cells)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
