"""Score prospectively frozen fusion rules on seed-29 and seed-31 partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_positive_narrative_hybrid_analysis_v5 import RULES, answer_set, fuse, public_cell, render, sha256
from run_e1_evidence_audit import read_rows
from score_editorial_extension_experiments_v4 import geometry_joined_ocr_prediction
from score_evidence_strengthening import add_comparison, score_cell


def score_seed(root: Path, seed: int, reps: int) -> dict[str, Any]:
    qwen_path = root / f"outputs/evidence_strengthening/qwen8_source_seed{seed}_resolution_v1/qwen8_source_seed{seed}_resolution_v1_3072.jsonl"
    ocr_path = root / f"outputs/positive_narrative/paddleocr_seed{seed}_v1.jsonl"
    qwen = [row for row in read_rows(qwen_path) if str(row.get("task")) == "value"]
    ocr_raw = read_rows(ocr_path)
    ocr = [geometry_joined_ocr_prediction(row) for row in ocr_raw]
    qwen_by_id = {str(row["instance_id"]): row for row in qwen}
    ocr_by_id = {str(row["instance_id"]): row for row in ocr}
    ids = set(qwen_by_id)
    failures: list[str] = []
    if len(qwen) != 100 or len(qwen_by_id) != 100:
        failures.append("qwen_not_100_unique")
    if len(ocr) != 100 or len(ocr_by_id) != 100:
        failures.append("ocr_not_100_unique")
    if ids != set(ocr_by_id):
        failures.append("qwen_ocr_membership_mismatch")
    if any(row.get("status", "ok") != "ok" for row in qwen + ocr_raw):
        failures.append("non_ok_output")
    if any(row.get("test_answer_used") is True for row in qwen + ocr_raw):
        failures.append("test_answer_used_true")
    if any({"reference_answer", "truth", "cypher"} & set(row) for row in qwen + ocr_raw):
        failures.append("reference_field_in_output")

    records = [row for row in read_rows(root / "data/processed/pidqa_records.jsonl") if str(row.get("instance_id")) in ids]
    if len(records) != 100 or {str(row["instance_id"]) for row in records} != ids:
        failures.append("scorer_membership_mismatch")
    if failures:
        raise ValueError(f"seed {seed}: {'; '.join(failures)}")

    predictions: dict[str, list[dict[str, Any]]] = {"qwen": qwen, "paddleocr_geometry": ocr}
    for rule in RULES:
        rows = []
        for instance_id in sorted(ids):
            tags = fuse(answer_set(qwen_by_id[instance_id]), answer_set(ocr_by_id[instance_id]), rule)
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
            label=f"seed{seed}_{label}",
            records=records,
            predictions=rows,
            metadata={"seed": seed, "analysis": "prospectively frozen truth-free fusion validation"},
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
            f"seed{seed}_{label}",
            cells[baseline],
            cells[condition],
            reps,
            5500 + seed * 10 + offset,
        )
    return {
        "seed": seed,
        "status": "pass",
        "record_count": len(records),
        "source_count": len({str(row["source_id"]) for row in records}),
        "cells": {label: public_cell(cell) for label, cell in cells.items()},
        "comparisons": comparisons,
        "paired_transitions": transitions,
        "sources": [
            {"path": str(qwen_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(qwen_path)},
            {"path": str(ocr_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(ocr_path)},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--output", default="reports/generated/positive_narrative_fusion_validation_v5.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    seeds = [score_seed(root, seed, args.bootstrap_reps) for seed in (29, 31)]
    qwen_paths = {
        "set_b": root / "outputs/final_replication/qwen8_b_p0_3072.jsonl",
        "seed29": root / "outputs/evidence_strengthening/qwen8_source_seed29_resolution_v1/qwen8_source_seed29_resolution_v1_3072.jsonl",
        "seed31": root / "outputs/evidence_strengthening/qwen8_source_seed31_resolution_v1/qwen8_source_seed31_resolution_v1_3072.jsonl",
    }
    memberships = {
        label: {
            str(row["source_id"])
            for row in read_rows(path)
            if str(row.get("task")) == "value"
        }
        for label, path in qwen_paths.items()
    }
    if any(len(values) != 100 for values in memberships.values()):
        raise ValueError("Expected 100 unique value sources in Set B and each validation partition")
    overlap = {
        "set_b_seed29": len(memberships["set_b"] & memberships["seed29"]),
        "set_b_seed31": len(memberships["set_b"] & memberships["seed31"]),
        "seed29_seed31": len(memberships["seed29"] & memberships["seed31"]),
        "three_way": len(memberships["set_b"] & memberships["seed29"] & memberships["seed31"]),
    }
    payload = {
        "version": "positive-narrative-fusion-validation-v5",
        "status": "pass",
        "validation_role": "fusion rules frozen from Set B before scoring seed-29/31; same synthetic source family, not external replication",
        "reference_used_in_prediction_construction": False,
        "all_rules_reported": True,
        "source_partition_overlap": overlap,
        "bootstrap_reps": args.bootstrap_reps,
        "seeds": seeds,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "seeds": len(seeds)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
