"""Audit that E2/E3/E5/E6/E7/E8 inference inputs remained answer-isolated.

The experiment runners intentionally save each generated model response in both
``answer`` and ``raw`` output fields for compatibility with the frozen scorer.
Those fields are not ground truth.  This script verifies the distinction from
the public inputs and the raw outputs without modifying either one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_input(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = read_rows(path)
    forbidden = sorted(
        {
            key
            for row in rows
            for key in ("answer", "cypher")
            if key in row
        }
    )
    ids = [str(row["instance_id"]) for row in rows]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "duplicate_instance_id_count": len(ids) - len(set(ids)),
        "forbidden_ground_truth_fields_present": forbidden,
        "status": "pass"
        if len(rows) == expected_rows and not forbidden and len(ids) == len(set(ids))
        else "fail",
        "ids": set(ids),
    }


def audit_output(
    path: Path,
    input_ids: set[str],
    expected_rows: int,
    expected_row_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = read_rows(path)
    ids = [str(row["instance_id"]) for row in rows]
    status_rows = [row for row in rows if row.get("status") == "ok"]
    answer_equals_raw = all(row.get("answer") == row.get("raw") for row in status_rows)
    extra_ids = sorted(set(ids) - input_ids)
    expected_row_fields = expected_row_fields or {}
    mismatched_expected_fields = {
        field: sum(row.get(field) != expected for row in status_rows)
        for field, expected in expected_row_fields.items()
    }
    mismatched_expected_fields = {
        field: count for field, count in mismatched_expected_fields.items() if count
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "duplicate_instance_id_count": len(ids) - len(set(ids)),
        "extra_instance_id_count": len(extra_ids),
        "ok_row_count": len(status_rows),
        "answer_field_equals_generated_raw_on_ok_rows": answer_equals_raw,
        "expected_success_row_fields": expected_row_fields,
        "mismatched_expected_success_row_fields": mismatched_expected_fields,
        "status": "pass"
        if len(rows) == expected_rows
        and len(ids) == len(set(ids))
        and not extra_ids
        and len(status_rows) == len(rows)
        and answer_equals_raw
        and not mismatched_expected_fields
        else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/evidence_input_answer_isolation_audit_v1.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    input_specs = {
        "e2": ("data/processed/main400_hashblind_set_b_remote_public.jsonl", 400),
        "e3": ("data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl", 400),
        "e5": ("data/processed/main400_set_b_ontology_visible_v1_public.jsonl", 400),
        "e7": ("data/processed/main400_hashblind_set_b_remote_public.jsonl", 400),
        "e8": ("data/processed/main400_hashblind_set_b_remote_public.jsonl", 400),
        "e6_seed29": ("data/processed/source_seed29_resolution_v1_remote_public.jsonl", 400),
        "e6_seed31": ("data/processed/source_seed31_resolution_v1_remote_public.jsonl", 400),
    }
    output_specs = {
        "e2_768": ("outputs/evidence_strengthening/qwen8_value_budget_v1/qwen8_value_budget_v1_768.jsonl", "e2", 100),
        "e2_3072": ("outputs/evidence_strengthening/qwen8_value_budget_v1/qwen8_value_budget_v1_3072.jsonl", "e2", 100),
        "e3_768": ("outputs/evidence_strengthening/qwen8_image_shuffle_v1/qwen8_image_shuffle_v1_768.jsonl", "e3", 400),
        "e3_3072": ("outputs/evidence_strengthening/qwen8_image_shuffle_v1/qwen8_image_shuffle_v1_3072.jsonl", "e3", 400),
        "e5_768": ("outputs/evidence_strengthening/qwen8_ontology_visible_v1/qwen8_ontology_visible_v1_768.jsonl", "e5", 400),
        "e5_3072": ("outputs/evidence_strengthening/qwen8_ontology_visible_v1/qwen8_ontology_visible_v1_3072.jsonl", "e5", 400),
        "e7_768": (
            "outputs/evidence_strengthening/qwen8_ontology_permuted_v1/qwen8_ontology_permuted_v1_768.jsonl",
            "e7",
            100,
            {"content_condition": "image_with_legend", "input_image_count": 2},
        ),
        "e7_3072": (
            "outputs/evidence_strengthening/qwen8_ontology_permuted_v1/qwen8_ontology_permuted_v1_3072.jsonl",
            "e7",
            100,
            {"content_condition": "image_with_legend", "input_image_count": 2},
        ),
        "e8_text_only": (
            "outputs/evidence_strengthening/qwen8_text_only_v1/qwen8_text_only_v1_3072.jsonl",
            "e8",
            400,
            {"content_condition": "text_only", "input_image_count": 0},
        ),
        "e6_seed29_768": ("outputs/evidence_strengthening/qwen8_source_seed29_resolution_v1/qwen8_source_seed29_resolution_v1_768.jsonl", "e6_seed29", 400),
        "e6_seed29_3072": ("outputs/evidence_strengthening/qwen8_source_seed29_resolution_v1/qwen8_source_seed29_resolution_v1_3072.jsonl", "e6_seed29", 400),
        "e6_seed31_768": ("outputs/evidence_strengthening/qwen8_source_seed31_resolution_v1/qwen8_source_seed31_resolution_v1_768.jsonl", "e6_seed31", 400),
        "e6_seed31_3072": ("outputs/evidence_strengthening/qwen8_source_seed31_resolution_v1/qwen8_source_seed31_resolution_v1_3072.jsonl", "e6_seed31", 400),
    }
    inputs = {name: audit_input(root / relative, rows) for name, (relative, rows) in input_specs.items()}
    outputs = {}
    for name, spec in output_specs.items():
        relative, input_name, rows, *field_expectations = spec
        outputs[name] = audit_output(
            root / relative,
            inputs[input_name]["ids"],
            rows,
            field_expectations[0] if field_expectations else None,
        )
    for summary in (*inputs.values(), *outputs.values()):
        summary["path"] = (
            Path(summary["path"]).resolve().relative_to(root).as_posix()
        )
    for summary in inputs.values():
        summary.pop("ids")
    payload = {
        "status": "pass"
        if all(row["status"] == "pass" for row in inputs.values())
        and all(row["status"] == "pass" for row in outputs.values())
        else "fail",
        "interpretation": {
            "public_input_ground_truth_fields": "answer and cypher are forbidden and absent",
            "output_answer_field": "for successful runner rows, answer is an alias of the generated raw model response, not the hidden reference answer",
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
