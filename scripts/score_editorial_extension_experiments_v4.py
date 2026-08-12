"""Score the v4 OCR baseline and InternVL counterfactual value ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from run_e1_evidence_audit import read_rows
from score_evidence_strengthening import add_comparison, cell_table, score_cell


OCR_PREFIX_FRAGMENT = re.compile(r"^[A-Za-z]{1,8}(?:-\d+(?:-\d+)*)?$")
OCR_NUMERIC_SUFFIX = re.compile(r"^\d{1,6}$")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_public_manifest(path: Path, expected_ids: set[str], *, shuffled: bool) -> dict[str, Any]:
    rows = read_rows(path)
    value_rows = [row for row in rows if str(row.get("task")) == "value"]
    ids = {str(row.get("instance_id")) for row in value_rows}
    failures: list[str] = []
    if len(value_rows) != 100 or ids != expected_ids:
        failures.append("value_membership_mismatch")
    forbidden = ("answer", "cypher", "reference_answer", "truth")
    if any(any(key in row for key in forbidden) for row in rows):
        failures.append("reference_field_present")
    if shuffled:
        mappings = [(str(row.get("source_id")), str(row.get("image_source_id"))) for row in value_rows]
        if any(source == image_source or not image_source for source, image_source in mappings):
            failures.append("shuffle_fixed_point_or_missing_image_source")
        if len({image_source for _, image_source in mappings}) != len(mappings):
            failures.append("shuffle_not_bijective_on_value_sources")
    if failures:
        raise ValueError(f"Invalid public manifest {path}: {', '.join(failures)}")
    return {
        "path": f"data/processed/{path.name}",
        "sha256": sha256(path),
        "row_count": len(rows),
        "value_row_count": len(value_rows),
        "answer_or_query_fields_present": False,
        "shuffle_no_fixed_point": shuffled,
    }


def validated_predictions(path: Path, label: str, expected_ids: set[str]) -> list[dict[str, Any]]:
    rows = read_rows(path)
    ids = [str(row.get("instance_id")) for row in rows]
    failures: list[str] = []
    if len(rows) != 100:
        failures.append(f"row_count={len(rows)}")
    if len(set(ids)) != len(ids):
        failures.append("duplicate_instance_id")
    if set(ids) != expected_ids:
        failures.append("instance_membership_mismatch")
    if any(row.get("status", "ok") != "ok" for row in rows):
        failures.append("non_ok_status")
    if any(row.get("test_answer_used") is True for row in rows):
        failures.append("test_answer_used_true")
    if any(any(key in row for key in ("reference_answer", "truth", "cypher")) for row in rows):
        failures.append("reference_field_present")
    if failures:
        raise ValueError(f"Invalid {label} output: {', '.join(failures)}")
    return rows


def box_bounds(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = line.get("box")
    if not isinstance(box, list) or len(box) < 2:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    return min(xs), min(ys), max(xs), max(ys)


def geometry_joined_ocr_prediction(row: dict[str, Any]) -> dict[str, Any]:
    """Join a prefix fragment to one immediately lower numeric OCR box.

    The rule is reference-free and fixed for all sources.  It addresses the
    common P&ID rendering in which a tag prefix/class (for example ``RO-10``)
    and its numeric suffix are emitted as two vertically stacked OCR lines.
    Fully formed tags with a final numeric segment longer than two characters
    are never extended.
    """
    prefix = str(row.get("candidate_prefix", "")).strip().lower()
    lines = row.get("ocr_lines") if isinstance(row.get("ocr_lines"), list) else []
    candidates = {str(tag).strip().lower() for tag in row.get("candidate_tags", []) if str(tag).strip()}
    numeric: list[tuple[str, tuple[float, float, float, float]]] = []
    for line in lines:
        text = " ".join(str(line.get("text", "")).strip().split())
        bounds = box_bounds(line)
        if bounds is not None and OCR_NUMERIC_SUFFIX.fullmatch(text):
            numeric.append((text, bounds))
    joined: list[dict[str, Any]] = []
    for line in lines:
        fragment = " ".join(str(line.get("text", "")).strip().split())
        bounds = box_bounds(line)
        if not prefix or bounds is None or not OCR_PREFIX_FRAGMENT.fullmatch(fragment):
            continue
        lowered = fragment.lower()
        if not lowered.startswith(prefix):
            continue
        final_hyphen_part = fragment.rsplit("-", 1)[-1] if "-" in fragment else ""
        needs_suffix = lowered == prefix or (final_hyphen_part.isdigit() and len(final_hyphen_part) <= 2)
        if not needs_suffix:
            continue
        left, top, right, bottom = bounds
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        center = (left + right) / 2
        eligible: list[tuple[float, str, tuple[float, float, float, float]]] = []
        for suffix, suffix_bounds in numeric:
            s_left, s_top, s_right, s_bottom = suffix_bounds
            s_width = max(1.0, s_right - s_left)
            s_height = max(1.0, s_bottom - s_top)
            vertical_gap = s_top - bottom
            horizontal_gap = abs((s_left + s_right) / 2 - center)
            if -0.35 * max(height, s_height) <= vertical_gap <= 3.0 * max(height, s_height):
                if horizontal_gap <= max(80.0, 1.25 * max(width, s_width)):
                    score = max(0.0, vertical_gap) + 0.25 * horizontal_gap
                    eligible.append((score, suffix, suffix_bounds))
        if not eligible:
            continue
        _, suffix, suffix_bounds = min(eligible, key=lambda value: (value[0], value[1]))
        candidates.discard(lowered)
        candidate = f"{lowered} {suffix}"
        candidates.add(candidate)
        joined.append({"fragment": fragment, "suffix": suffix, "candidate": candidate, "fragment_box": bounds, "suffix_box": suffix_bounds})
    derived = dict(row)
    derived["answer"] = ", ".join(sorted(candidates)) if candidates else "[]"
    derived["raw"] = derived["answer"]
    derived["candidate_tags"] = sorted(candidates)
    derived["geometry_joined_candidates"] = joined
    derived["postprocess"] = "reference-free vertical prefix/suffix join v1"
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--json", default="reports/generated/editorial_extension_experiments_v4.json")
    parser.add_argument("--csv", default="reports/generated/editorial_extension_experiments_v4.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    records = [row for row in read_rows(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl") if str(row["task"]) == "value"]
    if len(records) != 100:
        raise ValueError("Expected 100 hidden Set-B value records")

    ladder_dir = root / "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix"
    legacy_ladder_dir = root / "outputs/editorial_revision/internvl_counterfactual_ladder_v1"
    ocr_path = root / "outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl"
    required = {
        "internvl_correct": ladder_dir / "internvl35_8b_value_correct.jsonl",
        "internvl_shuffled": ladder_dir / "internvl35_8b_value_shuffled.jsonl",
        "internvl_text_only": ladder_dir / "internvl35_8b_value_text_only.jsonl",
        "paddleocr": ocr_path,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    legacy_paths = {
        "internvl_correct": legacy_ladder_dir / "internvl35_8b_value_correct.jsonl",
        "internvl_shuffled": legacy_ladder_dir / "internvl35_8b_value_shuffled.jsonl",
        "internvl_text_only": legacy_ladder_dir / "internvl35_8b_value_text_only.jsonl",
    }
    missing.extend(str(path) for path in legacy_paths.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"Missing extension outputs: {missing}")

    expected_ids = {str(row["instance_id"]) for row in records}
    public_manifests = [
        validate_public_manifest(
            root / "data/processed/main400_hashblind_set_b_remote_public.jsonl",
            expected_ids,
            shuffled=False,
        ),
        validate_public_manifest(
            root / "data/processed/main400_hashblind_set_b_shuffled_v1_remote_public.jsonl",
            expected_ids,
            shuffled=True,
        ),
    ]
    predictions = {
        label: validated_predictions(path, label, expected_ids)
        for label, path in required.items()
    }
    for label in ("internvl_correct", "internvl_shuffled", "internvl_text_only"):
        if any(row.get("tokenizer_fix_mistral_regex") is not True for row in predictions[label]):
            raise ValueError(f"{label} does not record tokenizer_fix_mistral_regex=true")
    tokenizer_correction = []
    for label, legacy_path in legacy_paths.items():
        legacy_rows = read_rows(legacy_path)
        legacy_by_id = {str(row.get("instance_id")): row for row in legacy_rows}
        fixed_by_id = {str(row.get("instance_id")): row for row in predictions[label]}
        if set(legacy_by_id) != set(fixed_by_id) or len(legacy_rows) != 100:
            raise ValueError(f"Legacy/fixed tokenizer membership mismatch for {label}")
        tokenizer_correction.append(
            {
                "condition": label,
                "legacy_path": f"outputs/editorial_revision/internvl_counterfactual_ladder_v1/{legacy_path.name}",
                "fixed_path": f"outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/{required[label].name}",
                "legacy_sha256": sha256(legacy_path),
                "fixed_sha256": sha256(required[label]),
                "raw_output_changed_count": sum(
                    str(legacy_by_id[key].get("raw", legacy_by_id[key].get("answer", "")))
                    != str(fixed_by_id[key].get("raw", fixed_by_id[key].get("answer", "")))
                    for key in legacy_by_id
                ),
            }
        )
    ocr_geometry = [geometry_joined_ocr_prediction(row) for row in predictions["paddleocr"]]

    cells = {
        "internvl35_8b_correct": score_cell(label="internvl35_8b_correct", records=records, predictions=predictions["internvl_correct"], metadata={"experiment": "X1", "family": "InternVL3.5-8B", "condition": "correct_image", "dynamic_preprocess_max_num": 12, "max_new_tokens": 512}),
        "internvl35_8b_shuffled": score_cell(label="internvl35_8b_shuffled", records=records, predictions=predictions["internvl_shuffled"], metadata={"experiment": "X1", "family": "InternVL3.5-8B", "condition": "source_shuffled_no_fixed_point", "dynamic_preprocess_max_num": 12, "max_new_tokens": 512}),
        "internvl35_8b_text_only": score_cell(label="internvl35_8b_text_only", records=records, predictions=predictions["internvl_text_only"], metadata={"experiment": "X1", "family": "InternVL3.5-8B", "condition": "text_only", "dynamic_preprocess_max_num": 0, "max_new_tokens": 512}),
        "paddleocr_literal_full_image": score_cell(label="paddleocr_literal_full_image", records=records, predictions=predictions["paddleocr"], metadata={"experiment": "X2", "family": "PaddleOCR 2.8.1 English (PP-OCRv3 detector; PP-OCRv4 recognizer)", "condition": "full_image_no_crop_literal_line_parse", "det_limit_side_len": 3072}),
        "paddleocr_full_image": score_cell(label="paddleocr_full_image", records=records, predictions=ocr_geometry, metadata={"experiment": "X2", "family": "PaddleOCR 2.8.1 English (PP-OCRv3 detector; PP-OCRv4 recognizer)", "condition": "full_image_no_crop_geometry_join_v1", "det_limit_side_len": 3072, "postprocess": "reference-free vertical prefix/suffix join v1"}),
    }
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    add_comparison(comparisons, transitions, "x1_internvl_correct_minus_shuffled", cells["internvl35_8b_shuffled"], cells["internvl35_8b_correct"], args.bootstrap_reps, 4101)
    add_comparison(comparisons, transitions, "x1_internvl_correct_minus_text_only", cells["internvl35_8b_text_only"], cells["internvl35_8b_correct"], args.bootstrap_reps, 4102)
    add_comparison(comparisons, transitions, "x1_internvl_shuffled_minus_text_only", cells["internvl35_8b_text_only"], cells["internvl35_8b_shuffled"], args.bootstrap_reps, 4103)
    payload = {
        "status": "pass",
        "experiments": {
            "X1": "InternVL3.5-8B correct/shuffled/text-only value ladder, tokenizer Mistral-regex fix enabled, 100 sources, 512-token cap",
            "X2": "PaddleOCR 2.8.1 / PaddlePaddle 2.6.2 English full-image value baseline (PP-OCRv3 detector; PP-OCRv4 recognizer), literal and reference-free geometry-joined parsing, 100 sources, detector max side 3072",
        },
        "records": {"task": "value", "record_count": len(records), "source_count": len({str(row["source_id"]) for row in records})},
        "input_boundary": {
            "status": "pass",
            "public_manifests": public_manifests,
            "output_reference_fields_present": False,
            "test_answer_used_true": False,
        },
        "ocr_postprocess": {
            "rule": "reference-free vertical prefix/suffix join v1",
            "records_with_join": sum(bool(row.get("geometry_joined_candidates")) for row in ocr_geometry),
            "joined_candidate_count": sum(len(row.get("geometry_joined_candidates", [])) for row in ocr_geometry),
            "literal_cell": "paddleocr_literal_full_image",
            "geometry_joined_cell": "paddleocr_full_image",
        },
        "internvl_tokenizer_correction": {
            "status": "pass",
            "legacy_warning": "Transformers reported an incorrect Mistral tokenizer regex pattern",
            "final_setting": "fix_mistral_regex=true",
            "legacy_scored_in_final_results": False,
            "conditions": tokenizer_correction,
        },
        "cells": {key: {field: value for field, value in cell.items() if field != "events"} for key, cell in cells.items()},
        "comparisons": comparisons,
        "paired_transitions": transitions,
        "bootstrap_reps": args.bootstrap_reps,
        "claim_policy": "Report all frozen conditions regardless of direction; no model-scale substitution claim is made.",
    }
    write_json(root / args.json, payload)
    rows: list[dict[str, Any]] = []
    for cell in cells.values():
        rows.extend(cell_table(cell))
    rows.extend({"row_type": "comparison", **row} for row in comparisons)
    csv_path = root / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "pass", "cells": len(cells), "comparisons": len(comparisons), "json": args.json, "csv": args.csv}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
