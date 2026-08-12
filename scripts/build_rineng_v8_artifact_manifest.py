"""Build the final SHA-256 inventory for the validated RINENG V8 extension."""

from __future__ import annotations

import argparse
import csv
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def artifact_category(relative: Path) -> str:
    value = relative.as_posix()
    if value.startswith("outputs/rineng_v8/"):
        return "raw_prediction"
    if value.startswith("data/manifests/"):
        return "frozen_plan"
    if value.startswith("data/answer_store/"):
        return "scorer_only_reference"
    if value.startswith("data/processed/"):
        return "answer_isolated_manifest"
    if value.startswith("paper/figures/"):
        return "paper_figure"
    if value.startswith("paper/tables/"):
        return "paper_table"
    if value.startswith("output/pdf/v8/"):
        return "submission_pdf"
    if value.startswith("scripts/") or value.startswith("tests/"):
        return "code_or_test"
    if value.startswith("licenses/"):
        return "license"
    if value.startswith("reports/logs/"):
        return "execution_log"
    return "derived_result_or_document"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="reports/RINENG_V8_ARTIFACT_MANIFEST.json"
    )
    parser.add_argument(
        "--csv", default="reports/RINENG_V8_ARTIFACT_MANIFEST.csv"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    extension_path = root / "reports/generated/rineng_v8_extension_score.json"
    external_path = root / "reports/generated/rineng_v8_dexpi_external_score.json"
    validation_path = root / "reports/generated/rineng_v8_independent_validation.json"
    figure_metadata_path = root / "paper/figures/figure_metadata_v8.json"
    summary_path = root / "reports/generated/rineng_v8_paper_summary.json"
    cost_path = root / "reports/generated/rineng_cost_sensitive_operating_modes_v8.json"
    documents = {
        "extension": read_json(extension_path),
        "external": read_json(external_path),
        "validation": read_json(validation_path),
        "figures": read_json(figure_metadata_path),
        "summary": read_json(summary_path),
        "cost": read_json(cost_path),
    }
    quality_plan = read_json(root / "data/manifests/rineng_v8_quality_robustness_plan.json")
    internvl_plan = read_json(root / "data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json")
    external_plan = read_json(root / "data/manifests/rineng_v8_dexpi_external_plan.json")
    failed = [key for key, value in documents.items() if value.get("status") != "pass"]
    if failed:
        raise SystemExit(f"Refusing to inventory failing artifacts: {failed}")

    raw_groups = {
        "quality": sorted((root / "outputs/rineng_v8/qwen3vl8b_quality").glob("*.jsonl")),
        "internvl_budget54": sorted(
            (root / "outputs/rineng_v8/internvl35_8b_budget54").glob("*.jsonl")
        ),
        "dexpi_qwen": sorted(
            (root / "outputs/rineng_v8/dexpi_external_qwen").glob("*.jsonl")
        ),
        "dexpi_ocr": [root / "outputs/rineng_v8/dexpi_external_ocr.jsonl"],
    }
    expected = {
        "quality": (24, 7_360),
        "internvl_budget54": (9, 2_760),
        "dexpi_qwen": (3, 195),
        "dexpi_ocr": (1, 65),
    }
    for key, paths in raw_groups.items():
        if any(not path.is_file() for path in paths):
            raise SystemExit(f"Missing raw artifact in {key}")
        actual = (len(paths), sum(line_count(path) for path in paths))
        if actual != expected[key]:
            raise SystemExit(f"Unexpected raw scope for {key}: {actual} vs {expected[key]}")

    expected_hashes = {
        str(cell["path"]): str(cell["sha256"])
        for cell in documents["extension"]["cells"].values()
        if cell.get("status") == "pass"
    }
    expected_hashes.update(
        {
            str(cell["path"]): str(cell["sha256"])
            for cell in documents["external"]["integrity"]["cells"].values()
        }
    )
    for paths in raw_groups.values():
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if expected_hashes.get(relative) != sha256(path):
                raise SystemExit(f"Raw prediction hash differs from score report: {relative}")

    explicit = [
        "README.md",
        "22_RINENG_V8_HIGH_VALUE_EVIDENCE_AND_SUBMISSION_CLOSEOUT.md",
        "LICENSES.md",
        "CITATION.cff",
        "pyproject.toml",
        "requirements-analysis-v6.txt",
        "licenses/DEXPI_TRAINING_TEST_CASES_LICENSE.txt",
        "licenses/PIDQA_LICENSE.txt",
        "data/processed/pidqa_records.jsonl",
        "data/manifests/rineng_v8_quality_robustness_plan.json",
        "data/manifests/rineng_v8_quality_images.json",
        "data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json",
        "data/manifests/rineng_v8_dexpi_external_plan.json",
        "data/answer_store/rineng_v8_dexpi_external_hidden.jsonl",
        "data/processed/rineng_v8_dexpi_external/dexpi_external_v8_correct_public.jsonl",
        "data/processed/rineng_v8_dexpi_external/dexpi_external_v8_shuffled_public.jsonl",
        "reports/generated/rineng_cost_sensitive_operating_modes_v8.json",
        "reports/generated/rineng_cost_sensitive_operating_modes_v8.csv",
        "reports/generated/rineng_cost_sensitive_decision_rule_v8.csv",
        "reports/generated/rineng_revision_per_source_v6.csv",
        "reports/generated/rineng_v8_extension_score.json",
        "reports/generated/rineng_v8_extension_score.csv",
        "reports/generated/rineng_v8_dexpi_external_score.json",
        "reports/generated/rineng_v8_dexpi_external_score.csv",
        "reports/generated/rineng_v8_independent_validation.json",
        "reports/generated/rineng_v8_paper_summary.json",
        "reports/generated/rineng_v8_quality_effects.csv",
        "reports/generated/rineng_v8_internvl_budget_effects.csv",
        "reports/generated/rineng_v8_dexpi_external_audit.json",
        "reports/generated/rineng_v8_pid2graph_open100_audit.json",
        "reports/generated/pid2graph_open100_complete_materialized_v8.json",
        "reports/RINENG_V8_PUBLIC_BACKUP_STATUS.md",
        "reports/RINENG_V8_H200_MAINTENANCE_RESUME.md",
        "reports/RINENG_V8_CLOSEOUT.md",
        "paper/manuscript.tex",
        "paper/supplementary.tex",
        "paper/figure_manifest.md",
        "paper/figure_captions.md",
        "paper/highlights.md",
        "paper/cover_letter.md",
        "paper/data_availability.md",
        "paper/figures/figure_4_cost_sensitive_operating_modes_v8.pdf",
        "paper/figures/figure_4_cost_sensitive_operating_modes_v8.png",
        "paper/figures/figure_1_qualification_decision_v6.pdf",
        "paper/figures/figure_1_qualification_decision_v6.png",
        "paper/figures/figure_2_qualification_effects_v6.pdf",
        "paper/figures/figure_2_qualification_effects_v6.png",
        "paper/figures/figure_v7_cross_model_counterfactual_replication.pdf",
        "paper/figures/figure_v7_cross_model_counterfactual_replication.png",
        "paper/figures/figure_3_operating_modes_v6.pdf",
        "paper/figures/figure_3_operating_modes_v6.png",
        "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
        "paper/figures/figure_s1_controls_and_operating_quantities_v4.png",
        "paper/figures/figure_s_v7_prompt_sensitivity.pdf",
        "paper/figures/figure_s_v7_prompt_sensitivity.png",
        "paper/figures/figure_5_quality_and_budget_matched_v8.pdf",
        "paper/figures/figure_5_quality_and_budget_matched_v8.png",
        "paper/figures/figure_6_dexpi_external_v8.pdf",
        "paper/figures/figure_6_dexpi_external_v8.png",
        "paper/figures/figure_metadata_v8.json",
        "paper/tables/table_rineng_v8_quality.tex",
        "paper/tables/table_rineng_v8_internvl_budget54.tex",
        "paper/tables/table_rineng_v8_dexpi_external.tex",
        "output/pdf/v8/manuscript.pdf",
        "output/pdf/v8/supplementary.pdf",
        "scripts/audit_pid2graph_open100_v8.py",
        "scripts/build_cost_sensitive_operating_modes_v8.py",
        "scripts/build_rineng_v8_extension_figures.py",
        "scripts/build_rineng_v8_tables.py",
        "scripts/fetch_remote_zip_subset.py",
        "scripts/prepare_dexpi_external_v8.py",
        "scripts/prepare_quality_robustness_v8.py",
        "scripts/run_e1_evidence_audit.py",
        "scripts/score_rineng_overnight_v7.py",
        "scripts/run_internvl_budget_matched_v8.py",
        "scripts/run_paddleocr_external_v8.py",
        "scripts/run_qwen_counterfactual_quality_v8.py",
        "scripts/score_dexpi_external_v8.py",
        "scripts/score_rineng_v8_extensions.py",
        "scripts/validate_rineng_v8_extensions.py",
        "scripts/reproduce_rineng_v8_extensions.py",
        "scripts/build_rineng_v8_artifact_manifest.py",
        "scripts/build_rineng_public_release_v8.py",
        "scripts/validate_rineng_public_release_v8.py",
        "tests/test_audit_pid2graph_open100_v8.py",
        "tests/test_build_rineng_v8_extension_figures.py",
        "tests/test_build_rineng_v8_tables.py",
        "tests/test_cost_sensitive_operating_modes_v8.py",
        "tests/test_prepare_dexpi_external_v8.py",
        "tests/test_rineng_v8_extension_preparation.py",
        "tests/test_run_internvl_budget_matched_v8.py",
        "tests/test_score_dexpi_external_v8.py",
        "tests/test_score_rineng_v8_extensions.py",
        "tests/test_validate_rineng_v8_extensions.py",
        "tests/test_reproduce_rineng_v8_extensions.py",
        "tests/test_build_rineng_v8_artifact_manifest.py",
        "tests/test_rineng_public_release_v8.py",
    ]
    paths = {path for values in raw_groups.values() for path in values}
    for relative in explicit:
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Required V8 artifact is missing: {relative}")
        paths.add(path)
    # Include every answer-isolated manifest named by the frozen plans rather
    # than maintaining a second hand-written list.
    for plan in (quality_plan, internvl_plan, external_plan):
        for spec in plan["datasets"]:
            for field in ("correct_input", "shuffled_input"):
                if field in spec:
                    path = root / spec[field]
                    if not path.is_file():
                        raise SystemExit(f"Frozen input manifest is missing: {spec[field]}")
                    paths.add(path)
    # The 54-tile-minus-native-12 comparison depends on these three immutable
    # V7 correct-image cells; they are inputs to scoring, not new V8 inference.
    for spec in internvl_plan["datasets"]:
        path = root / "outputs/rineng_overnight_v7/internvl35_8b" / (
            f"internvl35_8b_{spec['dataset_id']}_p0_correct_tiles12.jsonl"
        )
        if not path.is_file():
            raise SystemExit(f"Native-12 comparison input is missing: {path}")
        paths.add(path)
    paths.update(path for path in (root / "src").rglob("*.py") if path.is_file())
    paths.update((root / "reports/logs/rineng_v8_backup").glob("*"))

    rows = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "path": relative.as_posix(),
                "category": artifact_category(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": line_count(path) if path.suffix == ".jsonl" else "",
            }
        )
    report = {
        "version": "rineng-v8-artifact-manifest",
        "status": "pass",
        "integrity": {
            "raw_file_count": sum(len(value) for value in raw_groups.values()),
            "raw_row_count": sum(value[1] for value in expected.values()),
            "raw_bytes": sum(
                path.stat().st_size for values in raw_groups.values() for path in values
            ),
            "independent_validation_status": documents["validation"]["status"],
            "independent_validation_error_count": documents["validation"]["error_count"],
            "max_point_absolute_error": documents["validation"]["numeric_agreement"][
                "max_point_absolute_error"
            ],
            "max_independent_ci_endpoint_absolute_difference": documents["validation"][
                "numeric_agreement"
            ]["max_independent_ci_endpoint_absolute_difference"],
        },
        "public_backup_record": "reports/RINENG_V8_PUBLIC_BACKUP_STATUS.md",
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = root / args.csv
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_count": len(rows),
                "raw_file_count": report["integrity"]["raw_file_count"],
                "raw_row_count": report["integrity"]["raw_row_count"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
