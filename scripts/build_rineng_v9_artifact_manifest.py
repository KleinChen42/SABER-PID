"""Build the SHA-256 inventory for the validated SABER-PID RINENG V9 package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


IMMUTABLE_CATEGORIES = {
    "raw_prediction",
    "frozen_plan",
    "scorer_only_reference",
    "answer_isolated_manifest",
}

EXCLUDED_PREFIXES = (
    "paper/",
    "output/pdf/v8/",
)

EXCLUDED_EXACT = {
    "README.md",
    # The legacy V7 inventory fingerprints mutable workspace documents in
    # addition to the immutable prediction cells.  V9 imports and verifies
    # those raw cells directly below, so shipping the legacy inventory would
    # make the public archive depend on unrelated working-tree edits.
    "reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.json",
    "reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.csv",
    "reports/generated/rineng_submission_validation_v8.json",
    "reports/generated/pdf_render_validation_v8.json",
    "reports/generated/pdf_visual_inspection_v8.json",
}

V9_FILES = {
    "CITATION.cff",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/title_page.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_1_saber_pid_overview_v9.pdf",
    "paper/figures/figure_1_saber_pid_overview_v9.png",
    "paper/figures/figure_4_cost_aware_operation_v9.pdf",
    "paper/figures/figure_4_cost_aware_operation_v9.png",
    "paper/figures/figure_5_quality_and_budget_matched_v8.pdf",
    "paper/figures/figure_5_quality_and_budget_matched_v8.png",
    "paper/figures/figure_6_dexpi_external_v8.pdf",
    "paper/figures/figure_6_dexpi_external_v8.png",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.png",
    "paper/figures/figure_2_qualification_effects_v6.pdf",
    "paper/figures/figure_2_qualification_effects_v6.png",
    "paper/figures/figure_3_operating_modes_v6.pdf",
    "paper/figures/figure_3_operating_modes_v6.png",
    "paper/figures/figure_v7_cross_model_counterfactual_replication.pdf",
    "paper/figures/figure_v7_cross_model_counterfactual_replication.png",
    "paper/figures/figure_s_v7_prompt_sensitivity.pdf",
    "paper/figures/figure_s_v7_prompt_sensitivity.png",
    "paper/figures/figure_metadata_v6.json",
    "paper/figures/figure_metadata_v7.json",
    "paper/figures/figure_metadata_v8.json",
    "paper/figures/figure_metadata_v9.json",
    "paper/tables/table_rineng_v9_qualification_scorecard.tex",
    "paper/tables/table_rineng_v9_operating_modes.tex",
    "paper/tables/table_rineng_overnight_v7_counterfactual.tex",
    "paper/tables/table_rineng_overnight_v7_task_accuracy.tex",
    "paper/tables/table_rineng_v8_quality_by_subset.tex",
    "paper/tables/table_rineng_v8_internvl_budget54.tex",
    "output/pdf/v9/manuscript.pdf",
    "output/pdf/v9/supplementary.pdf",
    "reports/generated/rineng_v9_editorial_assets.json",
    "reports/generated/reproduction_validation_v9.json",
    "reports/generated/pdf_render_validation_v9.json",
    "reports/generated/pdf_visual_inspection_v9.json",
    "reports/generated/rineng_submission_validation_v9.json",
    "scripts/build_rineng_v9_editorial_assets.py",
    "scripts/reproduce_rineng_submission_v9.py",
    "scripts/record_pdf_visual_inspection_v9.py",
    "scripts/validate_rineng_submission_v9.py",
    "scripts/build_rineng_v9_artifact_manifest.py",
    "scripts/build_rineng_public_release_v9.py",
    "scripts/validate_rineng_public_release_v9.py",
    "tests/test_build_rineng_v9_editorial_assets.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int | str:
    if path.suffix != ".jsonl":
        return ""
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def category(path: str, inherited: str | None = None) -> str:
    if inherited:
        return inherited
    if path.startswith("paper/figures/"):
        return "paper_figure"
    if path.startswith("paper/tables/"):
        return "paper_table"
    if path.startswith("paper/"):
        return "submission_source"
    if path.startswith("output/pdf/v9/"):
        return "submission_pdf"
    if path.startswith(("scripts/", "tests/")):
        return "code_or_test"
    return "derived_result_or_document"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--base", default="reports/RINENG_V8_ARTIFACT_MANIFEST.json"
    )
    parser.add_argument(
        "--v7-base", default="reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.json"
    )
    parser.add_argument("--output", default="reports/RINENG_V9_ARTIFACT_MANIFEST.json")
    parser.add_argument("--csv", default="reports/RINENG_V9_ARTIFACT_MANIFEST.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    base = json.loads((root / args.base).read_text(encoding="utf-8"))
    if base.get("status") != "pass":
        raise SystemExit("Base V8 inventory must pass")

    inherited: dict[str, str] = {}
    selected: set[str] = set()
    immutable_verified = 0
    for row in base["artifacts"]:
        relative = str(row["path"])
        path = root / relative
        inherited[relative] = str(row["category"])
        if row["category"] in IMMUTABLE_CATEGORIES:
            if (
                not path.is_file()
                or path.stat().st_size != int(row["bytes"])
                or sha256(path) != row["sha256"]
            ):
                raise SystemExit(f"Immutable evidence differs from V8 inventory: {relative}")
            immutable_verified += 1
        if relative in EXCLUDED_EXACT or relative.startswith(EXCLUDED_PREFIXES):
            continue
        if path.is_file():
            selected.add(relative)

    # The V8 package used three native-budget V7 cells for its paired
    # comparison but did not copy the other 51 immutable V7 prediction cells.
    # V9 closes that release gap so the advertised complete 54-cell matrix can
    # be rescored from raw outputs rather than only checked against a hash list.
    v7 = json.loads((root / args.v7_base).read_text(encoding="utf-8"))
    if v7.get("status") != "pass":
        raise SystemExit("Base V7 inventory must pass")
    for row in v7["artifacts"]:
        if row.get("category") not in {"raw_prediction", "frozen_plan"}:
            continue
        relative = str(row["path"])
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise SystemExit(f"Immutable V7 evidence differs from inventory: {relative}")
        inherited[relative] = str(row["category"])
        selected.add(relative)
        immutable_verified += 1

    selected.update(V9_FILES)
    missing = sorted(relative for relative in selected if not (root / relative).is_file())
    if missing:
        raise SystemExit(f"Required V9 artifacts are missing: {missing}")

    rows = []
    for relative in sorted(selected):
        path = root / relative
        rows.append(
            {
                "path": relative,
                "category": category(relative, inherited.get(relative)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": line_count(path),
            }
        )
    report = {
        "version": "rineng-v9-artifact-manifest",
        "status": "pass",
        "base_manifest": args.base,
        "v7_base_manifest": args.v7_base,
        "immutable_base_artifact_count_verified": immutable_verified,
        "artifact_count": len(rows),
        "submission_pdf_count": sum(row["category"] == "submission_pdf" for row in rows),
        "raw_prediction_count": sum(row["category"] == "raw_prediction" for row in rows),
        "raw_prediction_rows": sum(
            int(row["rows"] or 0) for row in rows if row["category"] == "raw_prediction"
        ),
        "artifacts": rows,
    }
    output = root / args.output
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
                "raw_prediction_count": report["raw_prediction_count"],
                "raw_prediction_rows": report["raw_prediction_rows"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
