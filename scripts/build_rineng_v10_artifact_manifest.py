"""Build the SHA-256 inventory for the validated SABER-PID RINENG V10 release."""

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
EXCLUDED_PREFIXES = ("paper/", "output/pdf/v9/", "release/")
EXCLUDED_EXACT = {
    "CITATION.cff",
    "LICENSES.md",
    "reports/generated/reproduction_validation_v9.json",
    "reports/generated/pdf_render_validation_v9.json",
    "reports/generated/pdf_visual_inspection_v9.json",
    "reports/generated/rineng_submission_validation_v9.json",
}
V10_FILES = {
    "CITATION.cff",
    "LICENSE",
    "LICENSES.md",
    "pyproject.toml",
    "requirements-analysis-v6.txt",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/title_page.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/Declaration_of_Interests.docx",
    "paper/Highlights.docx",
    "paper/Title_Page.docx",
    "paper/CRediT_Authorship_Statement.docx",
    "paper/Author_Information.docx",
    "paper/SUBMISSION_FILE_INDEX.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_metadata_v10.json",
    "output/pdf/v10/manuscript.pdf",
    "output/pdf/v10/supplementary.pdf",
    "reports/generated/rineng_v10_publication_figures.json",
    "reports/generated/rineng_figure_layout_audit_v10.json",
    "reports/generated/reproduction_validation_v10.json",
    "reports/generated/pdf_render_validation_v10.json",
    "reports/generated/pdf_visual_inspection_v10.json",
    "reports/generated/rineng_submission_validation_v10.json",
    "scripts/build_rineng_v10_publication_figures.py",
    "scripts/reproduce_rineng_submission_v10.py",
    "scripts/record_pdf_visual_inspection_v10.py",
    "scripts/validate_rineng_submission_v10.py",
    "scripts/build_declaration_of_interests_docx.py",
    "scripts/build_rineng_submission_editorial_documents.py",
    "scripts/build_rineng_v10_submission_package.py",
    "scripts/build_rineng_v10_artifact_manifest.py",
    "scripts/build_rineng_public_release_v10.py",
    "scripts/validate_rineng_public_release_v10.py",
    "tests/test_build_rineng_v10_publication_figures.py",
}
V10_FILES.update(
    f"paper/figures/{stem}.{suffix}"
    for stem in (
        "figure_1_saber_pid_overview_v10",
        "figure_2_quality_and_budget_v10",
        "figure_3_dexpi_external_v10",
        "figure_4_cost_aware_operation_v10",
        "figure_s1_boundary_controls_v10",
        "figure_s2_qualification_effects_v10",
        "figure_s3_operating_modes_v10",
        "figure_s4_cross_model_replication_v10",
        "figure_s5_prompt_sensitivity_v10",
    )
    for suffix in ("pdf", "png")
)
V10_FILES.update(
    f"paper/tables/{name}"
    for name in (
        "table_rineng_v9_qualification_scorecard.tex",
        "table_rineng_v9_operating_modes.tex",
        "table_rineng_overnight_v7_counterfactual.tex",
        "table_rineng_overnight_v7_task_accuracy.tex",
        "table_rineng_v8_quality_by_subset.tex",
        "table_rineng_v8_internvl_budget54.tex",
    )
)


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


def category(relative: str, inherited: str | None = None) -> str:
    if inherited:
        return inherited
    if relative.startswith("paper/figures/"):
        return "paper_figure"
    if relative.startswith("paper/tables/"):
        return "paper_table"
    if relative.startswith("paper/"):
        return "submission_source"
    if relative.startswith("output/pdf/v10/"):
        return "submission_pdf"
    if relative == "LICENSE" or relative.startswith("licenses/"):
        return "license"
    if relative.startswith(("scripts/", "tests/", "src/")):
        return "code_or_test"
    return "derived_result_or_document"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--base", default="reports/RINENG_V9_ARTIFACT_MANIFEST.json")
    parser.add_argument("--output", default="reports/RINENG_V10_ARTIFACT_MANIFEST.json")
    parser.add_argument("--csv", default="reports/RINENG_V10_ARTIFACT_MANIFEST.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    base = json.loads((root / args.base).read_text(encoding="utf-8"))
    if base.get("status") != "pass":
        raise SystemExit("Base V9 inventory must pass")
    selected: set[str] = set()
    inherited: dict[str, str] = {}
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
                raise SystemExit(f"Immutable evidence differs from V9 inventory: {relative}")
            immutable_verified += 1
        if relative in EXCLUDED_EXACT or relative.startswith(EXCLUDED_PREFIXES):
            continue
        if path.is_file():
            selected.add(relative)
    selected.update(V10_FILES)
    missing = sorted(relative for relative in selected if not (root / relative).is_file())
    if missing:
        raise SystemExit(f"Required V10 artifacts are missing: {missing}")
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
        "version": "rineng-v10-artifact-manifest",
        "status": "pass",
        "base_manifest": args.base,
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
    with (root / args.csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "artifact_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
