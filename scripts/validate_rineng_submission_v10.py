"""Validate the V10 publication-figure SABER-PID submission package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import validate_rineng_submission_v9 as base


FIGURE_STEMS = (
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


def configure() -> None:
    base.VALIDATION_VERSION = "rineng-submission-validation-v10"
    base.PDF_DIR = "output/pdf/v10"
    base.RENDER_REPORT = "reports/generated/pdf_render_validation_v10.json"
    base.VISUAL_REPORT = "reports/generated/pdf_visual_inspection_v10.json"
    base.MAIN_FIGURES = {
        "figure_1_saber_pid_overview_v10.pdf",
        "figure_2_quality_and_budget_v10.pdf",
        "figure_3_dexpi_external_v10.pdf",
        "figure_4_cost_aware_operation_v10.pdf",
    }
    base.SUPPLEMENTARY_BOUNDARY_FIGURES = {
        "figure_s1_boundary_controls_v10.pdf",
        "figure_s2_qualification_effects_v10.pdf",
        "figure_s3_operating_modes_v10.pdf",
        "figure_s4_cross_model_replication_v10.pdf",
        "figure_s5_prompt_sensitivity_v10.pdf",
    }
    figure_files = tuple(
        f"paper/figures/{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("pdf", "png")
    )
    base.REQUIRED_FILES = (
        "CITATION.cff",
        "23_RINENG_ACCEPTANCE_ORIENTED_SELF_REVIEW_AND_AUTOMATIC_REVISION_CHARTER.md",
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
        "LICENSE",
        "paper/figure_manifest.md",
        "paper/figure_captions.md",
        "paper/figures/figure_metadata_v10.json",
        "paper/tables/table_rineng_v9_qualification_scorecard.tex",
        "paper/tables/table_rineng_v9_operating_modes.tex",
        "reports/RINENG_V10_FIGURE_VISUAL_QA.md",
        "reports/generated/rineng_v10_publication_figures.json",
        "reports/generated/rineng_figure_layout_audit_v10.json",
        base.RENDER_REPORT,
        base.VISUAL_REPORT,
        f"{base.PDF_DIR}/manuscript.pdf",
        f"{base.PDF_DIR}/supplementary.pdf",
        *figure_files,
    )
    base.PASS_REPORTS = (
        "reports/generated/rineng_revision_analysis_v6.json",
        "reports/generated/rineng_overnight_v7_score.json",
        "reports/generated/rineng_cost_sensitive_operating_modes_v8.json",
        "reports/generated/rineng_v8_extension_score.json",
        "reports/generated/rineng_v8_dexpi_external_score.json",
        "reports/generated/rineng_v8_independent_validation.json",
        "reports/generated/rineng_v8_paper_summary.json",
        "paper/figures/figure_metadata_v10.json",
        "reports/generated/rineng_v10_publication_figures.json",
        "reports/generated/rineng_figure_layout_audit_v10.json",
    )


EXPECTED_AUTHORS = (
    "Zhuo Chen",
    "Shuhao Liu",
    "Zhi Ling",
    "Yu Yan",
    "Qiuxue Wu",
    "Ziyi Kuang",
    "Zihan Zhao",
    "Caixin Tan",
    "Haiyou Zhang",
)
EXPECTED_AFFILIATIONS = (
    "Harbin University of Science and Technology",
    "Chalmers University of Technology",
    "Kiwiar Co., Ltd.",
    "Nanjing Agricultural University",
    "Hunan University",
)


def validate_administrative_metadata(root: Path, report: dict[str, object]) -> None:
    failures = list(report.get("failure_reasons", []))
    manuscript = (root / "paper/manuscript.tex").read_text(encoding="utf-8")
    title_page = (root / "paper/title_page.md").read_text(encoding="utf-8")
    declarations = (root / "paper/declarations.md").read_text(encoding="utf-8")
    supplementary = (root / "paper/supplementary.tex").read_text(encoding="utf-8")
    data_availability = (root / "paper/data_availability.md").read_text(encoding="utf-8")
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    combined = "\n".join((manuscript, title_page, declarations, cff))

    missing_authors = [name for name in EXPECTED_AUTHORS if name not in combined]
    missing_affiliations = [
        name for name in EXPECTED_AFFILIATIONS if name not in combined
    ]
    if missing_authors:
        failures.append("author_metadata_missing")
    if missing_affiliations:
        failures.append("affiliation_metadata_missing")
    if "zhuoc@chalmers.se" not in combined or "Corresponding author" not in combined:
        failures.append("corresponding_author_metadata_missing")
    if "no known competing financial interests" not in declarations:
        failures.append("competing_interest_declaration_missing")
    if "received no specific grant" not in declarations:
        if "did not receive any specific grant" not in declarations:
            failures.append("funding_declaration_missing")
    if "Zhuo Chen\\textsuperscript{a,*}" not in manuscript:
        failures.append("zhuo_affiliation_mapping_incorrect")
    if "Zhuo Chen\\textsuperscript{a,*}" not in supplementary:
        failures.append("supplementary_author_metadata_missing")
    if "Shuhao Liu\\textsuperscript{b}" not in manuscript or "Qiuxue Wu\\textsuperscript{b}" not in manuscript:
        failures.append("harbin_affiliation_mapping_incorrect")
    for address in (
        "2 Jalan Mat Jambol",
        "No. 1 Weigang",
        "Lushan South Road",
    ):
        if address not in manuscript or address not in supplementary:
            failures.append("full_postal_address_missing")
            break
    if "https://github.com/KleinChen42/SABER-PID" not in (
        manuscript + data_availability + cff
    ):
        failures.append("public_repository_url_missing")
    if re.search(r"(?i)zenodo|archival\s+DOI|archive\s+DOI|after\s+it\s+is\s+minted", manuscript + data_availability):
        failures.append("doi_or_zenodo_declaration_remaining")
    if not (root / "LICENSE").is_file() or "MIT License" not in (
        root / "LICENSE"
    ).read_text(encoding="utf-8"):
        failures.append("project_code_license_missing")
    normalized_manuscript = re.sub(r"\s+", " ", manuscript)
    if "OpenAI Codex" not in normalized_manuscript or "ChatGPT Pro" not in normalized_manuscript:
        failures.append("ai_tool_identification_missing")
    placeholder_files = (
        root / "paper/manuscript.tex",
        root / "paper/supplementary.tex",
        root / "paper/title_page.md",
        root / "paper/data_availability.md",
        root / "paper/declarations.md",
        root / "paper/cover_letter.md",
    )
    if any("[SUBMITTER:" in path.read_text(encoding="utf-8") for path in placeholder_files):
        failures.append("administrative_placeholder_remaining")
    credit_section = re.search(
        r"\\section\*\{CRediT authorship contribution statement\}(.*?)"
        r"\\section\*\{Declaration of generative AI",
        manuscript,
        flags=re.S,
    )
    if credit_section is None or any(
        name not in credit_section.group(1) for name in EXPECTED_AUTHORS
    ):
        failures.append("credit_statement_incomplete")
    for required_docx in (
        "paper/Declaration_of_Interests.docx",
        "paper/Highlights.docx",
        "paper/Title_Page.docx",
        "paper/CRediT_Authorship_Statement.docx",
        "paper/Author_Information.docx",
    ):
        if not (root / required_docx).is_file() or (root / required_docx).stat().st_size == 0:
            failures.append("editable_submission_component_missing")
            break

    report["failure_reasons"] = list(dict.fromkeys(failures))
    report["status"] = "pass" if not report["failure_reasons"] else "fail"
    report["authors"] = {
        "count": len(EXPECTED_AUTHORS),
        "missing": missing_authors,
        "corresponding_author": "Zhuo Chen",
    }
    report["affiliations"] = {
        "count": len(EXPECTED_AFFILIATIONS),
        "missing": missing_affiliations,
    }
    report["administrative_placeholders"] = {
        "github_only_data_link": "https://github.com/KleinChen42/SABER-PID" in (manuscript + data_availability),
        "authors": False,
        "funding": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="reports/generated/rineng_submission_validation_v10.json"
    )
    args = parser.parse_args()
    configure()
    root = Path(args.root).resolve()
    report = base.validate(root)
    validate_administrative_metadata(root, report)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
