"""Validate the V10 publication-figure SABER-PID submission package."""

from __future__ import annotations

import argparse
import json
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
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
