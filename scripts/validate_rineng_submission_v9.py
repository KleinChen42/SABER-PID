"""Validate the acceptance-oriented Results in Engineering V9 submission.

The validator checks editorial limits, evidence visibility, deterministic
assets, LaTeX references, PDF layout metadata, and the separate all-page
visual-inspection record.  It performs no inference and changes no artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from validate_rineng_submission_v6 import expand_tex_inputs, latex_ids, words


TITLE = (
    "SABER-PID: Source-Isolated Qualification and Cost-Aware Operation of "
    "Vision--Language Models for P&ID Tag Retrieval"
)

VALIDATION_VERSION = "rineng-submission-validation-v9"
PDF_DIR = "output/pdf/v9"
RENDER_REPORT = "reports/generated/pdf_render_validation_v9.json"
VISUAL_REPORT = "reports/generated/pdf_visual_inspection_v9.json"

REQUIRED_FILES = (
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
    "paper/figures/figure_1_saber_pid_overview_v9.pdf",
    "paper/figures/figure_1_saber_pid_overview_v9.png",
    "paper/figures/figure_4_cost_aware_operation_v9.pdf",
    "paper/figures/figure_4_cost_aware_operation_v9.png",
    "paper/figures/figure_5_quality_and_budget_matched_v8.pdf",
    "paper/figures/figure_6_dexpi_external_v8.pdf",
    "paper/figures/figure_metadata_v9.json",
    "paper/tables/table_rineng_v9_qualification_scorecard.tex",
    "paper/tables/table_rineng_v9_operating_modes.tex",
    "reports/generated/rineng_v9_editorial_assets.json",
    "reports/generated/pdf_render_validation_v9.json",
    "reports/generated/pdf_visual_inspection_v9.json",
    "output/pdf/v9/manuscript.pdf",
    "output/pdf/v9/supplementary.pdf",
)

MAIN_FIGURES = {
    "figure_1_saber_pid_overview_v9.pdf",
    "figure_5_quality_and_budget_matched_v8.pdf",
    "figure_6_dexpi_external_v8.pdf",
    "figure_4_cost_aware_operation_v9.pdf",
}

MAIN_TABLES = {
    "tables/table_rineng_v9_qualification_scorecard.tex",
    "tables/table_rineng_v9_operating_modes.tex",
}

SUPPLEMENTARY_BOUNDARY_FIGURES = {
    "figure_s1_controls_and_operating_quantities_v4.pdf",
    "figure_2_qualification_effects_v6.pdf",
    "figure_3_operating_modes_v6.pdf",
    "figure_v7_cross_model_counterfactual_replication.pdf",
    "figure_s_v7_prompt_sensitivity.pdf",
}

PASS_REPORTS = (
    "reports/generated/rineng_revision_analysis_v6.json",
    "reports/generated/rineng_overnight_v7_score.json",
    "reports/generated/rineng_cost_sensitive_operating_modes_v8.json",
    "reports/generated/rineng_v8_extension_score.json",
    "reports/generated/rineng_v8_dexpi_external_score.json",
    "reports/generated/rineng_v8_independent_validation.json",
    "reports/generated/rineng_v8_paper_summary.json",
    "paper/figures/figure_metadata_v9.json",
    "reports/generated/rineng_v9_editorial_assets.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_title(value: str) -> str:
    return value.replace(r"\&", "&").strip()


def validate(root: Path) -> dict[str, Any]:
    failures: list[str] = []
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        failures.append("required_files_missing")

    manuscript_path = root / "paper/manuscript.tex"
    supplement_path = root / "paper/supplementary.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8") if manuscript_path.is_file() else ""
    supplement = supplement_path.read_text(encoding="utf-8") if supplement_path.is_file() else ""
    expanded = expand_tex_inputs(root, manuscript) + "\n" + expand_tex_inputs(root, supplement)

    title_match = re.search(r"\\title\{([^}]*)\}", manuscript)
    title = normalized_title(title_match.group(1)) if title_match else ""
    if title != TITLE:
        failures.append("title_mismatch")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", manuscript, flags=re.S
    )
    abstract_words = words(abstract_match.group(1)) if abstract_match else 0
    if not 150 <= abstract_words <= 250:
        failures.append("abstract_word_limit")

    keywords_match = re.search(r"\\textbf\{Keywords:\}\s*(.*?)\n\n", manuscript, flags=re.S)
    keywords = (
        [
            item.strip()
            for item in keywords_match.group(1).replace("\n", " ").split(";")
            if item.strip()
        ]
        if keywords_match
        else []
    )
    if not 1 <= len(keywords) <= 7:
        failures.append("keyword_count_outside_1_to_7")

    highlights_path = root / "paper/highlights.md"
    highlights = (
        [
            line[2:].strip()
            for line in highlights_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
        if highlights_path.is_file()
        else []
    )
    if not 3 <= len(highlights) <= 5 or any(len(value) > 85 for value in highlights):
        failures.append("highlights_outside_editorial_limits")

    cited = latex_ids(manuscript, "cite")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    labels = latex_ids(expanded, "label")
    references = latex_ids(expanded, "ref")
    if cited != bibliography:
        failures.append("citation_bibliography_mismatch")
    if references - labels:
        failures.append("undefined_latex_reference")
    if not {"alimin2025", "alimin2026", "zhu2026"}.issubset(cited):
        failures.append("recent_direct_work_not_cited")

    main_figures = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript
    )
    main_tables = re.findall(r"\\input\{(tables/[^}]+)\}", manuscript)
    supplement_figures = set(
        re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", supplement)
    )
    if len(main_figures) != 4 or set(main_figures) != MAIN_FIGURES:
        failures.append("main_figure_hierarchy_mismatch")
    if len(main_tables) != 2 or set(main_tables) != MAIN_TABLES:
        failures.append("main_table_hierarchy_mismatch")
    if SUPPLEMENTARY_BOUNDARY_FIGURES - supplement_figures:
        failures.append("material_boundary_figure_missing_from_supplement")

    boundary_tokens = (
        "0.0090--0.0257",
        "[-0.0078, 0.0764]",
        "Eight of nine intervals include zero",
        "no-image accuracy is 0.3025",
        "PID2Graph/OPEN100",
    )
    if any(token not in supplement for token in boundary_tokens):
        failures.append("material_boundary_result_missing")

    headline_tokens = (
        "0.5549",
        "0.0062",
        "+0.5250",
        "+0.4715",
        "0.9057",
        "0.7040",
        "0.9907",
        "2.0143",
    )
    if any(token not in expanded for token in headline_tokens):
        failures.append("headline_metric_missing")

    supporting_documents = (
        root / "paper/cover_letter.md",
        root / "paper/title_page.md",
        root / "CITATION.cff",
    )
    if any(path.is_file() and TITLE not in path.read_text(encoding="utf-8") for path in supporting_documents):
        failures.append("supporting_title_mismatch")
    if "Qualifying Image-Grounded Tag Retrieval" in "\n".join(
        path.read_text(encoding="utf-8")
        for path in supporting_documents + (manuscript_path, supplement_path)
        if path.is_file()
    ):
        failures.append("superseded_title_present")

    report_status: dict[str, str | None] = {}
    for relative in PASS_REPORTS:
        path = root / relative
        status = read_json(path).get("status") if path.is_file() else None
        report_status[relative] = status
        if status != "pass":
            failures.append("derived_report_not_pass")

    render_path = root / RENDER_REPORT
    visual_path = root / VISUAL_REPORT
    render = read_json(render_path) if render_path.is_file() else {}
    visual = read_json(visual_path) if visual_path.is_file() else {}
    if render.get("status") != "pass" or visual.get("status") != "pass":
        failures.append("pdf_validation_not_pass")
    render_documents = {str(row.get("name")): row for row in render.get("documents", [])}
    visual_hashes = visual.get("pdf_sha256", {})
    pdf_hashes: dict[str, str | None] = {}
    for name in ("manuscript", "supplementary"):
        path = root / f"{PDF_DIR}/{name}.pdf"
        observed = sha256(path) if path.is_file() else None
        pdf_hashes[name] = observed
        if (
            observed is None
            or render_documents.get(name, {}).get("sha256") != observed
            or visual_hashes.get(name) != observed
        ):
            failures.append("pdf_hash_validation_mismatch")
        if not str(render_documents.get(name, {}).get("metadata_title", "")).startswith(
            "SABER-PID" if name == "manuscript" else "Supplementary material: SABER-PID"
        ):
            failures.append("pdf_metadata_title_mismatch")

    visual_pages = visual.get("pages", [])
    rendered_pages = [
        path
        for document in render.get("documents", [])
        for path in document.get("rendered_pngs", [])
    ]
    if len(visual_pages) != len(rendered_pages) or {
        str(row.get("path")) for row in visual_pages
    } != set(rendered_pages):
        failures.append("pdf_page_inspection_incomplete")
    for row in visual_pages:
        path = root / str(row.get("path", ""))
        if (
            row.get("inspected") is not True
            or not path.is_file()
            or row.get("sha256") != (sha256(path) if path.is_file() else None)
        ):
            failures.append("pdf_page_inspection_hash_mismatch")
            break

    failures = list(dict.fromkeys(failures))
    administrative_placeholders = {
        "archive_doi_or_url": "[SUBMITTER: ARCHIVE DOI/URL]" in manuscript,
        "authors": "[SUBMITTER:" in (root / "paper/title_page.md").read_text(encoding="utf-8")
        if (root / "paper/title_page.md").is_file()
        else False,
        "funding": "[SUBMITTER:" in (root / "paper/declarations.md").read_text(encoding="utf-8")
        if (root / "paper/declarations.md").is_file()
        else False,
    }
    return {
        "version": VALIDATION_VERSION,
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "missing_files": missing,
        "title": title,
        "abstract_words": abstract_words,
        "keywords": {"count": len(keywords), "values": keywords},
        "highlights": {"count": len(highlights), "lengths": [len(x) for x in highlights]},
        "citations": {
            "cited": len(cited),
            "bibliography": len(bibliography),
            "missing": sorted(cited - bibliography),
            "uncited": sorted(bibliography - cited),
        },
        "main_figure_count": len(main_figures),
        "main_table_count": len(main_tables),
        "derived_status": report_status,
        "pdf_sha256": pdf_hashes,
        "rendered_page_count": len(rendered_pages),
        "inspected_page_count": len(visual_pages),
        "administrative_placeholders": administrative_placeholders,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="reports/generated/rineng_submission_validation_v9.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = validate(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
