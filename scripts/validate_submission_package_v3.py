"""Validate the evidence-aligned v3 Results in Engineering submission package.

This validator is intentionally deterministic.  It verifies that the revised
manuscript, supplement, figures, compiled PDFs, central evidence artifacts,
and release metadata agree on the bounded claims in the v3 paper.  It does not
rerun model inference or reopen the blocked external-data branch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "20_POSITIVE_NARRATIVE_SELF_REVIEW_AND_REVISION_CHARTER.md",
    "CITATION.cff",
    "LICENSES.md",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/title_page.md",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_1_saber_pid_overview.pdf",
    "paper/figures/figure_1_saber_pid_overview.png",
    "paper/figures/figure_2_core_effects.pdf",
    "paper/figures/figure_2_core_effects.png",
    "paper/figures/figure_3_tag_reading_stability.pdf",
    "paper/figures/figure_3_tag_reading_stability.png",
    "paper/figures/figure_s1_task_calibration_and_boundaries.pdf",
    "paper/figures/figure_s1_task_calibration_and_boundaries.png",
    "paper/figures/figure_metadata_v3.json",
    "output/pdf/manuscript.pdf",
    "output/pdf/supplementary.pdf",
    "reports/generated/final_statistical_summary_v3.json",
    "reports/generated/final_claim_evidence_matrix_v3.csv",
    "reports/generated/manuscript_number_audit_v3.json",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/ontology_mapping_control_v1.json",
    "reports/generated/text_only_image_grounding_control_v1.json",
    "reports/generated/pdf_render_validation_v3.json",
    "reports/E7_ONTOLOGY_MAPPING_CONTROL_CLOSEOUT.md",
    "reports/E8_TEXT_ONLY_IMAGE_GROUNDING_CLOSEOUT.md",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text))


def extract_abstract(text: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S)
    if not match:
        raise ValueError("Missing abstract environment")
    return re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", match.group(1))


def latex_ids(text: str, command: str) -> set[str]:
    values: set[str] = set()
    for match in re.finditer(rf"\\{command}\{{([^}}]+)\}}", text):
        values.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/submission_package_validation_v3.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    missing_files = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    manuscript = (root / "paper/manuscript.tex").read_text(encoding="utf-8")
    supplementary = (root / "paper/supplementary.tex").read_text(encoding="utf-8")
    abstract_words = word_count(extract_abstract(manuscript))
    cited = latex_ids(manuscript, "cite")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    labels = latex_ids(manuscript, "label") | latex_ids(supplementary, "label")
    refs = latex_ids(manuscript, "ref") | latex_ids(supplementary, "ref")
    figure_refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript + supplementary)
    figures_missing = [reference for reference in figure_refs if not (root / "paper" / "figures" / reference).is_file()]
    highlight_lines = [
        line[2:].strip()
        for line in (root / "paper/highlights.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]

    summary = read_json(root / "reports/generated/final_statistical_summary_v3.json")
    number_audit = read_json(root / "reports/generated/manuscript_number_audit_v3.json")
    isolation = read_json(root / "reports/generated/evidence_input_answer_isolation_audit_v2.json")
    e7 = read_json(root / "reports/generated/ontology_mapping_control_v1.json")
    e8 = read_json(root / "reports/generated/text_only_image_grounding_control_v1.json")
    pdf_validation = read_json(root / "reports/generated/pdf_render_validation_v3.json")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    licenses = (root / "LICENSES.md").read_text(encoding="utf-8")

    required_tokens = (
        "19.5",
        "+0.5250",
        "+0.5487",
        "+0.5549",
        "+0.3100",
        "+0.2400",
        "+0.0000",
        "-0.0175",
    )
    missing_numeric_tokens = [token for token in required_tokens if token not in manuscript]
    citation_metadata_ok = (
        "pidqa-evidence-submission-v3" in citation
        and "repository-code:" not in citation
        and "license:" not in citation
        and "example.invalid" not in citation
    )
    evidence_status = {
        "final_summary": summary.get("status"),
        "number_audit": number_audit.get("status"),
        "answer_isolation": isolation.get("status"),
        "e7_mapping_control": e7.get("status"),
        "e8_text_only_control": e8.get("status"),
        "pdf_validation": pdf_validation.get("status"),
    }
    failure_reasons: list[str] = []
    if missing_files:
        failure_reasons.append("required_files_missing")
    if abstract_words > 250:
        failure_reasons.append("abstract_exceeds_250_words")
    if cited - bibliography:
        failure_reasons.append("citations_missing_from_bibliography")
    if bibliography - cited:
        failure_reasons.append("uncited_bibliography_entries")
    if refs - labels:
        failure_reasons.append("unresolved_latex_references")
    if figures_missing:
        failure_reasons.append("missing_included_figures")
    if missing_numeric_tokens:
        failure_reasons.append("required_evidence_numbers_missing")
    if not 3 <= len(highlight_lines) <= 5 or any(len(line) > 85 for line in highlight_lines):
        failure_reasons.append("highlights_outside_editorial_limits")
    if not citation_metadata_ok:
        failure_reasons.append("citation_metadata_incomplete_or_placeholder")
    if "InternVL" not in licenses or "weights are not stored" not in licenses:
        failure_reasons.append("license_inventory_missing_internvl_boundary")
    if any(status != "pass" for status in evidence_status.values()):
        failure_reasons.append("central_evidence_artifact_failed")

    report = {
        "validation_version": "submission-package-v3",
        "status": "fail" if failure_reasons else "pass",
        "failure_reasons": failure_reasons,
        "required_files": {"count": len(REQUIRED_FILES), "missing": missing_files},
        "abstract": {"word_count": abstract_words, "max_allowed": 250},
        "manuscript": {
            "word_count": word_count(manuscript),
            "supplement_word_count": word_count(supplementary),
            "citation_count": len(cited),
            "bibliography_count": len(bibliography),
            "missing_citations": sorted(cited - bibliography),
            "uncited_bibliography": sorted(bibliography - cited),
            "missing_refs": sorted(refs - labels),
            "figures_missing": figures_missing,
            "missing_required_numeric_tokens": missing_numeric_tokens,
        },
        "highlights": {
            "count": len(highlight_lines),
            "lengths": [len(line) for line in highlight_lines],
            "max_allowed_per_line": 85,
        },
        "citation_metadata_ok": citation_metadata_ok,
        "evidence_status": evidence_status,
        "claim_scope": {
            "mapping_attribution": "qualified by E7 cyclic-label permutation",
            "image_grounding": "supported for frozen Qwen value/tag reading by E3 and E8",
            "cross_family": "boundary control only; no universal visual-budget law",
            "external_data": "PID2Graph/OPEN100 score not reported",
        },
        "submitter_owned_placeholders": [
            "author names, affiliations, and corresponding-author details",
            "competing-interest and funding declarations",
            "journal originality/exclusive-submission confirmation",
            "permanent public archive URL",
        ],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
