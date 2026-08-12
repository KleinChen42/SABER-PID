"""Validate the v4 Results in Engineering submission package and claim scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "CITATION.cff", "LICENSES.md", "requirements-lock.txt",
    "review/SABER_PID_model_based_editorial_review.md",
    "licenses/PIDQA_LICENSE.txt",
    "reports/MODEL_BASED_EDITORIAL_REVISION_CLOSEOUT_V4.md",
    "paper/manuscript.tex", "paper/supplementary.tex",
    "paper/templates/manuscript_v4.tex.in", "paper/templates/supplementary_v4.tex.in",
    "paper/title_page.md", "paper/highlights.md", "paper/cover_letter.md",
    "paper/data_availability.md", "paper/declarations.md", "paper/figure_manifest.md", "paper/figure_captions.md",
    "paper/figures/figure_1_counterfactual_evidence_ladder.pdf", "paper/figures/figure_1_counterfactual_evidence_ladder.png",
    "paper/figures/figure_2_core_effects_v4.pdf", "paper/figures/figure_2_core_effects_v4.png",
    "paper/figures/figure_3_task_calibration_v4.pdf", "paper/figures/figure_3_task_calibration_v4.png",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf", "paper/figures/figure_s1_controls_and_operating_quantities_v4.png",
    "paper/figures/figure_s2_tag_reading_stability_v4.pdf", "paper/figures/figure_s2_tag_reading_stability_v4.png",
    "paper/figures/figure_metadata_v4.json",
    "paper/assets/pidqa_sheet_282.jpg", "paper/assets/pidqa_sheet_184.jpg",
    "output/pdf/manuscript.pdf", "output/pdf/supplementary.pdf",
    "reports/generated/editorial_revision_evidence_v4.json",
    "reports/generated/editorial_extension_experiments_v4.json",
    "reports/generated/editorial_extension_experiments_v4.csv",
    "reports/generated/editorial_revision_submission_v4.json",
    "reports/generated/editorial_revision_task_effects_v4.csv",
    "reports/generated/paddleocr_environment_v1.txt",
    "reports/generated/paddleocr_model_artifacts_v1.json",
    "reports/generated/internvl35_8b_editorial_checkpoint_v1.json",
    "reports/generated/reproduction_validation_v4.json",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/pdf_render_validation_v4.json",
    "reports/generated/pdf_visual_inspection_v4.json",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/internvl35_8b_value_correct.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/internvl35_8b_value_shuffled.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/internvl35_8b_value_text_only.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v1/internvl35_8b_value_correct.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v1/internvl35_8b_value_shuffled.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v1/internvl35_8b_value_text_only.jsonl",
    "outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl",
    "scripts/reproduce_submission_v4.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text))


def abstract_text(text: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S)
    if not match:
        raise ValueError("Missing abstract")
    return re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", match.group(1))


def latex_ids(text: str, command: str) -> set[str]:
    values: set[str] = set()
    for match in re.finditer(rf"\\{command}\{{([^}}]+)\}}", text):
        values.update(item.strip() for item in match.group(1).split(",") if item.strip())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/generated/submission_package_validation_v4.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    manuscript = (root / "paper/manuscript.tex").read_text(encoding="utf-8")
    supplement = (root / "paper/supplementary.tex").read_text(encoding="utf-8")
    cited = latex_ids(manuscript, "cite")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    labels = latex_ids(manuscript, "label") | latex_ids(supplement, "label")
    refs = latex_ids(manuscript, "ref") | latex_ids(supplement, "ref")
    figure_refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript + supplement)
    figures_missing = [name for name in figure_refs if not (root / "paper/figures" / name).is_file()]
    abstract_words = word_count(abstract_text(manuscript))
    highlights = [line[2:].strip() for line in (root / "paper/highlights.md").read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    reports = {
        "analysis": read_json(root / "reports/generated/editorial_revision_evidence_v4.json").get("status"),
        "extension": read_json(root / "reports/generated/editorial_extension_experiments_v4.json").get("status"),
        "submission_build": read_json(root / "reports/generated/editorial_revision_submission_v4.json").get("status"),
        "reproduction": read_json(root / "reports/generated/reproduction_validation_v4.json").get("status"),
        "figure_build": read_json(root / "paper/figures/figure_metadata_v4.json").get("status"),
        "answer_isolation": read_json(root / "reports/generated/evidence_input_answer_isolation_audit_v2.json").get("status"),
        "pdf_render": read_json(root / "reports/generated/pdf_render_validation_v4.json").get("status"),
        "pdf_visual": read_json(root / "reports/generated/pdf_visual_inspection_v4.json").get("status"),
    }
    render_report = read_json(root / "reports/generated/pdf_render_validation_v4.json")
    visual_report = read_json(root / "reports/generated/pdf_visual_inspection_v4.json")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    failures: list[str] = []
    if missing: failures.append("required_files_missing")
    if abstract_words > 250: failures.append("abstract_exceeds_250_words")
    if cited - bibliography: failures.append("citations_missing_from_bibliography")
    if bibliography - cited: failures.append("uncited_bibliography_entries")
    if refs - labels: failures.append("unresolved_latex_references")
    if figures_missing: failures.append("included_figures_missing")
    if not 3 <= len(highlights) <= 5 or any(len(line) > 85 for line in highlights): failures.append("highlights_outside_editorial_limits")
    if any(status != "pass" for status in reports.values()): failures.append("central_artifact_failed")
    rendered_hashes = {row["name"]: row.get("sha256") for row in render_report.get("documents", [])}
    if visual_report.get("pdf_sha256") != rendered_hashes:
        failures.append("visual_record_pdf_hash_mismatch")
    visual_pages = {row.get("path"): row.get("sha256") for row in visual_report.get("pages", [])}
    rendered_pages = {
        path
        for document in render_report.get("documents", [])
        for path in document.get("rendered_pngs", [])
    }
    if set(visual_pages) != rendered_pages:
        failures.append("visual_record_page_membership_mismatch")
    elif any(
        not (root / path).is_file()
        or visual_pages[path] != hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in rendered_pages
    ):
        failures.append("visual_record_page_hash_mismatch")
    if "pidqa-evidence-submission-v4" not in citation or "example.invalid" in citation: failures.append("citation_metadata_not_v4")
    required_phrases = (
        "not observed VLM", "No mapping-specific advantage was detected",
        "not a universally", "single public synthetic", "not an equivalence test",
        "candidate tag extraction", "not justify autonomous design review",
    )
    missing_phrases = [value for value in required_phrases if value not in manuscript]
    if missing_phrases: failures.append("claim_boundary_language_missing")
    forbidden_title = "Ontology-Aware"
    if forbidden_title in manuscript.split("\\begin{abstract}", 1)[0]: failures.append("ontology_claim_remains_in_title")
    if "@@" in manuscript or "@@" in supplement: failures.append("template_marker_unresolved")
    report = {
        "validation_version": "submission-package-v4",
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "required_files": {"count": len(REQUIRED_FILES), "missing": missing},
        "abstract_words": abstract_words,
        "manuscript_words": word_count(manuscript),
        "supplement_words": word_count(supplement),
        "citations": {"cited": len(cited), "bibliography": len(bibliography), "missing": sorted(cited-bibliography), "uncited": sorted(bibliography-cited)},
        "references_missing": sorted(refs-labels),
        "figures_missing": figures_missing,
        "highlights": {"count": len(highlights), "lengths": [len(line) for line in highlights]},
        "central_status": reports,
        "visual_record_matches_current_pdf": visual_report.get("pdf_sha256") == rendered_hashes,
        "missing_claim_boundary_phrases": missing_phrases,
        "submitter_owned_placeholders": ["authors/affiliations", "funding/competing interests/CRediT", "archive DOI or URL"],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
