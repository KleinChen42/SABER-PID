"""Validate the positive-narrative manuscript, evidence, and final PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


TITLE = "Image-Grounded Tag Reading in Piping and Instrumentation Diagrams: Source-Isolated Counterfactual Evaluation"
REQUIRED_FILES = (
    "CITATION.cff",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/title_page.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_1_image_grounded_tag_reading_v5.pdf",
    "paper/figures/figure_1_image_grounded_tag_reading_v5.png",
    "paper/figures/figure_2_tag_reading_robustness_v5.pdf",
    "paper/figures/figure_2_tag_reading_robustness_v5.png",
    "paper/figures/figure_3_hybrid_tag_operating_envelope_v5.pdf",
    "paper/figures/figure_3_hybrid_tag_operating_envelope_v5.png",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
    "paper/figures/figure_s2_tag_reading_stability_v4.pdf",
    "paper/figures/figure_metadata_v5.json",
    "reports/generated/positive_narrative_hybrid_analysis_v5.json",
    "reports/generated/positive_narrative_hybrid_analysis_v5.csv",
    "reports/generated/positive_narrative_fusion_validation_v5.json",
    "outputs/positive_narrative/paddleocr_seed29_v1.jsonl",
    "outputs/positive_narrative/paddleocr_seed31_v1.jsonl",
    "reports/generated/positive_narrative_submission_v5.json",
    "reports/generated/reproduction_validation_v5.json",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/pdf_render_validation_v5.json",
    "reports/generated/pdf_visual_inspection_v5.json",
    "reports/POSITIVE_NARRATIVE_REVISION_CLOSEOUT_V5.md",
    "reports/POSITIVE_NARRATIVE_SELF_REVIEW_AND_JOURNAL_STRATEGY_V5.md",
    "output/pdf/v5/manuscript.pdf",
    "output/pdf/v5/supplementary.pdf",
    "scripts/build_positive_narrative_hybrid_analysis_v5.py",
    "scripts/score_positive_narrative_fusion_validation_v5.py",
    "scripts/build_positive_narrative_figures_v5.py",
    "scripts/reproduce_submission_v5.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text))


def latex_ids(text: str, command: str) -> set[str]:
    result: set[str] = set()
    for match in re.finditer(rf"\\{command}\{{([^}}]+)\}}", text):
        result.update(item.strip() for item in match.group(1).split(",") if item.strip())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/generated/positive_narrative_submission_validation_v5.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    failures: list[str] = []
    if missing:
        failures.append("required_files_missing")

    manuscript = (root / "paper/manuscript.tex").read_text(encoding="utf-8")
    supplement = (root / "paper/supplementary.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    title_match = re.search(r"\\title\{([^}]*)\}", manuscript)
    observed_title = title_match.group(1).replace(r"\&", "&") if title_match else ""
    if observed_title != TITLE:
        failures.append("title_mismatch")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", manuscript, flags=re.S)
    abstract_words = words(abstract_match.group(1)) if abstract_match else 0
    if not 150 <= abstract_words <= 250:
        failures.append("abstract_word_limit")
    keywords_match = re.search(r"\\textbf\{Keywords:\}\s*(.*?)\n\n", manuscript, flags=re.S)
    keywords = [item.strip() for item in keywords_match.group(1).replace("\n", " ").split(";")] if keywords_match else []
    if not 1 <= len(keywords) <= 7:
        failures.append("keyword_count")

    cited = latex_ids(manuscript, "cite")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    labels = latex_ids(combined, "label")
    refs = latex_ids(combined, "ref")
    if cited != bibliography:
        failures.append("citation_bibliography_mismatch")
    if refs - labels:
        failures.append("undefined_latex_reference")
    figure_refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", combined)
    missing_figures = [name for name in figure_refs if not (root / "paper/figures" / name).is_file()]
    if missing_figures:
        failures.append("included_figure_missing")
    artwork = []
    for relative in (
        "paper/figures/figure_1_image_grounded_tag_reading_v5.png",
        "paper/figures/figure_2_tag_reading_robustness_v5.png",
        "paper/figures/figure_3_hybrid_tag_operating_envelope_v5.png",
    ):
        with Image.open(root / relative) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
            artwork.append({"path": relative, "width": image.width, "height": image.height, "dpi": dpi})
    if any(min(row["dpi"]) < 499.0 for row in artwork):
        failures.append("artwork_raster_fallback_below_500dpi")

    highlights = [
        line[2:].strip()
        for line in (root / "paper/highlights.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]
    if not 3 <= len(highlights) <= 5 or any(len(line) > 85 for line in highlights):
        failures.append("highlights_outside_editorial_limits")

    required_main = (
        "image-grounded candidate-tag retrieval",
        "reference-free descriptive",
        "single synthetic source family",
        "small detected direction rather than a practical replication",
        "post-hoc",
        "do not establish general topology",
    )
    required_supplement = (
        "Correct labels minus permuted labels",
        "InternVL wrong image",
        "InternVL no image",
        "Set intersection",
        "Qwen if non-empty, otherwise OCR",
        "training-only task-majority prior scores",
    )
    normalized_manuscript = re.sub(r"\s+", " ", manuscript)
    normalized_supplement = re.sub(r"\s+", " ", supplement)
    missing_main_phrases = [phrase for phrase in required_main if phrase not in normalized_manuscript]
    missing_supplement_phrases = [phrase for phrase in required_supplement if phrase not in normalized_supplement]
    if missing_main_phrases or missing_supplement_phrases:
        failures.append("material_boundary_or_rule_missing")
    if "@@" in combined:
        failures.append("unresolved_template_marker")

    reports = {
        "editorial_analysis": read_json(root / "reports/generated/editorial_revision_evidence_v4.json").get("status"),
        "extension": read_json(root / "reports/generated/editorial_extension_experiments_v4.json").get("status"),
        "hybrid": read_json(root / "reports/generated/positive_narrative_hybrid_analysis_v5.json").get("status"),
        "fusion_validation": read_json(root / "reports/generated/positive_narrative_fusion_validation_v5.json").get("status"),
        "figure_v5": read_json(root / "paper/figures/figure_metadata_v5.json").get("status"),
        "submission_v5": read_json(root / "reports/generated/positive_narrative_submission_v5.json").get("status"),
        "reproduction_v5": read_json(root / "reports/generated/reproduction_validation_v5.json").get("status"),
        "answer_isolation": read_json(root / "reports/generated/evidence_input_answer_isolation_audit_v2.json").get("status"),
        "pdf_render": read_json(root / "reports/generated/pdf_render_validation_v5.json").get("status"),
        "pdf_visual": read_json(root / "reports/generated/pdf_visual_inspection_v5.json").get("status"),
    }
    if any(status != "pass" for status in reports.values()):
        failures.append("central_artifact_failed")

    fusion_validation = read_json(root / "reports/generated/positive_narrative_fusion_validation_v5.json")
    if (
        fusion_validation.get("reference_used_in_prediction_construction") is not False
        or fusion_validation.get("all_rules_reported") is not True
        or [row.get("seed") for row in fusion_validation.get("seeds", [])] != [29, 31]
        or any(row.get("status") != "pass" for row in fusion_validation.get("seeds", []))
        or fusion_validation.get("source_partition_overlap")
        != {"set_b_seed29": 17, "set_b_seed31": 17, "seed29_seed31": 20, "three_way": 2}
    ):
        failures.append("fusion_validation_contract_failed")

    render = read_json(root / "reports/generated/pdf_render_validation_v5.json")
    visual = read_json(root / "reports/generated/pdf_visual_inspection_v5.json")
    rendered_hashes = {row["name"]: row.get("sha256") for row in render.get("documents", [])}
    if visual.get("pdf_sha256") != rendered_hashes:
        failures.append("visual_pdf_hash_mismatch")
    visual_pages = {row.get("path"): row.get("sha256") for row in visual.get("pages", [])}
    rendered_pages = {path for document in render.get("documents", []) for path in document.get("rendered_pngs", [])}
    if set(visual_pages) != rendered_pages:
        failures.append("visual_page_membership_mismatch")
    elif any(
        not (root / path).is_file()
        or visual_pages[path] != hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in rendered_pages
    ):
        failures.append("visual_page_hash_mismatch")
    if any(
        row.get("overfull_hbox_count")
        or row.get("underfull_hbox_count")
        or row.get("undefined_citation_count")
        or row.get("undefined_reference_count")
        for row in render.get("documents", [])
    ):
        failures.append("pdf_log_warning")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if "pidqa-positive-narrative-submission-v5" not in citation or TITLE not in citation:
        failures.append("citation_metadata_mismatch")

    report = {
        "validation_version": "positive-narrative-submission-v5",
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "required_files": {"count": len(REQUIRED_FILES), "missing": missing},
        "title": observed_title,
        "abstract_words": abstract_words,
        "keywords": {"count": len(keywords), "values": keywords},
        "manuscript_words": words(manuscript),
        "supplement_words": words(supplement),
        "highlights": {"count": len(highlights), "lengths": [len(line) for line in highlights]},
        "citations": {"cited": len(cited), "bibliography": len(bibliography), "missing": sorted(cited - bibliography), "uncited": sorted(bibliography - cited)},
        "undefined_references": sorted(refs - labels),
        "missing_figures": missing_figures,
        "artwork": artwork,
        "missing_main_phrases": missing_main_phrases,
        "missing_supplement_phrases": missing_supplement_phrases,
        "central_status": reports,
        "visual_record_matches_current_pdf": visual.get("pdf_sha256") == rendered_hashes,
        "submitter_owned_placeholders": ["authors/affiliations", "funding/competing interests/CRediT", "archive DOI or URL"],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
