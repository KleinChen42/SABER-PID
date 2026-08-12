"""Validate the Results in Engineering v6 manuscript, evidence, and PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


TITLE = (
    "Qualifying Image-Grounded Tag Retrieval in Piping and Instrumentation "
    "Diagrams with Source-Isolated Counterfactual Evaluation"
)

REQUIRED_FILES = (
    "CITATION.cff",
    "requirements-analysis-v6.txt",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/title_page.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_1_qualification_decision_v6.pdf",
    "paper/figures/figure_1_qualification_decision_v6.png",
    "paper/figures/figure_2_qualification_effects_v6.pdf",
    "paper/figures/figure_2_qualification_effects_v6.png",
    "paper/figures/figure_3_operating_modes_v6.pdf",
    "paper/figures/figure_3_operating_modes_v6.png",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
    "paper/figures/figure_metadata_v6.json",
    "reports/generated/rineng_revision_analysis_v6.json",
    "reports/generated/rineng_revision_analysis_v6.csv",
    "reports/generated/rineng_revision_per_source_v6.csv",
    "reports/generated/rineng_revision_error_taxonomy_v6.csv",
    "reports/generated/rineng_revision_environment_v6.json",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/pdf_render_validation_v6.json",
    "reports/generated/pdf_visual_inspection_v6.json",
    "output/pdf/v6/manuscript.pdf",
    "output/pdf/v6/supplementary.pdf",
    "scripts/build_rineng_revision_analysis_v6.py",
    "scripts/build_rineng_revision_figures_v6.py",
    "scripts/validate_rineng_submission_v6.py",
    "tests/test_rineng_revision_analysis_v6.py",
)

ARTWORK = (
    "paper/figures/figure_1_qualification_decision_v6.png",
    "paper/figures/figure_2_qualification_effects_v6.png",
    "paper/figures/figure_3_operating_modes_v6.png",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 5e-5) -> bool:
    return abs(actual - expected) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/rineng_submission_validation_v6.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    failures: list[str] = []
    if missing:
        failures.append("required_files_missing")

    manuscript = (root / "paper/manuscript.tex").read_text(encoding="utf-8")
    supplement = (root / "paper/supplementary.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    normalized_main = re.sub(r"\s+", " ", manuscript)
    normalized_supp = re.sub(r"\s+", " ", supplement)

    title_match = re.search(r"\\title\{([^}]*)\}", manuscript)
    observed_title = title_match.group(1).replace(r"\&", "&") if title_match else ""
    if observed_title != TITLE:
        failures.append("title_mismatch")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        manuscript,
        flags=re.S,
    )
    abstract_words = words(abstract_match.group(1)) if abstract_match else 0
    if not 150 <= abstract_words <= 250:
        failures.append("abstract_word_limit")

    keywords_match = re.search(
        r"\\textbf\{Keywords:\}\s*(.*?)\n\n",
        manuscript,
        flags=re.S,
    )
    keywords = (
        [
            item.strip()
            for item in keywords_match.group(1).replace("\n", " ").split(";")
            if item.strip()
        ]
        if keywords_match
        else []
    )
    if len(keywords) != 6:
        failures.append("keyword_count_not_six")

    cited = latex_ids(manuscript, "cite")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    labels = latex_ids(combined, "label")
    refs = latex_ids(combined, "ref")
    if cited != bibliography:
        failures.append("citation_bibliography_mismatch")
    if refs - labels:
        failures.append("undefined_latex_reference")

    figure_refs = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
        combined,
    )
    missing_figures = [
        name for name in figure_refs if not (root / "paper/figures" / name).is_file()
    ]
    if missing_figures:
        failures.append("included_figure_missing")

    artwork: list[dict[str, Any]] = []
    for relative in ARTWORK:
        with Image.open(root / relative) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
            artwork.append(
                {
                    "path": relative,
                    "width": image.width,
                    "height": image.height,
                    "dpi": dpi,
                }
            )
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
        "qualification decision",
        "source-macro",
        "true positives",
        "median candidates per drawing",
        "Set-B-excluded",
        "not an independent-replication claim",
        "general P\\&ID understanding",
    )
    required_supplement = (
        "empty-reference",
        "Geometry join v1",
        "strictly disjoint",
        "form categories, not manual",
        "reproduce\\_submission\\_v6.py",
        "public release excludes author-side editorial prompts",
    )
    missing_main_phrases = [
        phrase for phrase in required_main if phrase not in normalized_main
    ]
    missing_supplement_phrases = [
        phrase for phrase in required_supplement if phrase not in normalized_supp
    ]
    if missing_main_phrases or missing_supplement_phrases:
        failures.append("material_method_or_boundary_missing")

    forbidden_phrases = (
        "Prospective within-family",
        "This confirms",
        "wrong image",
        "Image-Grounded Tag Reading in Piping and Instrumentation Diagrams:",
    )
    observed_forbidden = [
        phrase for phrase in forbidden_phrases if phrase.lower() in combined.lower()
    ]
    if observed_forbidden:
        failures.append("deprecated_or_overclaiming_terminology")
    if "@@" in combined:
        failures.append("unresolved_template_marker")

    analysis = read_json(root / "reports/generated/rineng_revision_analysis_v6.json")
    figure = read_json(root / "paper/figures/figure_metadata_v6.json")
    answer_isolation = read_json(
        root / "reports/generated/evidence_input_answer_isolation_audit_v2.json"
    )
    render = read_json(root / "reports/generated/pdf_render_validation_v6.json")
    visual = read_json(root / "reports/generated/pdf_visual_inspection_v6.json")
    environment = read_json(
        root / "reports/generated/rineng_revision_environment_v6.json"
    )
    statuses = {
        "analysis": analysis.get("status"),
        "figure": figure.get("status"),
        "answer_isolation": answer_isolation.get("status"),
        "render": render.get("status"),
        "visual": visual.get("status"),
        "environment": environment.get("status"),
    }
    if any(value != "pass" for value in statuses.values()):
        failures.append("central_artifact_failed")

    datasets = analysis.get("datasets", {})
    expected_sources = {
        "set_b": 100,
        "seed29_excluding_set_b": 83,
        "seed31_excluding_set_b": 83,
        "seed29_strictly_disjoint": 65,
        "seed31_strictly_disjoint": 65,
    }
    if {
        name: datasets.get(name, {}).get("source_count") for name in expected_sources
    } != expected_sources:
        failures.append("source_exclusion_membership_failed")

    set_b = datasets.get("set_b", {}).get("methods", {})
    qwen = set_b.get("qwen", {}).get("micro_pooled", {})
    union = set_b.get("set_union", {}).get("micro_pooled", {})
    intersection = set_b.get("set_intersection", {}).get("micro_pooled", {})
    if not (
        qwen.get("tp") == 192
        and qwen.get("fp") == 152
        and qwen.get("fn") == 156
        and close(float(qwen.get("f1", -1)), 0.5549132948)
        and close(float(union.get("f1", -1)), 0.6338939198)
        and close(float(intersection.get("precision", -1)), 0.9906542056)
    ):
        failures.append("headline_metric_contract_failed")

    recorded_sources = analysis.get("sources", [])
    if any(
        not (root / row.get("path", "")).is_file()
        or file_sha256(root / row["path"]) != row.get("sha256")
        for row in recorded_sources
    ):
        failures.append("analysis_input_hash_mismatch")

    rendered_hashes = {
        row["name"]: row.get("sha256") for row in render.get("documents", [])
    }
    if visual.get("pdf_sha256") != rendered_hashes:
        failures.append("visual_pdf_hash_mismatch")
    rendered_pages = {
        path
        for document in render.get("documents", [])
        for path in document.get("rendered_pngs", [])
    }
    visual_pages = {
        row.get("path"): row.get("sha256") for row in visual.get("pages", [])
    }
    if set(visual_pages) != rendered_pages:
        failures.append("visual_page_membership_mismatch")
    elif any(
        not (root / path).is_file()
        or file_sha256(root / path) != visual_pages[path]
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
    if "pidqa-rineng-qualification-submission-v6" not in citation or TITLE not in citation:
        failures.append("citation_metadata_mismatch")

    report = {
        "validation_version": "rineng-submission-v6",
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "required_files": {"count": len(REQUIRED_FILES), "missing": missing},
        "title": observed_title,
        "abstract_words": abstract_words,
        "keywords": {"count": len(keywords), "values": keywords},
        "manuscript_words": words(manuscript),
        "supplement_words": words(supplement),
        "highlights": {
            "count": len(highlights),
            "lengths": [len(line) for line in highlights],
        },
        "citations": {
            "cited": len(cited),
            "bibliography": len(bibliography),
            "missing": sorted(cited - bibliography),
            "uncited": sorted(bibliography - cited),
        },
        "undefined_references": sorted(refs - labels),
        "missing_figures": missing_figures,
        "artwork": artwork,
        "missing_main_phrases": missing_main_phrases,
        "missing_supplement_phrases": missing_supplement_phrases,
        "forbidden_phrases": observed_forbidden,
        "central_status": statuses,
        "visual_record_matches_current_pdf": visual.get("pdf_sha256")
        == rendered_hashes,
        "submitter_owned_placeholders": [
            "authors/affiliations/corresponding author/ORCID",
            "funding/competing interests/CRediT/originality",
            "public archive DOI or URL",
            "public code-license choice",
        ],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
