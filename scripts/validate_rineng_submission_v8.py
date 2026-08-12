"""Validate the final Results in Engineering V8 submission artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from build_rineng_v8_tables import (
    build_external_table,
    build_internvl_table,
    build_quality_subset_table,
    build_quality_table,
)
from validate_rineng_submission_v6 import expand_tex_inputs, latex_ids, words


TITLE = (
    "Qualifying Image-Grounded Tag Retrieval in Piping and Instrumentation "
    "Diagrams with Source-Isolated Counterfactual Evaluation"
)

REQUIRED_FILES = (
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/figures/figure_4_cost_sensitive_operating_modes_v8.pdf",
    "paper/figures/figure_5_quality_and_budget_matched_v8.pdf",
    "paper/figures/figure_6_dexpi_external_v8.pdf",
    "paper/tables/table_rineng_v8_quality.tex",
    "paper/tables/table_rineng_v8_quality_by_subset.tex",
    "paper/tables/table_rineng_v8_internvl_budget54.tex",
    "paper/tables/table_rineng_v8_dexpi_external.tex",
    "reports/generated/rineng_cost_sensitive_operating_modes_v8.json",
    "reports/generated/rineng_v8_extension_score.json",
    "reports/generated/rineng_v8_dexpi_external_score.json",
    "reports/generated/rineng_v8_independent_validation.json",
    "reports/generated/rineng_v8_paper_summary.json",
    "paper/figures/figure_metadata_v8.json",
    "reports/generated/pdf_render_validation_v8.json",
    "reports/generated/pdf_visual_inspection_v8.json",
    "output/pdf/v8/manuscript.pdf",
    "output/pdf/v8/supplementary.pdf",
)

REQUIRED_FIGURES = (
    "figure_4_cost_sensitive_operating_modes_v8.pdf",
    "figure_5_quality_and_budget_matched_v8.pdf",
    "figure_6_dexpi_external_v8.pdf",
)

REQUIRED_TABLE_INPUTS = (
    "tables/table_rineng_v8_quality.tex",
    "tables/table_rineng_v8_quality_by_subset.tex",
    "tables/table_rineng_v8_internvl_budget54.tex",
    "tables/table_rineng_v8_dexpi_external.tex",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="reports/generated/rineng_submission_validation_v8.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures: list[str] = []
    missing = [item for item in REQUIRED_FILES if not (root / item).is_file()]
    if missing:
        failures.append("required_files_missing")

    manuscript_path = root / "paper/manuscript.tex"
    supplement_path = root / "paper/supplementary.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8") if manuscript_path.is_file() else ""
    supplement = supplement_path.read_text(encoding="utf-8") if supplement_path.is_file() else ""
    expanded = expand_tex_inputs(root, manuscript) + "\n" + expand_tex_inputs(root, supplement)

    title_match = re.search(r"\\title\{([^}]*)\}", manuscript)
    title = title_match.group(1).replace(r"\&", "&") if title_match else ""
    if title != TITLE:
        failures.append("title_mismatch")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", manuscript, flags=re.S
    )
    abstract_words = words(abstract_match.group(1)) if abstract_match else 0
    if not 150 <= abstract_words <= 250:
        failures.append("abstract_word_limit")

    keywords_match = re.search(
        r"\\textbf\{Keywords:\}\s*(.*?)\n\n", manuscript, flags=re.S
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
    if not 3 <= len(highlights) <= 5 or any(len(line) > 85 for line in highlights):
        failures.append("highlights_outside_editorial_limits")

    cited = latex_ids(manuscript, "cite")
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    labels = latex_ids(expanded, "label")
    refs = latex_ids(expanded, "ref")
    if cited != bibliography:
        failures.append("citation_bibliography_mismatch")
    if refs - labels:
        failures.append("undefined_latex_reference")

    included_figures = set(
        re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript + supplement)
    )
    if set(REQUIRED_FIGURES) - included_figures:
        failures.append("v8_figure_not_included")
    inputs = set(re.findall(r"\\input\{([^}]+)\}", manuscript + supplement))
    if set(REQUIRED_TABLE_INPUTS) - inputs:
        failures.append("v8_table_not_included")

    derived_status: dict[str, str | None] = {}
    for relative in (
        "reports/generated/rineng_cost_sensitive_operating_modes_v8.json",
        "reports/generated/rineng_v8_extension_score.json",
        "reports/generated/rineng_v8_dexpi_external_score.json",
        "reports/generated/rineng_v8_independent_validation.json",
        "reports/generated/rineng_v8_paper_summary.json",
        "paper/figures/figure_metadata_v8.json",
    ):
        path = root / relative
        status = read_json(path).get("status") if path.is_file() else None
        derived_status[relative] = status
        if status != "pass":
            failures.append("derived_artifact_not_pass")

    extension_path = root / "reports/generated/rineng_v8_extension_score.json"
    external_path = root / "reports/generated/rineng_v8_dexpi_external_score.json"
    if (
        extension_path.is_file()
        and external_path.is_file()
        and read_json(extension_path).get("status") == "pass"
        and read_json(external_path).get("status") == "pass"
    ):
        extension = read_json(extension_path)
        external = read_json(external_path)
        expected_tables = {
            "paper/tables/table_rineng_v8_quality.tex": build_quality_table(extension)[0],
            "paper/tables/table_rineng_v8_quality_by_subset.tex": build_quality_subset_table(extension)[0],
            "paper/tables/table_rineng_v8_internvl_budget54.tex": build_internvl_table(extension)[0],
            "paper/tables/table_rineng_v8_dexpi_external.tex": build_external_table(external)[0],
        }
        for relative, expected_text in expected_tables.items():
            path = root / relative
            if not path.is_file() or path.read_text(encoding="utf-8") != expected_text:
                failures.append("paper_table_differs_from_score_report")

        headline_tokens = {
            f"{external['metrics']['correct']['f1']:.4f}",
            f"{external['metrics']['paddleocr_full_image']['f1']:.4f}",
        }
        for row in extension.get("quality_comparisons", []):
            if (
                row.get("dataset") == "pooled_three_source_disjoint_subsets"
                and row.get("contrast") == "correct_minus_shuffled"
            ):
                headline_tokens.add(f"{float(row['value_f1_difference']):.4f}")
        if any(token not in expanded for token in headline_tokens):
            failures.append("headline_metric_missing_from_paper")

    pdf_hashes: dict[str, str | None] = {}
    render_path = root / "reports/generated/pdf_render_validation_v8.json"
    visual_path = root / "reports/generated/pdf_visual_inspection_v8.json"
    render = read_json(render_path) if render_path.is_file() else {}
    visual = read_json(visual_path) if visual_path.is_file() else {}
    if render.get("status") != "pass" or visual.get("status") != "pass":
        failures.append("pdf_validation_not_pass")
    render_documents = {
        str(item.get("name")): item for item in render.get("documents", [])
    }
    visual_hashes = visual.get("pdf_sha256", {})
    for name in ("manuscript", "supplementary"):
        path = root / f"output/pdf/v8/{name}.pdf"
        observed = sha256(path) if path.is_file() else None
        pdf_hashes[name] = observed
        if (
            observed is None
            or render_documents.get(name, {}).get("sha256") != observed
            or visual_hashes.get(name) != observed
        ):
            failures.append("pdf_hash_validation_mismatch")
    visual_pages = visual.get("pages", [])
    render_page_count = sum(
        int(item.get("page_count", 0)) for item in render.get("documents", [])
    )
    if (
        len(visual_pages) != render_page_count
        or any(item.get("inspected") is not True for item in visual_pages)
    ):
        failures.append("pdf_page_inspection_incomplete")

    # Deduplicate while retaining a stable diagnostic order.
    failures = list(dict.fromkeys(failures))
    administrative_placeholders = [
        value
        for value, present in (
            ("public archive DOI/URL", "[SUBMITTER: ARCHIVE DOI/URL]" in manuscript),
            ("author identities and affiliations", "[SUBMITTER:" in (root / "paper/title_page.md").read_text(encoding="utf-8") if (root / "paper/title_page.md").is_file() else False),
        )
        if present
    ]
    report = {
        "version": "rineng-submission-validation-v8",
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
        "undefined_references": sorted(refs - labels),
        "derived_status": derived_status,
        "pdf_sha256": pdf_hashes,
        "rendered_page_count": render_page_count,
        "inspected_page_count": len(visual_pages),
        "administrative_placeholders": administrative_placeholders,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
