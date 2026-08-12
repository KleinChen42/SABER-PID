"""Build the Results in Engineering author-side model-review package.

The package is a hash-verified pre-submission critique bundle. It is not an
official journal review, a human peer-review record, or evidence of editorial
endorsement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


PACKAGE_VERSION = "rineng-model-review-v2"
MANUSCRIPT_VERSION = "v5"
TARGET_JOURNAL = "Results in Engineering"
EXPECTED_TITLE = (
    "Image-Grounded Tag Reading in Piping and Instrumentation Diagrams: "
    "Source-Isolated Counterfactual Evaluation"
)
PACKAGE_DIR = "release/rineng_model_review_package_v2"
ARCHIVE_PATH = "release/rineng_model_review_package_v2.zip"
VALIDATION_PATH = "reports/generated/rineng_model_review_package_v2_validation.json"

MAPPINGS = (
    # Review instructions and prompts.
    ("review/rineng_model_review_v2/00_README_FIRST.md", "00_README_FIRST.md"),
    (
        "review/rineng_model_review_v2/01_RINENG_EXHAUSTIVE_REVIEW_PROMPT_EN.md",
        "prompts/01_RINENG_EXHAUSTIVE_REVIEW_PROMPT_EN.md",
    ),
    (
        "review/rineng_model_review_v2/02_REVIEW_OUTPUT_TEMPLATE.md",
        "prompts/02_REVIEW_OUTPUT_TEMPLATE.md",
    ),
    (
        "review/rineng_model_review_v2/03_PACKAGE_INDEX_AND_EVIDENCE_GUIDE.md",
        "evidence/03_PACKAGE_INDEX_AND_EVIDENCE_GUIDE.md",
    ),
    (
        "review/rineng_model_review_v2/04_RINENG_REQUIREMENTS_AND_FIT_BRIEF.md",
        "journal/04_RINENG_REQUIREMENTS_AND_FIT_BRIEF.md",
    ),
    (
        "review/rineng_model_review_v2/05_REVISION_FOLLOWUP_PROMPT_EN.md",
        "prompts/05_REVISION_FOLLOWUP_PROMPT_EN.md",
    ),
    # Frozen manuscript and supplement.
    ("output/pdf/v5/manuscript.pdf", "core/manuscript.pdf"),
    ("output/pdf/v5/supplementary.pdf", "core/supplementary.pdf"),
    ("paper/manuscript.tex", "core/manuscript.tex"),
    ("paper/supplementary.tex", "core/supplementary.tex"),
    # Main and supplementary artwork in vector and high-resolution raster form.
    (
        "paper/figures/figure_1_image_grounded_tag_reading_v5.pdf",
        "figures/figure_1_image_grounded_tag_reading_v5.pdf",
    ),
    (
        "paper/figures/figure_1_image_grounded_tag_reading_v5.png",
        "figures/figure_1_image_grounded_tag_reading_v5.png",
    ),
    (
        "paper/figures/figure_2_tag_reading_robustness_v5.pdf",
        "figures/figure_2_tag_reading_robustness_v5.pdf",
    ),
    (
        "paper/figures/figure_2_tag_reading_robustness_v5.png",
        "figures/figure_2_tag_reading_robustness_v5.png",
    ),
    (
        "paper/figures/figure_3_hybrid_tag_operating_envelope_v5.pdf",
        "figures/figure_3_hybrid_tag_operating_envelope_v5.pdf",
    ),
    (
        "paper/figures/figure_3_hybrid_tag_operating_envelope_v5.png",
        "figures/figure_3_hybrid_tag_operating_envelope_v5.png",
    ),
    (
        "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
        "figures/figure_s1_controls_and_operating_quantities_v4.pdf",
    ),
    (
        "paper/figures/figure_s1_controls_and_operating_quantities_v4.png",
        "figures/figure_s1_controls_and_operating_quantities_v4.png",
    ),
    (
        "paper/figures/figure_s2_tag_reading_stability_v4.pdf",
        "figures/figure_s2_tag_reading_stability_v4.pdf",
    ),
    (
        "paper/figures/figure_s2_tag_reading_stability_v4.png",
        "figures/figure_s2_tag_reading_stability_v4.png",
    ),
    ("paper/figures/figure_metadata_v5.json", "figures/figure_metadata_v5.json"),
    ("paper/figure_manifest.md", "figures/FIGURE_MANIFEST.md"),
    # Submission-facing files.
    ("paper/figure_captions.md", "submission/FIGURE_CAPTIONS.md"),
    ("paper/title_page.md", "submission/TITLE_PAGE.md"),
    ("paper/highlights.md", "submission/HIGHLIGHTS.md"),
    ("paper/cover_letter.md", "submission/COVER_LETTER.md"),
    ("paper/data_availability.md", "submission/DATA_AVAILABILITY.md"),
    ("paper/declarations.md", "submission/DECLARATIONS.md"),
    ("CITATION.cff", "submission/CITATION.cff"),
    ("LICENSES.md", "submission/LICENSES.md"),
    # Author framing is intentionally separated from the independent first pass.
    (
        "reports/POSITIVE_NARRATIVE_SELF_REVIEW_AND_JOURNAL_STRATEGY_V5.md",
        "author_brief/POSITIVE_NARRATIVE_SELF_REVIEW_AND_JOURNAL_STRATEGY_V5.md",
    ),
    (
        "reports/POSITIVE_NARRATIVE_REVISION_CLOSEOUT_V5.md",
        "author_brief/POSITIVE_NARRATIVE_REVISION_CLOSEOUT_V5.md",
    ),
    # Frozen machine-readable evidence.
    (
        "reports/generated/positive_narrative_hybrid_analysis_v5.json",
        "evidence/positive_narrative_hybrid_analysis_v5.json",
    ),
    (
        "reports/generated/positive_narrative_hybrid_analysis_v5.csv",
        "evidence/positive_narrative_hybrid_analysis_v5.csv",
    ),
    (
        "reports/generated/positive_narrative_fusion_validation_v5.json",
        "evidence/positive_narrative_fusion_validation_v5.json",
    ),
    (
        "reports/generated/qwen8_value_budget_sensitivity_v1.json",
        "evidence/qwen8_value_budget_sensitivity_v1.json",
    ),
    (
        "reports/generated/qwen8_value_budget_sensitivity_v1.csv",
        "evidence/qwen8_value_budget_sensitivity_v1.csv",
    ),
    (
        "reports/generated/image_dependence_control_v1.json",
        "evidence/image_dependence_control_v1.json",
    ),
    (
        "reports/generated/image_dependence_control_v1.csv",
        "evidence/image_dependence_control_v1.csv",
    ),
    (
        "reports/generated/text_only_image_grounding_control_v1.json",
        "evidence/text_only_image_grounding_control_v1.json",
    ),
    (
        "reports/generated/text_only_image_grounding_control_v1.csv",
        "evidence/text_only_image_grounding_control_v1.csv",
    ),
    (
        "reports/generated/source_seed_resolution_sensitivity_v1.json",
        "evidence/source_seed_resolution_sensitivity_v1.json",
    ),
    (
        "reports/generated/source_seed_resolution_sensitivity_v1.csv",
        "evidence/source_seed_resolution_sensitivity_v1.csv",
    ),
    (
        "reports/generated/ontology_mapping_control_v1.json",
        "evidence/ontology_mapping_control_v1.json",
    ),
    (
        "reports/generated/ontology_mapping_control_v1.csv",
        "evidence/ontology_mapping_control_v1.csv",
    ),
    (
        "reports/generated/editorial_extension_experiments_v4.json",
        "evidence/editorial_extension_experiments_v4.json",
    ),
    (
        "reports/generated/editorial_extension_experiments_v4.csv",
        "evidence/editorial_extension_experiments_v4.csv",
    ),
    (
        "reports/generated/editorial_revision_evidence_v4.json",
        "evidence/editorial_revision_evidence_v4.json",
    ),
    (
        "reports/generated/editorial_revision_task_effects_v4.csv",
        "evidence/editorial_revision_task_effects_v4.csv",
    ),
    (
        "reports/generated/evidence_input_answer_isolation_audit_v2.json",
        "evidence/evidence_input_answer_isolation_audit_v2.json",
    ),
    (
        "reports/generated/positive_narrative_submission_v5.json",
        "evidence/positive_narrative_submission_v5.json",
    ),
    ("reports/F4_EXTERNAL_SOURCE_STATUS_V1.md", "evidence/EXTERNAL_DATA_BOUNDARY.md"),
    # Independent technical validations.
    (
        "reports/generated/positive_narrative_submission_validation_v5.json",
        "validation/positive_narrative_submission_validation_v5.json",
    ),
    (
        "reports/generated/reproduction_validation_v5.json",
        "validation/reproduction_validation_v5.json",
    ),
    (
        "reports/generated/pdf_render_validation_v5.json",
        "validation/pdf_render_validation_v5.json",
    ),
    (
        "reports/generated/pdf_visual_inspection_v5.json",
        "validation/pdf_visual_inspection_v5.json",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_path", "package_path", "size_bytes", "sha256"),
        )
        writer.writeheader()
        writer.writerows(rows)


def deterministic_zip(archive: Path, package_dir: Path) -> None:
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        files = sorted(
            (candidate for candidate in package_dir.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(package_dir).as_posix(),
        )
        for source in files:
            name = source.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(filename=name, date_time=(2026, 8, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(
                info,
                source.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def validate_inputs(root: Path) -> dict[str, Any]:
    missing = [source for source, _ in MAPPINGS if not (root / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing review-package inputs: " + ", ".join(missing))

    duplicate_destinations = sorted(
        destination
        for destination in {destination for _, destination in MAPPINGS}
        if sum(item_destination == destination for _, item_destination in MAPPINGS) > 1
    )
    if duplicate_destinations:
        raise ValueError("Duplicate package destinations: " + ", ".join(duplicate_destinations))

    manuscript = (root / "paper/manuscript.tex").read_text(encoding="utf-8")
    prompt = (
        root / "review/rineng_model_review_v2/01_RINENG_EXHAUSTIVE_REVIEW_PROMPT_EN.md"
    ).read_text(encoding="utf-8")
    if EXPECTED_TITLE not in manuscript:
        raise ValueError("The frozen v5 title is absent from paper/manuscript.tex")
    if EXPECTED_TITLE not in prompt:
        raise ValueError("The review prompt does not identify the frozen v5 title")

    submission_validation = read_json(
        root / "reports/generated/positive_narrative_submission_validation_v5.json"
    )
    reproduction_validation = read_json(
        root / "reports/generated/reproduction_validation_v5.json"
    )
    pdf_validation = read_json(root / "reports/generated/pdf_render_validation_v5.json")
    status_by_report = {
        "positive_narrative_submission_validation_v5": submission_validation.get("status"),
        "reproduction_validation_v5": reproduction_validation.get("status"),
        "pdf_render_validation_v5": pdf_validation.get("status"),
    }
    failed = [name for name, status in status_by_report.items() if status != "pass"]
    if failed:
        raise ValueError("Non-passing frozen validation reports: " + ", ".join(failed))
    if submission_validation.get("title") != EXPECTED_TITLE:
        raise ValueError("Submission validator title does not match the expected v5 title")
    if submission_validation.get("abstract_words", 10**9) > 250:
        raise ValueError("Validated abstract exceeds 250 words")
    highlights = submission_validation.get("highlights", {})
    if highlights.get("count") not in {3, 4, 5}:
        raise ValueError("Validated highlights count is outside 3--5")
    if any(length > 85 for length in highlights.get("lengths", [])):
        raise ValueError("A validated highlight exceeds 85 characters")

    return {
        "status_by_report": status_by_report,
        "abstract_words": submission_validation.get("abstract_words"),
        "highlight_count": highlights.get("count"),
        "keyword_count": submission_validation.get("keywords", {}).get("count"),
        "manuscript_words": submission_validation.get("manuscript_words"),
        "supplement_words": submission_validation.get("supplement_words"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package-dir", default=PACKAGE_DIR)
    parser.add_argument("--archive", default=ARCHIVE_PATH)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    package_dir = root / args.package_dir
    archive = root / args.archive

    input_validation = validate_inputs(root)
    if package_dir.exists() and any(package_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package directory: {package_dir}")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {archive}")
    package_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source_relative, package_relative in MAPPINGS:
        source = root / source_relative
        destination = package_dir / package_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Hash mismatch after copy: {source_relative}")
        rows.append(
            {
                "source_path": source_relative,
                "package_path": package_relative,
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
            }
        )

    package_manifest = {
        "package_version": PACKAGE_VERSION,
        "package_date": "2026-08-11",
        "target_journal": TARGET_JOURNAL,
        "manuscript_version": MANUSCRIPT_VERSION,
        "manuscript_title": EXPECTED_TITLE,
        "review_type": "author-side model-based pre-submission critique",
        "official_journal_review": False,
        "human_review_evidence": False,
        "editorial_endorsement": False,
        "generated_ai_artwork": False,
        "status": "pass",
        "input_validation": input_validation,
        "artifact_count": len(rows),
        "artifacts": rows,
        "scientific_integrity_boundary": [
            "Do not present model output as human or official journal peer review.",
            "Do not hide material counterevidence or fabricate external results.",
            "Do not infer human annotation, expert judgment, or reviewer identity.",
            "Treat same-family source partitions as within-family checks, not external replication.",
            "Keep post-hoc rule development separate from frozen-rule validation.",
        ],
    }
    manifest_json = package_dir / "PACKAGE_MANIFEST.json"
    manifest_csv = package_dir / "PACKAGE_MANIFEST.csv"
    manifest_json.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_manifest_csv(manifest_csv, rows)
    deterministic_zip(archive, package_dir)

    expected = {row["package_path"]: row["sha256"] for row in rows}
    missing_members: list[str] = []
    mismatches: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as handle:
        members = set(handle.namelist())
        for name, expected_hash in expected.items():
            if name not in members:
                missing_members.append(name)
                continue
            with handle.open(name) as member:
                observed = digest_stream(member)
            if observed != expected_hash:
                mismatches.append(
                    {"path": name, "expected": expected_hash, "observed": observed}
                )
        unexpected = sorted(
            members - set(expected) - {"PACKAGE_MANIFEST.json", "PACKAGE_MANIFEST.csv"}
        )
        bad_zip_member = handle.testzip()

    validation_status = (
        "pass"
        if not missing_members and not mismatches and not unexpected and bad_zip_member is None
        else "fail"
    )
    validation = {
        "package_version": PACKAGE_VERSION,
        "target_journal": TARGET_JOURNAL,
        "manuscript_version": MANUSCRIPT_VERSION,
        "status": validation_status,
        "archive": archive.relative_to(root).as_posix(),
        "archive_sha256": sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
        "artifact_count": len(rows),
        "zip_member_count": len(members),
        "missing_members": missing_members,
        "content_hash_mismatches": mismatches,
        "unexpected_members": unexpected,
        "bad_zip_member": bad_zip_member,
        "package_manifest_sha256": sha256(manifest_json),
        "official_journal_review": False,
        "human_review_evidence": False,
        "input_validation": input_validation,
    }
    validation_path = root / VALIDATION_PATH
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": validation_status,
                "archive": str(archive),
                "archive_sha256": validation["archive_sha256"],
                "archive_size_bytes": validation["archive_size_bytes"],
                "artifact_count": len(rows),
                "zip_member_count": len(members),
            },
            sort_keys=True,
        )
    )
    return 0 if validation_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
