"""Build a compact, hash-verified GPT Pro model-review package.

The package supports model-based editorial critique.  It is not evidence of a
completed human peer review and contains no fabricated reviewer identities.
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


PACKAGE_VERSION = "gpt-pro-model-review-v1"
PACKAGE_DIR = "release/gpt_pro_model_review_package_v1"
ARCHIVE_PATH = "release/gpt_pro_model_review_package_v1.zip"
VALIDATION_PATH = "reports/generated/gpt_pro_model_review_package_v1_validation.json"

MAPPINGS = (
    ("review/gpt_pro_model_review_v1/00_README_FIRST.md", "00_README_FIRST.md"),
    ("review/gpt_pro_model_review_v1/01_GPT_PRO_MASTER_REVIEW_PROMPT.md", "prompts/01_GPT_PRO_MASTER_REVIEW_PROMPT.md"),
    ("review/gpt_pro_model_review_v1/02_REVIEW_EVIDENCE_DIGEST.md", "evidence/REVIEW_EVIDENCE_DIGEST.md"),
    ("review/gpt_pro_model_review_v1/03_EDITORIAL_STORY_BRIEF.md", "author_brief/03_EDITORIAL_STORY_BRIEF.md"),
    ("review/gpt_pro_model_review_v1/04_CANDIDATE_EXPERIMENTS_AND_FIGURES.md", "author_brief/04_CANDIDATE_EXPERIMENTS_AND_FIGURES.md"),
    ("review/gpt_pro_model_review_v1/05_REVIEW_OUTPUT_TEMPLATE.md", "prompts/05_REVIEW_OUTPUT_TEMPLATE.md"),
    ("review/gpt_pro_model_review_v1/06_GPT_PRO_REVISION_FOLLOWUP_PROMPT.md", "prompts/06_GPT_PRO_REVISION_FOLLOWUP_PROMPT.md"),
    ("output/pdf/manuscript.pdf", "core/manuscript.pdf"),
    ("output/pdf/supplementary.pdf", "core/supplementary.pdf"),
    ("paper/manuscript.tex", "core/manuscript.tex"),
    ("paper/supplementary.tex", "core/supplementary.tex"),
    ("paper/figures/figure_1_saber_pid_overview.png", "figures/figure_1_saber_pid_overview.png"),
    ("paper/figures/figure_2_core_effects.png", "figures/figure_2_core_effects.png"),
    ("paper/figures/figure_3_tag_reading_stability.png", "figures/figure_3_tag_reading_stability.png"),
    ("paper/figures/figure_s1_task_calibration_and_boundaries.png", "figures/figure_s1_task_calibration_and_boundaries.png"),
    ("paper/figure_manifest.md", "figures/FIGURE_MANIFEST.md"),
    ("paper/figure_captions.md", "submission/FIGURE_CAPTIONS.md"),
    ("paper/title_page.md", "submission/TITLE_PAGE.md"),
    ("paper/highlights.md", "submission/HIGHLIGHTS.md"),
    ("paper/cover_letter.md", "submission/COVER_LETTER.md"),
    ("paper/data_availability.md", "submission/DATA_AVAILABILITY.md"),
    ("CITATION.cff", "submission/CITATION.cff"),
    ("LICENSES.md", "submission/LICENSES.md"),
    ("reports/generated/final_claim_evidence_matrix_v3.csv", "evidence/FINAL_CLAIM_EVIDENCE_MATRIX.csv"),
    ("reports/generated/final_statistical_summary_v3.json", "evidence/FINAL_STATISTICAL_SUMMARY.json"),
    ("reports/generated/submission_package_validation_v3.json", "evidence/SUBMISSION_PACKAGE_VALIDATION.json"),
    ("reports/generated/pdf_render_validation_v3.json", "evidence/PDF_RENDER_VALIDATION.json"),
    ("reports/generated/ontology_mapping_control_v1.csv", "evidence/E7_ONTOLOGY_MAPPING_CONTROL.csv"),
    ("reports/generated/text_only_image_grounding_control_v1.csv", "evidence/E8_TEXT_ONLY_CONTROL.csv"),
    ("reports/E7_ONTOLOGY_MAPPING_CONTROL_CLOSEOUT.md", "evidence/E7_CLOSEOUT.md"),
    ("reports/E8_TEXT_ONLY_IMAGE_GROUNDING_CLOSEOUT.md", "evidence/E8_CLOSEOUT.md"),
    ("reports/INTERNVL_CORRECTED_REPLICATION_CLOSEOUT.md", "evidence/INTERNVL_BOUNDARY_CLOSEOUT.md"),
    ("reports/F4_EXTERNAL_SOURCE_STATUS_V1.md", "evidence/EXTERNAL_DATA_BOUNDARY.md"),
    ("reports/EXPERIMENT_DECISION_LEDGER.md", "evidence/EXPERIMENT_DECISION_LEDGER.md"),
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


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_path", "package_path", "size_bytes", "sha256"),
        )
        writer.writeheader()
        writer.writerows(rows)


def deterministic_zip(archive: Path, package_dir: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for source in sorted(
            (candidate for candidate in package_dir.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(package_dir).as_posix(),
        ):
            name = source.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(filename=name, date_time=(2026, 8, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package-dir", default=PACKAGE_DIR)
    parser.add_argument("--archive", default=ARCHIVE_PATH)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    package_dir = root / args.package_dir
    archive = root / args.archive
    missing = [source for source, _ in MAPPINGS if not (root / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing review-package inputs: " + ", ".join(missing))
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
        "review_type": "model-based GPT Pro editorial and peer-review simulation",
        "human_review_evidence": False,
        "status": "pass",
        "artifact_count": len(rows),
        "artifacts": rows,
        "scientific_integrity_boundary": [
            "Do not present GPT Pro output as human peer review.",
            "Do not hide material counterevidence or fabricate external results.",
            "Do not infer human annotation, expert judgment, or reviewer identity.",
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
                mismatches.append({"path": name, "expected": expected_hash, "observed": observed})
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
        "human_review_evidence": False,
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
                "artifact_count": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0 if validation_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
