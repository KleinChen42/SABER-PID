"""Build a hash-verified, recoverable v2 technical submission package.

Only the frozen evidence route is packaged.  This script never calls legacy
generators that rewrite the manuscript and never restarts external downloads or
model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


RELEASE_VERSION = "pidqa-evidence-submission-v2"
PACKAGE_DIR = "release/pidqa_evidence_submission_v2"
ARCHIVE_PATH = "release/pidqa_evidence_submission_v2.zip"
ROOT_MANIFEST_JSON = "reports/generated/submission_release_manifest_v2.json"
ROOT_MANIFEST_CSV = "reports/generated/submission_release_manifest_v2.csv"
ARCHIVE_VALIDATION_JSON = "reports/generated/submission_archive_validation_v2.json"
RELEASE_NOTE = "reports/SUBMISSION_PACKAGE_RELEASE_V2.md"

EXPLICIT_PATHS = (
    "19_PAPER_EVIDENCE_STRENGTHENING_AND_SUBMISSION_EXECUTION_CHARTER.md",
    "README.md",
    "LICENSES.md",
    "CITATION.cff",
    "PACKAGE_INVENTORY.json",
    "pyproject.toml",
    "requirements-lock.txt",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/title_page.md",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "output/pdf/manuscript.pdf",
    "output/pdf/supplementary.pdf",
    "reports/EXPERIMENT_DECISION_LEDGER.md",
    "reports/PAPER_EVIDENCE_STRENGTHENING_CLOSEOUT.md",
    "reports/E1_EVIDENCE_AUDIT_CLOSEOUT.md",
    "reports/E2_VALUE_BUDGET_CLOSEOUT.md",
    "reports/E3_IMAGE_DEPENDENCE_CONTROL_CLOSEOUT.md",
    "reports/INTERNVL_CORRECTED_REPLICATION_CLOSEOUT.md",
    "reports/E5_ONTOLOGY_PROVENANCE_V1.md",
    "reports/E5_ONTOLOGY_VISIBILITY_CLOSEOUT.md",
    "reports/E6_SOURCE_SPLIT_SENSITIVITY_CLOSEOUT.md",
    "reports/F4_EXTERNAL_SOURCE_STATUS_V1.md",
    "reports/GIT_SNAPSHOT_STATUS_V2.md",
    "reports/generated/e1_evidence_audit_validation_v1.json",
    "reports/generated/evidence_input_answer_isolation_audit_v1.json",
    "reports/generated/effective_visual_budget_audit_v1.json",
    "reports/generated/effective_visual_budget_audit_v1.csv",
    "reports/generated/final_claim_evidence_matrix_v2.csv",
    "reports/generated/final_reproducibility_validation_v2.json",
    "reports/generated/final_statistical_summary_v2.json",
    "reports/generated/image_dependence_control_v1.json",
    "reports/generated/image_dependence_control_v1.csv",
    "reports/generated/internvl_tile_budget_v1.json",
    "reports/generated/internvl_tile_budget_v1.csv",
    "reports/generated/ontology_visibility_effect_v1.json",
    "reports/generated/ontology_visibility_effect_v1.csv",
    "reports/generated/output_budget_audit_v1.json",
    "reports/generated/output_budget_by_task_v1.csv",
    "reports/generated/pdf_render_validation_v2.json",
    "reports/generated/pidqa_loader_validation_v2.json",
    "reports/generated/pid2graph_recheck_v1.json",
    "reports/generated/qwen8_value_budget_sensitivity_v1.json",
    "reports/generated/qwen8_value_budget_sensitivity_v1.csv",
    "reports/generated/semantic_scoring_audit_v1.json",
    "reports/generated/set_b_model_vs_prior_bootstrap_v2.json",
    "reports/generated/set_b_model_vs_prior_table_v2.csv",
    "reports/generated/set_b_task_prior_v2.json",
    "reports/generated/source_seed_resolution_sensitivity_v1.json",
    "reports/generated/source_seed_resolution_sensitivity_v1.csv",
    "reports/generated/submission_package_validation_v2.json",
    "reports/generated/task_level_bootstrap_v2.json",
    "data/assets/pidqa_symbol_ontology_v1.png",
    "data/manifests/pidqa_symbol_ontology_v1.json",
    "data/manifests/set_b_source_shuffle_v1.json",
    "data/manifests/source_seed29_resolution_v1.json",
    "data/manifests/source_seed31_resolution_v1.json",
    "data/processed/main400_hashblind_set_b_public.jsonl",
    "data/processed/main400_hashblind_set_b_remote_public.jsonl",
    "data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl",
    "data/processed/main400_set_b_ontology_visible_v1_public.jsonl",
    "data/processed/source_seed29_resolution_v1_public.jsonl",
    "data/processed/source_seed29_resolution_v1_remote_public.jsonl",
    "data/processed/source_seed31_resolution_v1_public.jsonl",
    "data/processed/source_seed31_resolution_v1_remote_public.jsonl",
    "data/answer_store/main400_hashblind_set_b_hidden.jsonl",
    "data/answer_store/source_seed29_resolution_v1_hidden.jsonl",
    "data/answer_store/source_seed31_resolution_v1_hidden.jsonl",
    "scripts/audit_evidence_input_isolation.py",
    "scripts/attach_ontology_legend_v1.py",
    "scripts/build_e6_source_seed_sensitivity.py",
    "scripts/build_m0_evidence_freeze_v2.py",
    "scripts/build_pdf_render_validation_v2.py",
    "scripts/build_paper_figures_v2.py",
    "scripts/build_pidqa_symbol_ontology_v1.py",
    "scripts/build_submission_release_manifest_v2.py",
    "scripts/prepare_source_shuffled_control.py",
    "scripts/run_e1_evidence_audit.py",
    "scripts/run_final_submission_checks_v2.py",
    "scripts/run_internvl_tile_budget_v1.py",
    "scripts/run_qwen_evidence_matrix.py",
    "scripts/run_remote_internvl_tile_budget_v1.sh",
    "scripts/run_remote_qwen_evidence_condition.sh",
    "scripts/score_evidence_strengthening.py",
    "scripts/validate_submission_package_v2.py",
    "scripts/validate_submission_archive_v2.py",
    "scripts/verify_pidqa_loader_fixture_v2.py",
    RELEASE_NOTE,
)

GLOB_PATTERNS = (
    "paper/figures/*",
    "outputs/evidence_strengthening/**/*",
    "src/pidbench/**/*.py",
    "tests/test_*.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_note(root: Path) -> None:
    note = root / RELEASE_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        """# PIDQA evidence submission package v2

## Scope

This is the recoverable technical package for the authoritative
`19_PAPER_EVIDENCE_STRENGTHENING_AND_SUBMISSION_EXECUTION_CHARTER.md` route.
It contains the completed E1--E6 evidence chain, the bounded F4 external-data
status, the frozen claim matrix, LaTeX sources, separate vector/raster figures,
compiled PDFs, reproducibility scripts, selected answer-isolated manifests,
scorer-only references, and raw frozen model outputs.

## Reproduction boundary

The package can rebuild the deterministic audit, scoring, claim-freeze, figure,
and submission-validation layers from the included outputs. It intentionally
does not rerun model inference, restart PID2Graph retrieval, extract the
incomplete external archive, or claim an external PID2Graph/OPEN100 score.

Original PIDQA/Dataset-PID image assets are acquisition-by-reference rather
than redistributed here. The included manifests preserve their identifiers and
paths; the package retains only the selected public/hidden evaluation records
and the raw experiment results needed to regenerate the reported evidence.

## Submitter-owned fields

Author names, affiliations, corresponding-author contact information, final
declaration confirmation, and a permanent public archive URL remain explicit
placeholders. They must be supplied by the submitting authors; this package
does not fabricate them.

## Key entry points

```text
python scripts/run_final_submission_checks_v2.py --root .
python scripts/build_pdf_render_validation_v2.py --root .
python scripts/validate_submission_package_v2.py --root .
```

The release manifest and archive validation report record SHA-256 values for
every packaged artifact. The external-data boundary is documented in
`reports/F4_EXTERNAL_SOURCE_STATUS_V1.md`.
""",
        encoding="utf-8",
    )


def collect_paths(root: Path) -> list[Path]:
    missing = [relative for relative in EXPLICIT_PATHS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Required package artifacts are missing: " + ", ".join(missing))
    collected = {root / relative for relative in EXPLICIT_PATHS}
    for pattern in GLOB_PATTERNS:
        collected.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(collected, key=lambda path: path.relative_to(root).as_posix())


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = ["source_path", "package_path", "size_bytes", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_deterministic_zip(archive: Path, package_dir: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for source in sorted((path for path in package_dir.rglob("*") if path.is_file()), key=lambda path: path.relative_to(package_dir).as_posix()):
            name = source.relative_to(package_dir).as_posix()
            details = zipfile.ZipInfo(filename=name, date_time=(2026, 8, 10, 0, 0, 0))
            details.compress_type = zipfile.ZIP_DEFLATED
            details.external_attr = 0o100644 << 16
            handle.writestr(details, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package-dir", default=PACKAGE_DIR)
    parser.add_argument("--archive", default=ARCHIVE_PATH)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    write_release_note(root)
    package_dir = root / args.package_dir
    archive = root / args.archive
    if package_dir.exists() and any(package_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package directory: {package_dir}")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    package_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source in collect_paths(root):
        relative = source.relative_to(root)
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        if source_hash != sha256(destination):
            raise RuntimeError(f"Copy hash mismatch: {relative}")
        rows.append(
            {
                "source_path": relative.as_posix(),
                "package_path": relative.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
            }
        )

    package_manifest = {
        "release_version": RELEASE_VERSION,
        "status": "pass",
        "artifact_count": len(rows),
        "artifact_rows": rows,
        "external_boundary": {
            "pid2graph_status": "blocked_external_archive_incomplete",
            "external_score_reported": False,
            "action": "not_restarted_or_extracted",
        },
        "omitted_source_assets": ["data/raw/PIDQA/ (acquisition-by-reference)"],
        "submitter_owned_placeholders": [
            "author names and affiliations",
            "corresponding author contact details",
            "final declaration confirmation",
            "permanent public archive URL",
        ],
    }
    package_manifest_path = package_dir / "RELEASE_MANIFEST.json"
    package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(package_dir / "RELEASE_MANIFEST.csv", rows)

    archive_members = write_deterministic_zip(archive, package_dir)
    with zipfile.ZipFile(archive) as handle:
        verified_members = sorted(handle.namelist())
    expected_members = sorted(archive_members)
    archive_validation = {
        "release_version": RELEASE_VERSION,
        "status": "pass" if verified_members == expected_members else "fail",
        "archive": archive.relative_to(root).as_posix(),
        "archive_sha256": sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
        "member_count": len(verified_members),
        "package_manifest_sha256": sha256(package_manifest_path),
        "missing_members": sorted(set(expected_members) - set(verified_members)),
        "unexpected_members": sorted(set(verified_members) - set(expected_members)),
    }
    root_manifest = {
        "release_version": RELEASE_VERSION,
        "status": archive_validation["status"],
        "package_directory": package_dir.relative_to(root).as_posix(),
        "archive": archive_validation,
        "artifact_count": len(rows),
        "artifact_rows": rows,
        "external_boundary": package_manifest["external_boundary"],
        "omitted_source_assets": package_manifest["omitted_source_assets"],
    }
    json_path = root / ROOT_MANIFEST_JSON
    csv_path = root / ROOT_MANIFEST_CSV
    validation_path = root / ARCHIVE_VALIDATION_JSON
    for path in (json_path, csv_path, validation_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    validation_path.write_text(json.dumps(archive_validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": root_manifest["status"], "artifact_count": len(rows), "archive": str(archive)}, sort_keys=True))
    return 0 if root_manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
