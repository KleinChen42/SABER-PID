"""Build lean review and full reproducibility v3 submission archives.

The builder packages only frozen artifacts.  It never reruns inference, alters
the paper, or restarts the blocked external PID2Graph/OPEN100 branch.  The
review archive is intentionally compact; the full archive adds all local
planning, reporting, deterministic-code, manifest, and raw-output records
while still omitting original PIDQA image assets and model weights.
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


RELEASE_VERSION = "pidqa-evidence-submission-v3"
ROOT_MANIFEST_JSON = "reports/generated/submission_release_manifest_v3.json"
ROOT_MANIFEST_CSV = "reports/generated/submission_release_manifest_v3.csv"
ARCHIVE_VALIDATION_JSON = "reports/generated/submission_archive_validation_v3.json"
RELEASE_NOTE = "reports/SUBMISSION_PACKAGE_RELEASE_V3.md"

COMMON_PATHS = (
    "20_POSITIVE_NARRATIVE_SELF_REVIEW_AND_REVISION_CHARTER.md",
    "README.md",
    "LICENSES.md",
    "CITATION.cff",
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
    "reports/POSITIVE_NARRATIVE_REVISION_CLOSEOUT_V3.md",
    "reports/PAPER_EVIDENCE_STRENGTHENING_CLOSEOUT.md",
    "reports/E1_EVIDENCE_AUDIT_CLOSEOUT.md",
    "reports/E2_VALUE_BUDGET_CLOSEOUT.md",
    "reports/E3_IMAGE_DEPENDENCE_CONTROL_CLOSEOUT.md",
    "reports/INTERNVL_CORRECTED_REPLICATION_CLOSEOUT.md",
    "reports/E5_ONTOLOGY_PROVENANCE_V1.md",
    "reports/E5_ONTOLOGY_VISIBILITY_CLOSEOUT.md",
    "reports/E6_SOURCE_SPLIT_SENSITIVITY_CLOSEOUT.md",
    "reports/E7_ONTOLOGY_MAPPING_CONTROL_CLOSEOUT.md",
    "reports/E8_TEXT_ONLY_IMAGE_GROUNDING_CLOSEOUT.md",
    "reports/F4_EXTERNAL_SOURCE_STATUS_V1.md",
    "reports/generated/pidqa_input_retrieval_seed_sweep.json",
    "reports/generated/set_b_task_prior_v2.json",
    "reports/generated/qwen8_value_budget_sensitivity_v1.json",
    "reports/generated/image_dependence_control_v1.json",
    "reports/generated/internvl_tile_budget_v1.json",
    "reports/generated/ontology_visibility_effect_v1.json",
    "reports/generated/source_seed_resolution_sensitivity_v1.json",
    "reports/generated/ontology_mapping_control_v1.json",
    "reports/generated/ontology_mapping_control_v1.csv",
    "reports/generated/text_only_image_grounding_control_v1.json",
    "reports/generated/text_only_image_grounding_control_v1.csv",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/final_statistical_summary_v3.json",
    "reports/generated/final_claim_evidence_matrix_v3.csv",
    "reports/generated/manuscript_number_audit_v3.json",
    "reports/generated/pdf_render_validation_v3.json",
    "reports/generated/final_reproducibility_validation_v3.json",
    "reports/generated/submission_package_validation_v3.json",
    "reports/generated/pid2graph_recheck_v1.json",
    "data/assets/pidqa_symbol_ontology_v1.png",
    "data/assets/pidqa_symbol_ontology_label_shift1_v1.png",
    "data/manifests/pidqa_symbol_ontology_v1.json",
    "data/manifests/pidqa_symbol_ontology_label_shift1_v1.json",
    "data/manifests/set_b_source_shuffle_v1.json",
    "data/manifests/source_seed29_resolution_v1.json",
    "data/manifests/source_seed31_resolution_v1.json",
    "data/processed/main400_hashblind_set_b_remote_public.jsonl",
    "data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl",
    "data/processed/main400_set_b_ontology_visible_v1_public.jsonl",
    "data/processed/main400_set_b_ontology_label_shift1_v1_public.jsonl",
    "data/processed/source_seed29_resolution_v1_remote_public.jsonl",
    "data/processed/source_seed31_resolution_v1_remote_public.jsonl",
    "data/answer_store/main400_hashblind_set_b_hidden.jsonl",
    "data/answer_store/source_seed29_resolution_v1_hidden.jsonl",
    "data/answer_store/source_seed31_resolution_v1_hidden.jsonl",
    "outputs/final_replication/qwen8_b_p0_768.jsonl",
    "outputs/final_replication/qwen8_b_p0_3072.jsonl",
    "outputs/evidence_strengthening/set_b_task_prior_v2.jsonl",
    "outputs/evidence_strengthening/qwen8_value_budget_v1/qwen8_value_budget_v1_768.jsonl",
    "outputs/evidence_strengthening/qwen8_value_budget_v1/qwen8_value_budget_v1_3072.jsonl",
    "outputs/evidence_strengthening/qwen8_image_shuffle_v1/qwen8_image_shuffle_v1_768.jsonl",
    "outputs/evidence_strengthening/qwen8_image_shuffle_v1/qwen8_image_shuffle_v1_3072.jsonl",
    "outputs/evidence_strengthening/qwen8_ontology_visible_v1/qwen8_ontology_visible_v1_768.jsonl",
    "outputs/evidence_strengthening/qwen8_ontology_visible_v1/qwen8_ontology_visible_v1_3072.jsonl",
    "outputs/evidence_strengthening/qwen8_ontology_permuted_v1/qwen8_ontology_permuted_v1_768.jsonl",
    "outputs/evidence_strengthening/qwen8_ontology_permuted_v1/qwen8_ontology_permuted_v1_3072.jsonl",
    "outputs/evidence_strengthening/qwen8_text_only_v1/qwen8_text_only_v1_3072.jsonl",
    "outputs/evidence_strengthening/qwen8_source_seed29_resolution_v1/qwen8_source_seed29_resolution_v1_768.jsonl",
    "outputs/evidence_strengthening/qwen8_source_seed29_resolution_v1/qwen8_source_seed29_resolution_v1_3072.jsonl",
    "outputs/evidence_strengthening/qwen8_source_seed31_resolution_v1/qwen8_source_seed31_resolution_v1_768.jsonl",
    "outputs/evidence_strengthening/qwen8_source_seed31_resolution_v1/qwen8_source_seed31_resolution_v1_3072.jsonl",
    "outputs/evidence_strengthening/internvl_tile_budget_v1/internvl35_b_tile_low.jsonl",
    "outputs/evidence_strengthening/internvl_tile_budget_v1/internvl35_b_tile_high.jsonl",
    "scripts/audit_evidence_input_isolation.py",
    "scripts/build_pdf_render_validation_v3.py",
    "scripts/build_paper_figures_v3.py",
    "scripts/build_pidqa_symbol_ontology_v1.py",
    "scripts/build_positive_narrative_submission_v3.py",
    "scripts/build_submission_release_manifest_v3.py",
    "scripts/run_e1_evidence_audit.py",
    "scripts/run_final_submission_checks_v3.py",
    "scripts/run_release_reproducibility_checks_v3.py",
    "scripts/run_qwen_evidence_matrix.py",
    "scripts/run_remote_qwen_evidence_condition.sh",
    "scripts/score_evidence_strengthening.py",
    "scripts/score_positive_narrative_controls.py",
    "scripts/validate_submission_archive_v3.py",
    "scripts/validate_submission_package_v3.py",
    "scripts/verify_pidqa_loader_fixture_v2.py",
)

COMMON_GLOBS = (
    "paper/figures/*",
    "src/pidbench/**/*.py",
    "tests/test_*.py",
)
FULL_EXTRA_GLOBS = (
    "0[1-9]_*.md",
    "1[0-9]_*.md",
    "reports/**/*.md",
    "reports/generated/*",
    "data/manifests/*",
    "data/processed/**/*.jsonl",
    "data/answer_store/*.jsonl",
    "outputs/**/*.jsonl",
    "scripts/*.py",
    "scripts/*.sh",
    "src/**/*.py",
    "tests/**/*.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = ("package", "source_path", "package_path", "size_bytes", "sha256")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_release_note(root: Path) -> None:
    path = root / RELEASE_NOTE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# PIDQA evidence submission package v3

## Scope

This release implements the evidence-aligned revision charter in
`20_POSITIVE_NARRATIVE_SELF_REVIEW_AND_REVISION_CHARTER.md`. It contains the
SABER-PID manuscript, supplement, deterministic figures, frozen raw outputs,
answer-isolated manifests, scorer-only references, source-cluster statistics,
and hash manifests supporting E1--E8.

The positive findings are deliberately bounded: source isolation closes a
retrieval route; correct images support Qwen value/tag reading; and visible
symbol context changes spatial-count calibration. E7 retains the zero
correct-map-minus-permuted-map result, so the package does not attribute the
context effect to numeric mapping semantics. E8 retains text-only calibration,
so the package does not claim a universal advantage for visual input on every
task. The corrected InternVL result remains a boundary control.

## Archive roles

- `pidqa_evidence_submission_v3_review.zip` is the lean peer-review package.
- `pidqa_evidence_submission_v3_full.zip` is the complete local reproducibility
  archive, including prior plans and all local non-image raw artifacts.

Neither archive redistributes original PIDQA/Dataset-PID images, model weights,
or the incomplete PID2Graph/OPEN100 archive. The reported package never reruns
model inference or reports an external PID2Graph score.

## Deterministic reconstruction

```text
python scripts/run_release_reproducibility_checks_v3.py --root .
tectonic --keep-logs --outdir output/pdf paper/manuscript.tex
tectonic --keep-logs --outdir output/pdf paper/supplementary.tex
python scripts/build_pdf_render_validation_v3.py --root .
python scripts/validate_submission_package_v3.py --root .
python scripts/validate_submission_archive_v3.py --root .
```

The workspace-wide `run_final_submission_checks_v3.py` additionally validates
the upstream raw PIDQA loader and is intended only after acquiring the original
data under its upstream terms.

Authors, affiliations, funding, competing-interest declarations, originality
confirmation, and a permanent archive URL remain submitter-owned fields.
""",
        encoding="utf-8",
    )


def collect_paths(root: Path, package: str) -> list[Path]:
    required = list(COMMON_PATHS) + [RELEASE_NOTE]
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Required package artifacts are missing: " + ", ".join(missing))
    paths = {root / relative for relative in required}
    for pattern in COMMON_GLOBS:
        paths.update(candidate for candidate in root.glob(pattern) if candidate.is_file())
    if package == "full":
        for pattern in FULL_EXTRA_GLOBS:
            paths.update(candidate for candidate in root.glob(pattern) if candidate.is_file())
    excluded_prefixes = ("release/", "tmp/")
    return sorted(
        (path for path in paths if not path.relative_to(root).as_posix().startswith(excluded_prefixes)),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def write_deterministic_zip(archive: Path, package_dir: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for source in sorted(
            (candidate for candidate in package_dir.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(package_dir).as_posix(),
        ):
            name = source.relative_to(package_dir).as_posix()
            details = zipfile.ZipInfo(filename=name, date_time=(2026, 8, 10, 0, 0, 0))
            details.compress_type = zipfile.ZIP_DEFLATED
            details.external_attr = 0o100644 << 16
            handle.writestr(details, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            names.append(name)
    return names


def build_one(root: Path, package: str) -> dict[str, Any]:
    directory = root / f"release/pidqa_evidence_submission_v3_{package}"
    archive = root / f"release/pidqa_evidence_submission_v3_{package}.zip"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package directory: {directory}")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    directory.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in collect_paths(root, package):
        relative = source.relative_to(root)
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        if source_hash != sha256(destination):
            raise RuntimeError(f"Copy hash mismatch: {relative}")
        rows.append(
            {
                "package": package,
                "source_path": relative.as_posix(),
                "package_path": relative.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
            }
        )
    package_manifest = {
        "release_version": RELEASE_VERSION,
        "package_kind": package,
        "status": "pass",
        "artifact_count": len(rows),
        "artifact_rows": rows,
        "claim_boundary": {
            "e7": "visible-context effect retained; numeric mapping attribution unsupported",
            "e8": "image-grounded value/tag reading retained; structural-task visual superiority not claimed",
            "e4": "InternVL boundary control retained; universal visual-budget law not claimed",
            "pid2graph": "external archive blocked; no external score reported",
        },
        "omitted_assets": [
            "data/raw/PIDQA/ original images (acquisition-by-reference)",
            "remote model weights",
            "PID2Graph/OPEN100 archive",
        ],
        "submitter_owned_placeholders": [
            "author and affiliation data",
            "funding and competing-interest declarations",
            "originality/exclusive-submission confirmation",
            "permanent archive URL",
        ],
    }
    manifest_path = directory / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(directory / "RELEASE_MANIFEST.csv", rows)
    members = write_deterministic_zip(archive, directory)
    with zipfile.ZipFile(archive) as handle:
        observed_members = sorted(handle.namelist())
    expected_members = sorted(members)
    validation = {
        "package_kind": package,
        "status": "pass" if observed_members == expected_members else "fail",
        "archive": archive.relative_to(root).as_posix(),
        "archive_sha256": sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
        "member_count": len(observed_members),
        "package_manifest_sha256": sha256(manifest_path),
        "missing_members": sorted(set(expected_members) - set(observed_members)),
        "unexpected_members": sorted(set(observed_members) - set(expected_members)),
    }
    return {
        "package_kind": package,
        "package_directory": directory.relative_to(root).as_posix(),
        "archive": validation,
        "artifact_count": len(rows),
        "artifact_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", choices=("review", "full", "both"), default="both")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    write_release_note(root)
    packages = ("review", "full") if args.package == "both" else (args.package,)
    records = [build_one(root, package) for package in packages]
    status = "pass" if all(item["archive"]["status"] == "pass" for item in records) else "fail"
    root_manifest = {
        "release_version": RELEASE_VERSION,
        "status": status,
        "packages": records,
        "external_boundary": {
            "pid2graph_status": "blocked_external_archive_incomplete",
            "external_score_reported": False,
            "action": "not_restarted_or_extracted",
        },
    }
    json_path = root / ROOT_MANIFEST_JSON
    csv_path = root / ROOT_MANIFEST_CSV
    validation_path = root / ARCHIVE_VALIDATION_JSON
    all_rows = [row for item in records for row in item["artifact_rows"]]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, all_rows)
    validation_path.write_text(
        json.dumps(
            {
                "release_version": RELEASE_VERSION,
                "status": status,
                "packages": [item["archive"] for item in records],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "packages": len(records)}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
