"""Build the deterministic RINENG v6 public reproducibility release candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


VERSION = "pidqa-rineng-qualification-submission-v6"
KIND = "public-release-candidate"
STEM = "pidqa_rineng_submission_v6_public_release_candidate"
ROOT_REPORT = "reports/generated/rineng_public_release_manifest_v6.json"
ROOT_CSV = "reports/generated/rineng_public_release_manifest_v6.csv"

CORE_PATHS = (
    "README.md",
    "LICENSES.md",
    "CITATION.cff",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements-analysis-v6.txt",
    "licenses/PIDQA_LICENSE.txt",
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/title_page.md",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/assets/pidqa_sheet_282.jpg",
    "paper/assets/pidqa_sheet_184.jpg",
    "paper/figures/figure_1_qualification_decision_v6.pdf",
    "paper/figures/figure_1_qualification_decision_v6.png",
    "paper/figures/figure_2_qualification_effects_v6.pdf",
    "paper/figures/figure_2_qualification_effects_v6.png",
    "paper/figures/figure_3_operating_modes_v6.pdf",
    "paper/figures/figure_3_operating_modes_v6.png",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.png",
    "paper/figures/figure_metadata_v6.json",
    "output/pdf/v6/manuscript.pdf",
    "output/pdf/v6/supplementary.pdf",
)

SCRIPT_PATHS = (
    "scripts/audit_evidence_input_isolation.py",
    "scripts/build_editorial_revision_submission_v4.py",
    "scripts/build_paper_figures_v4.py",
    "scripts/build_pdf_render_validation_v4.py",
    "scripts/build_positive_narrative_hybrid_analysis_v5.py",
    "scripts/build_rineng_revision_analysis_v6.py",
    "scripts/build_rineng_revision_figures_v6.py",
    "scripts/build_rineng_public_release_v6.py",
    "scripts/reproduce_submission_v4.py",
    "scripts/reproduce_submission_v6.py",
    "scripts/run_e1_evidence_audit.py",
    "scripts/score_editorial_extension_experiments_v4.py",
    "scripts/score_evidence_strengthening.py",
    "scripts/validate_rineng_submission_v6.py",
    "scripts/validate_rineng_public_archive_v6.py",
    "scripts/run_rineng_archive_clean_check_v6.py",
)

REPORT_PATHS = (
    "reports/generated/editorial_revision_evidence_v4.json",
    "reports/generated/qwen8_value_budget_sensitivity_v1.json",
    "reports/generated/image_dependence_control_v1.json",
    "reports/generated/internvl_tile_budget_v1.json",
    "reports/generated/ontology_visibility_effect_v1.json",
    "reports/generated/source_seed_resolution_sensitivity_v1.json",
    "reports/generated/ontology_mapping_control_v1.json",
    "reports/generated/text_only_image_grounding_control_v1.json",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/rineng_revision_analysis_v6.json",
    "reports/generated/rineng_revision_analysis_v6.csv",
    "reports/generated/rineng_revision_per_source_v6.csv",
    "reports/generated/rineng_revision_error_taxonomy_v6.csv",
    "reports/generated/rineng_revision_environment_v6.json",
    "reports/generated/rineng_submission_validation_v6.json",
    "reports/generated/pdf_render_validation_v6.json",
    "reports/generated/pdf_visual_inspection_v6.json",
)

DATA_PATHS = (
    "data/processed/pidqa_records.jsonl",
    "data/processed/main400_hashblind_set_b_remote_public.jsonl",
    "data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl",
    "data/processed/main400_set_b_ontology_visible_v1_public.jsonl",
    "data/processed/source_seed29_resolution_v1_remote_public.jsonl",
    "data/processed/source_seed31_resolution_v1_remote_public.jsonl",
)

OUTPUT_PATHS = (
    "outputs/final_replication/qwen8_b_p0_3072.jsonl",
    "outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl",
    "outputs/positive_narrative/paddleocr_seed29_v1.jsonl",
    "outputs/positive_narrative/paddleocr_seed31_v1.jsonl",
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
)

README_TEXT = """# SABER-PID Results in Engineering v6 reproducibility release candidate

This archive accompanies the manuscript **Qualifying Image-Grounded Tag
Retrieval in Piping and Instrumentation Diagrams with Source-Isolated
Counterfactual Evaluation**.

## Supported result

The qualified outcome is candidate value-tag retrieval for the frozen
PIDQA/Qwen operating point. The archive does not qualify topology reasoning,
general P&ID understanding, cross-model invariance, or real-plant deployment.
The OCR--VLM rules are deterministic prediction-only operating modes.

## Reproduce

Install the packages in requirements-analysis-v6.txt, make Tectonic 0.17.0
available on PATH, and run:

    python scripts/reproduce_submission_v6.py --root .

The command performs no model inference and no external download. It rebuilds
the v6 analyses and figures from immutable outputs, runs 29 tests, compiles the
manuscript and supplement with frozen PDF metadata, renders every page, and
checks the recorded all-page visual inspection against identical rendered
page hashes.

## Evidence separation

Model-facing public manifests do not contain answer or Cypher fields.
data/processed/pidqa_records.jsonl is scorer-only reference material.
References are used to evaluate predictions, never to construct them. The
archive contains raw model/OCR responses but no model weights.

## Exclusions

Author-side editorial prompts, model-review text, private working notes, the
acquisition-by-reference PIDQA image collection, invalid external archives,
and model weights are excluded. Two CC0 source drawings used by the legacy
supplementary boundary-figure generator are retained with the PIDQA license.

## Administrative boundary

The technical archive is complete and locally clean-run validated. A public
DOI/URL and a project-code license choice remain submitter-owned. Dataset
material is covered by the vendored PIDQA CC0 record; the absence of a root
project-code license is disclosed in LICENSES.md and must be resolved before
public hosting if reuse permission is intended.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = ("path", "size_bytes", "sha256")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> list[Path]:
    required = CORE_PATHS + SCRIPT_PATHS + REPORT_PATHS + DATA_PATHS + OUTPUT_PATHS
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing release artifacts: " + ", ".join(missing))
    paths = {root / relative for relative in required}
    paths.update(path for path in (root / "src").rglob("*.py") if path.is_file())
    paths.update(path for path in (root / "tests").rglob("*.py") if path.is_file())
    paths.update(path for path in (root / "tests/fixtures").rglob("*.csv") if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def deterministic_zip(source: Path, destination: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            (candidate for candidate in source.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(source).as_posix(),
        ):
            name = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
            names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    release = root / "release"
    final_dir = release / STEM
    final_zip = release / f"{STEM}.zip"
    if final_dir.exists() or final_zip.exists():
        raise FileExistsError(
            f"Refusing to overwrite generated release target: {final_dir} / {final_zip}"
        )
    release.mkdir(parents=True, exist_ok=True)

    staging = release / f".{STEM}_staging"
    staged_zip = release / f".{STEM}.zip.partial"
    if staging.exists() or staged_zip.exists():
        raise FileExistsError(
            f"Refusing to reuse staging target: {staging} / {staged_zip}"
        )
    staging.mkdir()
    for source in collect(root):
        relative = source.relative_to(root)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (staging / "RELEASE_README.md").write_text(README_TEXT, encoding="utf-8")

    rows = [
        {
            "path": path.relative_to(staging).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(
            (candidate for candidate in staging.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(staging).as_posix(),
        )
    ]
    manifest = {
            "release_version": VERSION,
            "package_kind": KIND,
            "status": "pass",
            "technical_release_candidate_complete": True,
            "public_doi_or_url_present": False,
            "project_code_license_selected": False,
            "artifact_count": len(rows),
            "artifacts": rows,
            "inference_rerun_required": False,
            "reference_used_in_prediction_construction": False,
            "human_review_evidence_claimed": False,
            "claim_scope": (
                "candidate value-tag retrieval within the evaluated "
                "PIDQA/Qwen/model/budget setting"
            ),
            "omitted": [
                "author-side editorial prompts and model-review text",
                "private working notes",
                "model weights",
                "data/raw acquisition-by-reference image collection",
                "invalid PID2Graph/OPEN100 archive",
            ],
            "submitter_actions": [
                "select project-code license",
                "upload archive and insert public DOI/URL",
                "complete author-owned declarations",
            ],
    }
    (staging / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(staging / "RELEASE_MANIFEST.csv", rows)

    members = deterministic_zip(staging, staged_zip)
    with zipfile.ZipFile(staged_zip) as archive:
        if sorted(archive.namelist()) != sorted(members):
            raise RuntimeError("Archive member mismatch after creation")
    shutil.move(str(staging), str(final_dir))
    shutil.move(str(staged_zip), str(final_zip))

    report = {
        "release_version": VERSION,
        "status": "pass",
        "package_kind": KIND,
        "directory": final_dir.relative_to(root).as_posix(),
        "archive": final_zip.relative_to(root).as_posix(),
        "archive_sha256": sha256(final_zip),
        "archive_size_bytes": final_zip.stat().st_size,
        "member_count": len(members),
        "artifact_count": len(rows),
        "technical_release_candidate_complete": True,
        "external_boundary": {
            "public_doi_or_url": "submitter upload required",
            "project_code_license": "submitter decision required",
        },
    }
    output = root / ROOT_REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(root / ROOT_CSV, rows)
    print(
        json.dumps(
            {
                "status": "pass",
                "archive": report["archive"],
                "members": report["member_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
