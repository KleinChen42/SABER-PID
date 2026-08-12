"""Build deterministic v4 review and full reproducibility archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


VERSION = "pidqa-evidence-submission-v4"
ROOT_JSON = "reports/generated/submission_release_manifest_v4.json"
ROOT_CSV = "reports/generated/submission_release_manifest_v4.csv"
VALIDATION_JSON = "reports/generated/submission_archive_validation_v4.json"
RELEASE_NOTE = "reports/SUBMISSION_PACKAGE_RELEASE_V4.md"

BASE_PATHS = (
    "README.md",
    "LICENSES.md",
    "CITATION.cff",
    "pyproject.toml",
    "requirements-lock.txt",
    "review/SABER_PID_model_based_editorial_review.md",
    "licenses/PIDQA_LICENSE.txt",
    "reports/MODEL_BASED_EDITORIAL_REVISION_CLOSEOUT_V4.md",
    "reports/EXPERIMENT_DECISION_LEDGER.md",
    "output/pdf/manuscript.pdf",
    "output/pdf/supplementary.pdf",
)

REVIEW_PATHS = (
    "paper/manuscript.tex",
    "paper/supplementary.tex",
    "paper/templates/manuscript_v4.tex.in",
    "paper/templates/supplementary_v4.tex.in",
    "paper/title_page.md",
    "paper/highlights.md",
    "paper/cover_letter.md",
    "paper/data_availability.md",
    "paper/declarations.md",
    "paper/figure_manifest.md",
    "paper/figure_captions.md",
    "paper/assets/pidqa_sheet_282.jpg",
    "paper/assets/pidqa_sheet_184.jpg",
    "paper/figures/figure_1_counterfactual_evidence_ladder.pdf",
    "paper/figures/figure_1_counterfactual_evidence_ladder.png",
    "paper/figures/figure_2_core_effects_v4.pdf",
    "paper/figures/figure_2_core_effects_v4.png",
    "paper/figures/figure_3_task_calibration_v4.pdf",
    "paper/figures/figure_3_task_calibration_v4.png",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.pdf",
    "paper/figures/figure_s1_controls_and_operating_quantities_v4.png",
    "paper/figures/figure_s2_tag_reading_stability_v4.pdf",
    "paper/figures/figure_s2_tag_reading_stability_v4.png",
    "paper/figures/figure_metadata_v4.json",
    "reports/generated/pidqa_input_retrieval_seed_sweep.json",
    "reports/generated/set_b_task_prior_v2.json",
    "reports/generated/qwen8_value_budget_sensitivity_v1.json",
    "reports/generated/image_dependence_control_v1.json",
    "reports/generated/internvl_tile_budget_v1.json",
    "reports/generated/ontology_visibility_effect_v1.json",
    "reports/generated/source_seed_resolution_sensitivity_v1.json",
    "reports/generated/ontology_mapping_control_v1.json",
    "reports/generated/text_only_image_grounding_control_v1.json",
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/pid2graph_recheck_v1.json",
    "reports/generated/editorial_revision_evidence_v4.json",
    "reports/generated/editorial_revision_task_effects_v4.csv",
    "reports/generated/editorial_extension_experiments_v4.json",
    "reports/generated/editorial_extension_experiments_v4.csv",
    "reports/generated/editorial_revision_submission_v4.json",
    "reports/generated/paddleocr_environment_v1.txt",
    "reports/generated/paddleocr_model_artifacts_v1.json",
    "reports/generated/internvl35_8b_editorial_checkpoint_v1.json",
    "reports/generated/pidqa_loader_validation_v2.json",
    "reports/generated/reproduction_validation_v4.json",
    "reports/generated/pdf_render_validation_v4.json",
    "reports/generated/pdf_visual_inspection_v4.json",
    "reports/generated/submission_package_validation_v4.json",
    "data/assets/pidqa_symbol_ontology_v1.png",
    "data/assets/pidqa_symbol_ontology_label_shift1_v1.png",
    "data/manifests/pidqa_symbol_ontology_v1.json",
    "data/manifests/pidqa_symbol_ontology_label_shift1_v1.json",
    "data/manifests/set_b_source_shuffle_v1.json",
    "data/manifests/set_b_source_shuffle_v1_remote.json",
    "data/manifests/source_seed29_resolution_v1.json",
    "data/manifests/source_seed31_resolution_v1.json",
    "data/processed/main400_hashblind_set_b_remote_public.jsonl",
    "data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl",
    "data/processed/main400_hashblind_set_b_shuffled_v1_remote_public.jsonl",
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
    "outputs/editorial_revision/internvl_counterfactual_ladder_v1/internvl35_8b_value_correct.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v1/internvl35_8b_value_shuffled.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v1/internvl35_8b_value_text_only.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/internvl35_8b_value_correct.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/internvl35_8b_value_shuffled.jsonl",
    "outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/internvl35_8b_value_text_only.jsonl",
    "outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl",
)

REVIEW_GLOBS = (
    "licenses/*",
    "scripts/*.py",
    "scripts/*.sh",
    "src/**/*.py",
    "tests/test_*.py",
)

FULL_EXTRA_GLOBS = (
    "0[1-9]_*.md",
    "1[0-9]_*.md",
    "2[0-9]_*.md",
    "reports/**/*.md",
    "reports/generated/*",
    "reports/logs/*",
    "paper/**/*",
    "data/assets/*",
    "data/manifests/*",
    "data/processed/**/*.jsonl",
    "data/answer_store/*.jsonl",
    "outputs/**/*.jsonl",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = ("package", "path", "size_bytes", "sha256")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_release_note(root: Path) -> None:
    (root / RELEASE_NOTE).write_text(
        """# SABER-PID evidence submission package v4

This package implements the supplied model-based editorial review.  The paper
is framed as a counterfactual engineering measurement study: source isolation
closes an observed retrieval route, and correct/wrong/no-image controls locate
what a reported score measures at frozen operating points.  SABER-PID is a
demonstrated audit instrument, not a universal protocol or autonomous design
review system.

All directions of the frozen Qwen, tokenizer-corrected InternVL3.5-8B,
PaddleOCR, source-split,
visible-context, and label-permutation results are retained.  The random-split
diagnostic is not described as observed VLM leakage, and a zero detected
mapping contrast is not described as equivalence.  PIDQA is one public
synthetic source family, so external plant-drawing validity remains open.

The archives omit model weights and the acquisition-by-reference original
PIDQA collection.  Two CC0 source drawings used in deterministic Figure 1 are
included under `paper/assets/`.  No human participant, human rater, or real
peer-review evidence is claimed; the supplied critique is explicitly retained
as model-based editorial review.

Rebuild the frozen evidence layer from either archive with:

```text
python scripts/reproduce_submission_v4.py --root .
```

The submitter must still replace author, affiliation, funding, competing-
interest, CRediT, originality, and permanent-archive placeholders before
journal upload.
""",
        encoding="utf-8",
    )


def collect(root: Path, package: str) -> list[Path]:
    required = list(BASE_PATHS) + list(REVIEW_PATHS) + [RELEASE_NOTE]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required release artifacts: " + ", ".join(missing))
    paths = {root / name for name in required}
    for pattern in REVIEW_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    if package == "full":
        for pattern in FULL_EXTRA_GLOBS:
            paths.update(path for path in root.glob(pattern) if path.is_file())
    excluded = ("release/", "tmp/", ".git/", "data/raw/")
    return sorted(
        (
            path
            for path in paths
            if not path.relative_to(root).as_posix().startswith(excluded)
            and "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def deterministic_zip(source_dir: Path, destination: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(
            (path for path in source_dir.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(source_dir).as_posix(),
        ):
            name = source.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            names.append(name)
    return names


def build_one(root: Path, package: str) -> dict[str, Any]:
    release = root / "release"
    final_dir = release / f"pidqa_evidence_submission_v4_{package}"
    final_zip = release / f"pidqa_evidence_submission_v4_{package}.zip"
    if final_dir.exists() or final_zip.exists():
        raise FileExistsError(f"Refusing to overwrite existing v4 release target: {final_dir} / {final_zip}")
    release.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"saber_v4_{package}_", dir=release) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        rows: list[dict[str, Any]] = []
        for source in collect(root, package):
            relative = source.relative_to(root)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rows.append(
                {
                    "package": package,
                    "path": relative.as_posix(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )
        manifest = {
            "release_version": VERSION,
            "package_kind": package,
            "status": "pass",
            "artifact_count": len(rows),
            "artifacts": rows,
            "inference_rerun_required": False,
            "model_review_label": "model-based editorial review; not real human peer review",
            "claim_boundary": {
                "source_split": "same-source diagnostic route; not observed VLM leakage",
                "mapping_control": "no mapping-specific advantage detected; not equivalence",
                "external_validity": "single public synthetic source family",
                "deployment": "does not justify autonomous design review",
            },
            "omitted": ["model weights", "data/raw/PIDQA acquisition-by-reference collection"],
        }
        (staging / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(staging / "RELEASE_MANIFEST.csv", rows)
        staged_zip = Path(temporary) / "archive.zip"
        members = deterministic_zip(staging, staged_zip)
        with zipfile.ZipFile(staged_zip) as archive:
            observed = sorted(archive.namelist())
        if observed != sorted(members):
            raise RuntimeError(f"Archive member mismatch for {package}")
        shutil.move(str(staging), str(final_dir))
        shutil.move(str(staged_zip), str(final_zip))
    return {
        "package_kind": package,
        "directory": final_dir.relative_to(root).as_posix(),
        "archive": final_zip.relative_to(root).as_posix(),
        "archive_sha256": sha256(final_zip),
        "archive_size_bytes": final_zip.stat().st_size,
        "member_count": len(members),
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", choices=("review", "full", "both"), default="both")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    write_release_note(root)
    kinds = ("review", "full") if args.package == "both" else (args.package,)
    packages = [build_one(root, kind) for kind in kinds]
    report = {"release_version": VERSION, "status": "pass", "packages": packages}
    json_path = root / ROOT_JSON
    csv_path = root / ROOT_CSV
    validation_path = root / VALIDATION_JSON
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, (row for package in packages for row in package["artifacts"]))
    validation_path.write_text(
        json.dumps(
            {
                "release_version": VERSION,
                "status": "pass",
                "packages": [
                    {key: value for key, value in package.items() if key != "artifacts"}
                    for package in packages
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "packages": len(packages)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
