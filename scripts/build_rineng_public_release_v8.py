"""Build a deterministic public reproducibility release for SABER-PID V8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


VERSION = "saber-pid-rineng-v8"
STEM = "saber_pid_rineng_v8_public_release"
ZIP_TIME = (2026, 8, 12, 0, 0, 0)


README = """# SABER-PID Results in Engineering V8 reproducibility release

This archive accompanies **Qualifying Image-Grounded Tag Retrieval in Piping
and Instrumentation Diagrams with Source-Isolated Counterfactual Evaluation**.

## Evidence in this release

- source-isolated PIDQA qualification and frozen V7 cross-model replication;
- scorer-only cost-sensitive operating-mode decisions;
- paired clean/JPEG/blur/downsample Qwen robustness at 3072-side/512 tokens;
- the closest 512-token-safe 54-tile InternVL3.5-8B boundary comparison;
- a second public P&ID family from DEXPI, with correct/shuffled/no-image Qwen
  and frozen full-image OCR;
- an automated PID2Graph/OPEN100 task-fit audit that retains its structural
  annotations but does not invent an unavailable visible-tag reference;
- independent metric and bootstrap validation, manuscript tables, figures,
  PDFs, tests, and SHA-256 inventories.

## Inference-free reproduction

Install the declared Python analysis dependencies, then run from the archive
root:

    python scripts/reproduce_rineng_v8_extensions.py --root .

The command performs no inference and no download. It rescans immutable raw
responses, rebuilds all V8 estimates and 10,000-replicate intervals, runs an
independent-seed validator, and regenerates V8 figures and tables.

## Data and model boundary

Model-facing manifests contain no answer or Cypher field. Scorer-only
references are explicitly separated. PIDQA is covered by the vendored CC0
record. DEXPI provenance is frozen to official repository commit
`a23d61e2e089eb2ca464cd552f9ae580a2785963`, with its byte-preserved CC BY 4.0
license. Upstream drawing collections and model weights are acquisition by
reference and are not redistributed. The release supports candidate-tag
retrieval under the declared families and operating budgets; it does not
qualify topology reasoning, arbitrary models, or real-plant deployment.

## Administrative fields

The public DOI/URL, authors, affiliations, declarations, and project-code
license choice remain submitter-owned fields to complete before public upload.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = ("path", "size_bytes", "sha256")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_zip(source: Path, destination: Path) -> int:
    count = 0
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix(),
        ):
            name = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--artifact-manifest", default="reports/RINENG_V8_ARTIFACT_MANIFEST.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    artifact_manifest_path = root / args.artifact_manifest
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("status") != "pass":
        raise SystemExit("Refusing to package a failing V8 artifact inventory")
    paths = []
    for row in artifact_manifest["artifacts"]:
        path = root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Artifact changed after inventory: {row['path']}")
        paths.append(path)
    for relative in (
        args.artifact_manifest,
        "reports/RINENG_V8_ARTIFACT_MANIFEST.csv",
    ):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    release_root = root / "release"
    final_dir = release_root / STEM
    final_zip = release_root / f"{STEM}.zip"
    staging = release_root / f".{STEM}_staging"
    partial_zip = release_root / f".{STEM}.zip.partial"
    for target in (final_dir, final_zip, staging, partial_zip):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite release target: {target}")
    release_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    for source in sorted(set(paths)):
        relative = source.relative_to(root)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (staging / "RELEASE_README.md").write_text(README, encoding="utf-8")

    rows = [
        {
            "path": path.relative_to(staging).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(staging).as_posix(),
        )
    ]
    release_manifest = {
        "release_version": VERSION,
        "status": "pass",
        "package_kind": "public-reproducibility-release-candidate",
        "artifact_count": len(rows),
        "artifacts": rows,
        "inference_rerun_required": False,
        "external_download_required_for_rescoring": False,
        "model_weights_redistributed": False,
        "answer_used_in_prediction_construction": False,
        "human_review_evidence_claimed": False,
        "public_doi_or_url_present": False,
        "project_code_license_selected": False,
        "claim_scope": (
            "candidate-tag retrieval under the declared PIDQA and public DEXPI "
            "drawing-family, model, and visual-budget conditions"
        ),
        "acquisition_by_reference": [
            "PIDQA source drawing collection",
            "DEXPI Public Example PIDs source repository",
            "PID2Graph/OPEN100 upstream archive",
            "Qwen3-VL and InternVL model weights",
            "PaddleOCR downloaded inference weights",
        ],
    }
    (staging / "RELEASE_MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(staging / "RELEASE_MANIFEST.csv", rows)
    member_count = deterministic_zip(staging, partial_zip)
    with zipfile.ZipFile(partial_zip) as archive:
        if len(archive.namelist()) != member_count:
            raise RuntimeError("Release archive member verification failed")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Release archive CRC failure: {bad}")
    shutil.move(str(staging), str(final_dir))
    shutil.move(str(partial_zip), str(final_zip))
    report = {
        "version": "rineng-public-release-v8",
        "status": "pass",
        "release_version": VERSION,
        "directory": final_dir.relative_to(root).as_posix(),
        "archive": final_zip.relative_to(root).as_posix(),
        "archive_bytes": final_zip.stat().st_size,
        "archive_sha256": sha256(final_zip),
        "member_count": member_count,
        "artifact_count": len(rows),
    }
    output = root / "reports/generated/rineng_public_release_manifest_v8.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(root / "reports/generated/rineng_public_release_manifest_v8.csv", rows)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
