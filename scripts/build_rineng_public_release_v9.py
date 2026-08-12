"""Build the deterministic public reproducibility release for SABER-PID V9."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


VERSION = "saber-pid-rineng-v9"
STEM = "saber_pid_rineng_v9_public_release"
ZIP_TIME = (2026, 8, 12, 0, 0, 0)

README = """# SABER-PID Results in Engineering V9 reproducibility release

This archive accompanies **SABER-PID: Source-Isolated Qualification and
Cost-Aware Operation of Vision--Language Models for P&ID Tag Retrieval**.

## Evidence retained

- source-isolated PIDQA qualification with correct, shuffled, and no-image controls;
- the complete 54-cell native-budget matrix, including low InternVL cells;
- paired quality robustness at the qualified Qwen 3072/512 setting;
- closest-safe 54-tile InternVL transfer and a public DEXPI drawing family;
- complete OCR--VLM modes, overlap sensitivity, candidate workload, and the
  cost-sensitive decision grid;
- raw immutable outputs, scorer-only references, deterministic figures and
  tables, all-page PDF validation, tests, and SHA-256 inventories.

## Inference-free reproduction

From the archive root, install the declared analysis dependencies and run:

    python scripts/reproduce_rineng_submission_v9.py --root .

The command rescans immutable predictions, recomputes grouped estimates and
10,000-replicate intervals, rebuilds deterministic editorial assets, runs
tests, compiles both PDFs, renders every page, and validates the submission.
It performs no model inference and no external download.

## Scope

Model-facing manifests contain no answer or Cypher field. Scorer-only
references are separated. PIDQA is covered by the vendored CC0 record; DEXPI
is frozen to official commit `a23d61e2e089eb2ca464cd552f9ae580a2785963`
with its CC BY 4.0 license. Drawing collections and model weights are acquired
by reference and are not redistributed. The release supports candidate-tag
retrieval under the declared drawing families and budgets; it does not qualify
topology reasoning or field deployment.

The archive DOI/URL, author identities, affiliations, declarations, and
project-code license choice remain submitter-owned fields.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
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
            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--artifact-manifest", default="reports/RINENG_V9_ARTIFACT_MANIFEST.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    inventory_path = root / args.artifact_manifest
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "pass":
        raise SystemExit("Refusing to package a failing V9 artifact inventory")

    paths = []
    for row in inventory["artifacts"]:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"Artifact differs from V9 inventory: {row['path']}")
        paths.append(path)
    for relative in (args.artifact_manifest, "reports/RINENG_V9_ARTIFACT_MANIFEST.csv"):
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
        destination = staging / source.relative_to(root)
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
            "drawing-family, model, quality, and visual-budget conditions"
        ),
    }
    (staging / "RELEASE_MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(staging / "RELEASE_MANIFEST.csv", rows)
    member_count = deterministic_zip(staging, partial_zip)
    with zipfile.ZipFile(partial_zip) as archive:
        if len(archive.namelist()) != member_count or archive.testzip() is not None:
            raise RuntimeError("Release archive member or CRC verification failed")
    shutil.move(str(staging), str(final_dir))
    shutil.move(str(partial_zip), str(final_zip))
    report = {
        "version": "rineng-public-release-v9",
        "status": "pass",
        "release_version": VERSION,
        "directory": final_dir.relative_to(root).as_posix(),
        "archive": final_zip.relative_to(root).as_posix(),
        "archive_bytes": final_zip.stat().st_size,
        "archive_sha256": sha256(final_zip),
        "member_count": member_count,
        "artifact_count": len(rows),
    }
    output = root / "reports/generated/rineng_public_release_manifest_v9.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(root / "reports/generated/rineng_public_release_manifest_v9.csv", rows)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
