"""Build the deterministic public reproducibility release for SABER-PID V10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


VERSION = "saber-pid-rineng-v10"
STEM = "saber_pid_rineng_v10_public_release"
ZIP_TIME = (2026, 8, 13, 0, 0, 0)
REPOSITORY = "https://github.com/KleinChen42/SABER-PID"

README = f"""# SABER-PID Results in Engineering V10 reproducibility release

This archive accompanies **SABER-PID: Source-Isolated Qualification and
Cost-Aware Operation of Vision--Language Models for P&ID Tag Retrieval**.

The release retains the complete source-isolated qualification evidence,
qualified-budget mild-quality matrix, closest-safe 54-tile InternVL transfer,
public DEXPI branch, OCR--VLM operating family, cost-sensitive decision grid,
immutable raw outputs, scorer-only references, deterministic scripts, tests,
final PDFs, and SHA-256 inventories. No model inference is required to rebuild
the reported analyses.

Run `python scripts/reproduce_rineng_submission_v10.py --root .` from the
archive root. Original SABER-PID code is released under the MIT License.
PIDQA and DEXPI retain their vendored CC0 and CC BY 4.0 terms; model weights
are not redistributed. Public repository: {REPOSITORY}
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
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
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
    parser.add_argument("--artifact-manifest", default="reports/RINENG_V10_ARTIFACT_MANIFEST.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    inventory = json.loads((root / args.artifact_manifest).read_text(encoding="utf-8"))
    if inventory.get("status") != "pass":
        raise SystemExit("Refusing to package a failing V10 artifact inventory")
    paths = []
    for row in inventory["artifacts"]:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"Artifact differs from V10 inventory: {row['path']}")
        paths.append(path)
    for relative in (args.artifact_manifest, "reports/RINENG_V10_ARTIFACT_MANIFEST.csv"):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    release_root = root / "release"
    final_dir = release_root / STEM
    final_zip = release_root / f"{STEM}.zip"
    staging = release_root / f".{STEM}_staging"
    partial_zip = release_root / f".{STEM}.zip.partial"
    if args.force:
        for target in (final_dir, final_zip, staging, partial_zip):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
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
    manifest = {
        "release_version": VERSION,
        "status": "pass",
        "package_kind": "public-reproducibility-release",
        "artifact_count": len(rows),
        "artifacts": rows,
        "inference_rerun_required": False,
        "external_download_required_for_rescoring": False,
        "model_weights_redistributed": False,
        "answer_used_in_prediction_construction": False,
        "human_review_evidence_claimed": False,
        "public_repository_url": REPOSITORY,
        "project_code_license": "MIT",
    }
    (staging / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(staging / "RELEASE_MANIFEST.csv", rows)
    member_count = deterministic_zip(staging, partial_zip)
    with zipfile.ZipFile(partial_zip) as archive:
        if len(archive.namelist()) != member_count or archive.testzip() is not None:
            raise RuntimeError("Release archive member or CRC verification failed")
    shutil.move(str(staging), str(final_dir))
    shutil.move(str(partial_zip), str(final_zip))
    report = {
        "version": "rineng-public-release-v10",
        "status": "pass",
        "release_version": VERSION,
        "directory": final_dir.relative_to(root).as_posix(),
        "archive": final_zip.relative_to(root).as_posix(),
        "archive_bytes": final_zip.stat().st_size,
        "archive_sha256": sha256(final_zip),
        "member_count": member_count,
        "artifact_count": len(rows),
    }
    output = root / "reports/generated/rineng_public_release_manifest_v10.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(root / "reports/generated/rineng_public_release_manifest_v10.csv", rows)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
