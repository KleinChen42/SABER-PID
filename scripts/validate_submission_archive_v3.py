"""Verify every v3 archive member against its SHA-256 release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def digest_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def validate_one(root: Path, package: dict[str, Any]) -> dict[str, Any]:
    archive_record = package["archive"]
    archive = root / archive_record["archive"]
    rows = package["artifact_rows"]
    expected_names = {row["package_path"] for row in rows}
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    manifest_sha = ""
    with zipfile.ZipFile(archive) as handle:
        members = set(handle.namelist())
        for row in rows:
            name = row["package_path"]
            if name not in members:
                missing.append(name)
                continue
            with handle.open(name) as member:
                observed = digest_stream(member)
            if observed != row["sha256"]:
                mismatches.append({"path": name, "expected": row["sha256"], "observed": observed})
        if "RELEASE_MANIFEST.json" in members:
            with handle.open("RELEASE_MANIFEST.json") as member:
                manifest_sha = digest_stream(member)
        unexpected = sorted((members - expected_names) - {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST.csv"})
    status = (
        "pass"
        if not missing
        and not mismatches
        and not unexpected
        and manifest_sha == archive_record["package_manifest_sha256"]
        else "fail"
    )
    return {
        "package_kind": package["package_kind"],
        "archive": archive_record["archive"],
        "expected_artifact_count": len(rows),
        "missing_members": missing,
        "content_hash_mismatches": mismatches,
        "package_manifest_sha256_expected": archive_record["package_manifest_sha256"],
        "package_manifest_sha256_observed": manifest_sha,
        "unexpected_payload_members": unexpected,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--manifest",
        default="reports/generated/submission_release_manifest_v3.json",
    )
    parser.add_argument(
        "--output",
        default="reports/generated/submission_archive_content_validation_v3.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    results = [validate_one(root, package) for package in manifest["packages"]]
    report = {
        "validation_version": "submission-archive-content-v3",
        "release_version": manifest["release_version"],
        "packages": results,
        "status": "pass" if results and all(item["status"] == "pass" for item in results) else "fail",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "packages": len(results)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
