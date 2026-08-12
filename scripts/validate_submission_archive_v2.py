"""Verify every archived release member against the SHA-256 release manifest."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--manifest",
        default="reports/generated/submission_release_manifest_v2.json",
    )
    parser.add_argument(
        "--output",
        default="reports/generated/submission_archive_content_validation_v2.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    archive = root / manifest["archive"]["archive"]
    expected_rows = manifest["artifact_rows"]
    expected_names = {row["package_path"] for row in expected_rows}
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as handle:
        members = set(handle.namelist())
        for row in expected_rows:
            name = row["package_path"]
            if name not in members:
                missing.append(name)
                continue
            with handle.open(name) as member:
                observed = digest_stream(member)
            if observed != row["sha256"]:
                mismatches.append({"path": name, "expected": row["sha256"], "observed": observed})
        manifest_name = "RELEASE_MANIFEST.json"
        manifest_sha = ""
        if manifest_name in members:
            with handle.open(manifest_name) as member:
                manifest_sha = digest_stream(member)
        unexpected_payload = sorted((members - expected_names) - {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST.csv"})
    report = {
        "validation_version": "submission-archive-content-v2",
        "archive": manifest["archive"]["archive"],
        "expected_artifact_count": len(expected_rows),
        "missing_members": missing,
        "content_hash_mismatches": mismatches,
        "package_manifest_sha256_expected": manifest["archive"]["package_manifest_sha256"],
        "package_manifest_sha256_observed": manifest_sha,
        "unexpected_payload_members": unexpected_payload,
        "status": "pass"
        if not missing
        and not mismatches
        and not unexpected_payload
        and manifest_sha == manifest["archive"]["package_manifest_sha256"]
        else "fail",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
