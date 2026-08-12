"""Validate v4 release archive membership, hashes, and evidence boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


VERSION = "pidqa-evidence-submission-v4"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_one(root: Path, kind: str) -> dict[str, Any]:
    relative = f"release/pidqa_evidence_submission_v4_{kind}.zip"
    path = root / relative
    failures: list[str] = []
    if not path.is_file():
        return {"package_kind": kind, "archive": relative, "status": "fail", "failures": ["archive_missing"]}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            failures.append("duplicate_members")
        if "RELEASE_MANIFEST.json" not in names or "RELEASE_MANIFEST.csv" not in names:
            failures.append("release_manifest_missing")
            manifest: dict[str, Any] = {}
        else:
            manifest = json.loads(archive.read("RELEASE_MANIFEST.json").decode("utf-8"))
        if manifest.get("release_version") != VERSION or manifest.get("package_kind") != kind:
            failures.append("manifest_identity_mismatch")
        artifacts = manifest.get("artifacts", [])
        expected = {str(row.get("path")): row for row in artifacts}
        observed_artifacts = set(names) - {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST.csv"}
        if set(expected) != observed_artifacts:
            failures.append("artifact_membership_mismatch")
        bad_hashes = []
        bad_sizes = []
        for name, row in expected.items():
            if name not in names:
                continue
            value = archive.read(name)
            if digest(value) != row.get("sha256"):
                bad_hashes.append(name)
            if len(value) != row.get("size_bytes"):
                bad_sizes.append(name)
        if bad_hashes:
            failures.append("artifact_hash_mismatch")
        if bad_sizes:
            failures.append("artifact_size_mismatch")
        forbidden = []
        for name in names:
            normalized = name.replace("\\", "/")
            lowered = normalized.lower()
            # A helper script whose filename mentions a model registry is
            # reproducibility metadata, not a bundled cache or model weight.
            # Restrict the directory check to an actual `modelscope/` path
            # component and keep the weight-extension check independent.
            has_modelscope_directory = "/modelscope/" in f"/{lowered.strip('/')}/"
            if (
                lowered.startswith("data/raw/")
                or "/.git/" in f"/{lowered}"
                or has_modelscope_directory
                or lowered.endswith((".safetensors", ".bin", ".ckpt"))
            ):
                forbidden.append(name)
        if forbidden:
            failures.append("forbidden_weight_or_raw_asset")
        required = {
            "paper/manuscript.tex",
            "paper/supplementary.tex",
            "output/pdf/manuscript.pdf",
            "output/pdf/supplementary.pdf",
            "reports/generated/submission_package_validation_v4.json",
            "reports/generated/pdf_visual_inspection_v4.json",
            "review/SABER_PID_model_based_editorial_review.md",
            "scripts/reproduce_submission_v4.py",
        }
        if required - set(names):
            failures.append("required_review_artifact_missing")
    return {
        "package_kind": kind,
        "archive": relative,
        "status": "fail" if failures else "pass",
        "failures": failures,
        "member_count": len(names),
        "artifact_count": len(artifacts),
        "bad_hashes": bad_hashes,
        "bad_sizes": bad_sizes,
        "forbidden_members": forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/generated/submission_archive_validation_v4.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    packages = [validate_one(root, kind) for kind in ("review", "full")]
    status = "pass" if all(row["status"] == "pass" for row in packages) else "fail"
    report = {"release_version": VERSION, "status": status, "packages": packages}
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output)}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
