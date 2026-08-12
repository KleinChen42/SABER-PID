"""Validate the RINENG v6 public release candidate archive and boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any


VERSION = "pidqa-rineng-qualification-submission-v6"
KIND = "public-release-candidate"
STEM = "pidqa_rineng_submission_v6_public_release_candidate"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/rineng_public_archive_validation_v6.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    archive_relative = f"release/{STEM}.zip"
    archive_path = root / archive_relative
    failures: list[str] = []
    if not archive_path.is_file():
        failures.append("archive_missing")
        names: list[str] = []
        manifest: dict[str, Any] = {}
        artifacts: list[dict[str, Any]] = []
        bad_hashes: list[str] = []
        bad_sizes: list[str] = []
        forbidden: list[str] = []
        local_path_leaks: list[str] = []
        bad_timestamps: list[str] = []
    else:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                failures.append("duplicate_members")
            if "RELEASE_MANIFEST.json" not in names or "RELEASE_MANIFEST.csv" not in names:
                failures.append("release_manifest_missing")
                manifest = {}
            else:
                manifest = json.loads(
                    archive.read("RELEASE_MANIFEST.json").decode("utf-8")
                )
            if (
                manifest.get("release_version") != VERSION
                or manifest.get("package_kind") != KIND
                or manifest.get("status") != "pass"
            ):
                failures.append("manifest_identity_mismatch")
            if (
                manifest.get("technical_release_candidate_complete") is not True
                or manifest.get("public_doi_or_url_present") is not False
                or manifest.get("project_code_license_selected") is not False
                or manifest.get("reference_used_in_prediction_construction") is not False
                or manifest.get("human_review_evidence_claimed") is not False
            ):
                failures.append("release_boundary_contract_mismatch")

            artifacts = manifest.get("artifacts", [])
            expected = {str(row.get("path")): row for row in artifacts}
            observed = set(names) - {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST.csv"}
            if set(expected) != observed:
                failures.append("artifact_membership_mismatch")
            if manifest.get("artifact_count") != len(artifacts):
                failures.append("artifact_count_mismatch")

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
            forbidden_name_fragments = (
                "gpt_pro",
                "model_review",
                "review_output",
                "rineng_model_review",
            )
            for name in names:
                normalized = name.replace("\\", "/").strip("/")
                lowered = normalized.lower()
                if (
                    lowered.startswith(("data/raw/", "review/", "tmp/", ".git/"))
                    or "/.git/" in f"/{lowered}/"
                    or any(fragment in lowered for fragment in forbidden_name_fragments)
                    or lowered.endswith((".safetensors", ".ckpt", ".pth", ".pt"))
                ):
                    forbidden.append(name)
            if forbidden:
                failures.append("forbidden_private_raw_or_weight_member")

            local_path_leaks = []
            # Build the sentinels from separated byte fragments so this
            # validator does not flag its own source as a leaked local path.
            drive_c = b"C:"
            drive_e = b"E:"
            users_segment = b"Users"
            code_segment = b"CODE"
            local_patterns = (
                re.compile(
                    re.escape(drive_c + b"\\" + users_segment + b"\\"), re.I
                ),
                re.compile(
                    re.escape(drive_e + b"\\" + code_segment + b"\\"), re.I
                ),
                re.compile(re.escape(drive_e + b"/" + code_segment + b"/"), re.I),
            )
            text_extensions = (
                ".json",
                ".csv",
                ".md",
                ".tex",
                ".py",
                ".txt",
                ".cff",
                ".toml",
            )
            for name in names:
                if name.lower().endswith(text_extensions):
                    value = archive.read(name)
                    if any(pattern.search(value) for pattern in local_patterns):
                        local_path_leaks.append(name)
            if local_path_leaks:
                failures.append("local_absolute_path_leak")

            bad_timestamps = [
                info.filename
                for info in archive.infolist()
                if info.date_time != (2026, 8, 11, 0, 0, 0)
            ]
            if bad_timestamps:
                failures.append("nondeterministic_zip_timestamp")

            required = {
                "RELEASE_README.md",
                "LICENSES.md",
                "CITATION.cff",
                "licenses/PIDQA_LICENSE.txt",
                "paper/manuscript.tex",
                "paper/supplementary.tex",
                "output/pdf/v6/manuscript.pdf",
                "output/pdf/v6/supplementary.pdf",
                "reports/generated/rineng_revision_analysis_v6.json",
                "reports/generated/rineng_submission_validation_v6.json",
                "reports/generated/pdf_visual_inspection_v6.json",
                "scripts/reproduce_submission_v6.py",
                "scripts/validate_rineng_submission_v6.py",
                "tests/test_rineng_revision_analysis_v6.py",
                "data/processed/pidqa_records.jsonl",
                "outputs/final_replication/qwen8_b_p0_3072.jsonl",
            }
            missing_required = sorted(required - set(names))
            if missing_required:
                failures.append("required_reproducibility_member_missing")

    root_manifest_path = root / "reports/generated/rineng_public_release_manifest_v6.json"
    root_manifest = (
        json.loads(root_manifest_path.read_text(encoding="utf-8"))
        if root_manifest_path.is_file()
        else {}
    )
    archive_hash = digest(archive_path.read_bytes()) if archive_path.is_file() else None
    if (
        root_manifest.get("status") != "pass"
        or root_manifest.get("archive") != archive_relative
        or root_manifest.get("archive_sha256") != archive_hash
    ):
        failures.append("root_release_manifest_mismatch")

    report = {
        "validation_version": "rineng-public-archive-v6",
        "release_version": VERSION,
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "archive": archive_relative,
        "archive_sha256": archive_hash,
        "archive_size_bytes": archive_path.stat().st_size
        if archive_path.is_file()
        else None,
        "member_count": len(names),
        "artifact_count": len(artifacts),
        "bad_hashes": bad_hashes,
        "bad_sizes": bad_sizes,
        "forbidden_members": forbidden,
        "local_path_leaks": local_path_leaks,
        "bad_timestamps": bad_timestamps,
        "technical_release_candidate_complete": not failures,
        "external_boundary": {
            "public_doi_or_url": "submitter upload required",
            "project_code_license": "submitter decision required",
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
