"""Validate the deterministic SABER-PID V10 public release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from build_rineng_public_release_v10 import REPOSITORY, STEM, VERSION, ZIP_TIME


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/generated/rineng_public_release_validation_v10.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    archive_path = root / "release" / f"{STEM}.zip"
    failures: list[str] = []
    details: dict[str, object] = {}
    names: list[str] = []
    manifest = {}
    if not archive_path.is_file():
        failures.append("archive_missing")
    else:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                failures.append("duplicate_members")
            if archive.testzip() is not None:
                failures.append("crc_failure")
            if "RELEASE_MANIFEST.json" not in names:
                failures.append("release_manifest_missing")
            else:
                manifest = json.loads(archive.read("RELEASE_MANIFEST.json").decode("utf-8"))
            if (
                manifest.get("release_version") != VERSION
                or manifest.get("status") != "pass"
                or manifest.get("project_code_license") != "MIT"
                or manifest.get("public_repository_url") != REPOSITORY
                or manifest.get("model_weights_redistributed") is not False
            ):
                failures.append("release_contract_mismatch")
            expected = {str(row["path"]): row for row in manifest.get("artifacts", [])}
            observed = set(names) - {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST.csv"}
            if set(expected) != observed:
                failures.append("artifact_membership_mismatch")
            bad_hashes = []
            for name, row in expected.items():
                if name not in names:
                    continue
                data = archive.read(name)
                if digest(data) != row["sha256"] or len(data) != int(row["size_bytes"]):
                    bad_hashes.append(name)
            if bad_hashes:
                failures.append("artifact_hash_or_size_mismatch")
            details["bad_hashes"] = bad_hashes
            forbidden = []
            for name in names:
                lowered = name.replace("\\", "/").lower().strip("/")
                if (
                    lowered.startswith((".git/", "review/", "tmp/"))
                    or any(fragment in lowered for fragment in ("gpt_pro", "model_review", "review_output"))
                    or lowered.endswith((".safetensors", ".ckpt", ".pth", ".pt"))
                ):
                    forbidden.append(name)
            if forbidden:
                failures.append("forbidden_private_or_weight_member")
            details["forbidden_members"] = forbidden
            local_paths = []
            slash = b"\\"
            system_drive = bytes((67, 58))
            workspace_drive = bytes((69, 58))
            patterns = (
                re.compile(re.escape(system_drive + slash + b"Users" + slash), re.I),
                re.compile(re.escape(workspace_drive + slash + b"CODE" + slash), re.I),
                re.compile(re.escape(workspace_drive + b"/CODE/"), re.I),
            )
            for name in names:
                if name.lower().endswith((".json", ".csv", ".md", ".tex", ".py", ".txt", ".cff", ".toml")):
                    data = archive.read(name)
                    if any(pattern.search(data) for pattern in patterns):
                        local_paths.append(name)
            if local_paths:
                failures.append("local_windows_path_leak")
            details["local_path_leaks"] = local_paths
            bad_timestamps = [info.filename for info in archive.infolist() if info.date_time != ZIP_TIME]
            if bad_timestamps:
                failures.append("nondeterministic_timestamp")
            details["bad_timestamps"] = bad_timestamps
            required = {
                "RELEASE_README.md",
                "LICENSE",
                "CITATION.cff",
                "licenses/PIDQA_LICENSE.txt",
                "licenses/DEXPI_TRAINING_TEST_CASES_LICENSE.txt",
                "paper/manuscript.tex",
                "paper/supplementary.tex",
                "paper/figures/figure_1_saber_pid_overview_v10.pdf",
                "paper/figures/figure_4_cost_aware_operation_v10.pdf",
                "output/pdf/v10/manuscript.pdf",
                "output/pdf/v10/supplementary.pdf",
                "reports/generated/rineng_submission_validation_v10.json",
                "reports/generated/pdf_render_validation_v10.json",
                "reports/generated/pdf_visual_inspection_v10.json",
                "reports/RINENG_V10_ARTIFACT_MANIFEST.json",
                "scripts/reproduce_rineng_submission_v10.py",
                "scripts/validate_rineng_submission_v10.py",
                "data/answer_store/rineng_v8_dexpi_external_hidden.jsonl",
                "outputs/rineng_v8/dexpi_external_ocr.jsonl",
            }
            if required - set(names):
                failures.append("required_member_missing")
            counts = {
                "quality": sum(name.startswith("outputs/rineng_v8/qwen3vl8b_quality/") and name.endswith(".jsonl") for name in names),
                "internvl": sum(name.startswith("outputs/rineng_v8/internvl35_8b_budget54/") and name.endswith(".jsonl") for name in names),
                "dexpi": sum(name.startswith("outputs/rineng_v8/dexpi_external_qwen/") and name.endswith(".jsonl") for name in names),
                "v7": sum(name.startswith("outputs/rineng_overnight_v7/") and name.endswith(".jsonl") for name in names),
            }
            if tuple(counts.values()) != (24, 9, 3, 54):
                failures.append("raw_cell_scope_mismatch")
            details["raw_cell_counts"] = counts
    root_report_path = root / "reports/generated/rineng_public_release_manifest_v10.json"
    root_report = json.loads(root_report_path.read_text(encoding="utf-8")) if root_report_path.is_file() else {}
    archive_hash = digest(archive_path.read_bytes()) if archive_path.is_file() else None
    if root_report.get("status") != "pass" or root_report.get("archive_sha256") != archive_hash:
        failures.append("root_release_manifest_mismatch")
    failures = list(dict.fromkeys(failures))
    report = {
        "version": "rineng-public-release-validation-v10",
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "archive": f"release/{STEM}.zip",
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "member_count": len(names),
        **details,
    }
    output = root / args.output
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
