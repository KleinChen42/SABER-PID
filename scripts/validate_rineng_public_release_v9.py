"""Validate the deterministic SABER-PID V9 public release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from build_rineng_public_release_v9 import STEM, VERSION, ZIP_TIME


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="reports/generated/rineng_public_release_validation_v9.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    archive_path = root / "release" / f"{STEM}.zip"
    failures: list[str] = []
    bad_hashes: list[str] = []
    bad_sizes: list[str] = []
    forbidden: list[str] = []
    local_paths: list[str] = []
    bad_timestamps: list[str] = []
    names: list[str] = []
    manifest: dict[str, Any] = {}

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
                or manifest.get("inference_rerun_required") is not False
                or manifest.get("answer_used_in_prediction_construction") is not False
                or manifest.get("human_review_evidence_claimed") is not False
                or manifest.get("model_weights_redistributed") is not False
            ):
                failures.append("release_contract_mismatch")

            expected = {str(row["path"]): row for row in manifest.get("artifacts", [])}
            observed = set(names) - {"RELEASE_MANIFEST.json", "RELEASE_MANIFEST.csv"}
            if set(expected) != observed:
                failures.append("artifact_membership_mismatch")
            for name, row in expected.items():
                if name not in names:
                    continue
                value = archive.read(name)
                if digest(value) != row["sha256"]:
                    bad_hashes.append(name)
                if len(value) != int(row["size_bytes"]):
                    bad_sizes.append(name)
            if bad_hashes:
                failures.append("artifact_hash_mismatch")
            if bad_sizes:
                failures.append("artifact_size_mismatch")

            forbidden_fragments = (
                "gpt_pro",
                "model_review",
                "review_output",
                "positive_narrative_self_review",
            )
            for name in names:
                lowered = name.replace("\\", "/").lower().strip("/")
                if (
                    lowered.startswith((".git/", "review/", "tmp/"))
                    or any(fragment in lowered for fragment in forbidden_fragments)
                    or lowered.endswith((".safetensors", ".ckpt", ".pth", ".pt"))
                ):
                    forbidden.append(name)
            if forbidden:
                failures.append("forbidden_private_or_weight_member")

            slash = b"\\"
            patterns = (
                re.compile(re.escape(b"C:" + slash + b"Users" + slash), re.I),
                re.compile(re.escape(b"E:" + slash + b"CODE" + slash), re.I),
                re.compile(re.escape(b"E:" + b"/" + b"CODE" + b"/"), re.I),
            )
            text_suffixes = (".json", ".csv", ".md", ".tex", ".py", ".txt", ".cff")
            for name in names:
                if name.lower().endswith(text_suffixes):
                    value = archive.read(name)
                    if any(pattern.search(value) for pattern in patterns):
                        local_paths.append(name)
            if local_paths:
                failures.append("local_windows_path_leak")
            bad_timestamps = [
                info.filename for info in archive.infolist() if info.date_time != ZIP_TIME
            ]
            if bad_timestamps:
                failures.append("nondeterministic_timestamp")

            required = {
                "RELEASE_README.md",
                "licenses/PIDQA_LICENSE.txt",
                "licenses/DEXPI_TRAINING_TEST_CASES_LICENSE.txt",
                "paper/manuscript.tex",
                "paper/supplementary.tex",
                "paper/figures/figure_1_saber_pid_overview_v9.pdf",
                "paper/figures/figure_4_cost_aware_operation_v9.pdf",
                "output/pdf/v9/manuscript.pdf",
                "output/pdf/v9/supplementary.pdf",
                "reports/generated/rineng_submission_validation_v9.json",
                "reports/generated/pdf_render_validation_v9.json",
                "reports/generated/pdf_visual_inspection_v9.json",
                "reports/RINENG_V9_ARTIFACT_MANIFEST.json",
                "scripts/reproduce_rineng_submission_v9.py",
                "scripts/validate_rineng_submission_v9.py",
                "data/answer_store/rineng_v8_dexpi_external_hidden.jsonl",
                "outputs/rineng_v8/dexpi_external_ocr.jsonl",
            }
            if required - set(names):
                failures.append("required_member_missing")
            quality_count = sum(
                name.startswith("outputs/rineng_v8/qwen3vl8b_quality/")
                and name.endswith(".jsonl")
                for name in names
            )
            internvl_count = sum(
                name.startswith("outputs/rineng_v8/internvl35_8b_budget54/")
                and name.endswith(".jsonl")
                for name in names
            )
            dexpi_count = sum(
                name.startswith("outputs/rineng_v8/dexpi_external_qwen/")
                and name.endswith(".jsonl")
                for name in names
            )
            v7_count = sum(
                name.startswith("outputs/rineng_overnight_v7/")
                and name.endswith(".jsonl")
                for name in names
            )
            if (quality_count, internvl_count, dexpi_count, v7_count) != (24, 9, 3, 54):
                failures.append("raw_cell_scope_mismatch")
            for report_name in (
                "reports/generated/rineng_submission_validation_v9.json",
                "reports/generated/pdf_render_validation_v9.json",
                "reports/generated/pdf_visual_inspection_v9.json",
                "reports/generated/reproduction_validation_v9.json",
            ):
                if report_name in names:
                    report = json.loads(archive.read(report_name).decode("utf-8"))
                    if report.get("status") != "pass":
                        failures.append("embedded_validation_not_pass")

    root_report_path = root / "reports/generated/rineng_public_release_manifest_v9.json"
    root_report = (
        json.loads(root_report_path.read_text(encoding="utf-8"))
        if root_report_path.is_file()
        else {}
    )
    archive_hash = digest(archive_path.read_bytes()) if archive_path.is_file() else None
    if root_report.get("status") != "pass" or root_report.get("archive_sha256") != archive_hash:
        failures.append("root_release_manifest_mismatch")
    failures = list(dict.fromkeys(failures))
    report = {
        "version": "rineng-public-release-validation-v9",
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "archive": f"release/{STEM}.zip",
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "member_count": len(names),
        "bad_hashes": bad_hashes,
        "bad_sizes": bad_sizes,
        "forbidden_members": forbidden,
        "local_path_leaks": local_paths,
        "bad_timestamps": bad_timestamps,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
