"""Extract the RINENG v6 archive and run its full inference-free reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


STEM = "pidqa_rineng_submission_v6_public_release_candidate"
COMPARE_PATHS = (
    "reports/generated/evidence_input_answer_isolation_audit_v2.json",
    "reports/generated/rineng_revision_analysis_v6.json",
    "paper/figures/figure_metadata_v6.json",
    "output/pdf/v6/manuscript.pdf",
    "output/pdf/v6/supplementary.pdf",
    "reports/generated/rineng_submission_validation_v6.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(value: str, clean_root: Path) -> str:
    return (
        value.replace(str(clean_root), "<clean-root>")
        .replace(str(clean_root).replace("\\", "/"), "<clean-root>")
        .replace(sys.executable, "<python>")
    )


def safe_remove_generated_directory(root: Path, target: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    expected_parent = (resolved_root / "tmp").resolve()
    if resolved_target.parent != expected_parent or resolved_target.name != "rineng_archive_clean_v6":
        raise RuntimeError(f"Refusing to remove unexpected path: {resolved_target}")
    shutil.rmtree(resolved_target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/rineng_archive_clean_check_v6.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    archive_relative = f"release/{STEM}.zip"
    archive = root / archive_relative
    if not archive.is_file():
        raise FileNotFoundError(archive)

    clean_container = root / "tmp/rineng_archive_clean_v6"
    clean_root = clean_container / "package"
    if clean_container.exists():
        raise FileExistsError(
            f"Refusing to reuse clean-check directory: {clean_container}"
        )
    clean_root.mkdir(parents=True)

    failures: list[str] = []
    returncode = 1
    stdout_tail = ""
    stderr_tail = ""
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    reproduction: dict[str, Any] = {}
    validator_status = None
    try:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(clean_root)
        missing = [
            relative for relative in COMPARE_PATHS if not (clean_root / relative).is_file()
        ]
        if missing:
            failures.append("comparison_artifact_missing_before_run")
        else:
            before = {
                relative: sha256(clean_root / relative) for relative in COMPARE_PATHS
            }

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(clean_root / "src"),
                str(clean_root / "scripts"),
                env.get("PYTHONPATH", ""),
            )
        )
        completed = subprocess.run(
            [sys.executable, "scripts/reproduce_submission_v6.py", "--root", "."],
            cwd=clean_root,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        returncode = completed.returncode
        stdout_tail = sanitize(completed.stdout[-4000:], clean_root)
        stderr_tail = sanitize(completed.stderr[-4000:], clean_root)
        if returncode != 0:
            failures.append("clean_reproduction_failed")

        reproduction_path = (
            clean_root / "reports/generated/reproduction_validation_v6.json"
        )
        if reproduction_path.is_file():
            reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
        if reproduction.get("status") != "pass":
            failures.append("clean_reproduction_report_failed")

        missing_after = [
            relative for relative in COMPARE_PATHS if not (clean_root / relative).is_file()
        ]
        if missing_after:
            failures.append("comparison_artifact_missing_after_run")
        else:
            after = {
                relative: sha256(clean_root / relative) for relative in COMPARE_PATHS
            }
        mismatches = [
            relative
            for relative in COMPARE_PATHS
            if before.get(relative) != after.get(relative)
        ]
        if mismatches:
            failures.append("regenerated_artifact_hash_mismatch")

        validator_path = (
            clean_root / "reports/generated/rineng_submission_validation_v6.json"
        )
        validator_status = (
            json.loads(validator_path.read_text(encoding="utf-8")).get("status")
            if validator_path.is_file()
            else None
        )
        if validator_status != "pass":
            failures.append("clean_submission_validator_failed")
    finally:
        safe_remove_generated_directory(root, clean_container)

    report = {
        "validation_version": "rineng-archive-clean-check-v6",
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "archive": archive_relative,
        "archive_sha256": sha256(archive),
        "clean_reproduction_returncode": returncode,
        "clean_reproduction_status": reproduction.get("status"),
        "clean_submission_validator_status": validator_status,
        "inference_rerun": reproduction.get("inference_rerun"),
        "external_download_retried": reproduction.get("external_download_retried"),
        "visual_inspection_overwritten": reproduction.get(
            "visual_inspection_overwritten"
        ),
        "clean_steps": [
            {
                "name": step.get("name"),
                "returncode": step.get("returncode"),
                "stdout_tail": sanitize(str(step.get("stdout_tail", "")), clean_root),
                "stderr_tail": sanitize(str(step.get("stderr_tail", "")), clean_root),
            }
            for step in reproduction.get("steps", [])
        ],
        "compared_artifacts": [
            {
                "path": relative,
                "before_sha256": before.get(relative),
                "after_sha256": after.get(relative),
                "identical": before.get(relative) == after.get(relative),
            }
            for relative in COMPARE_PATHS
        ],
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
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
