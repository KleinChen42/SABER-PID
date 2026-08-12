"""Run the deterministic non-PDF checks required by the v2 submission package.

The script deliberately reuses frozen raw artifacts.  It does not rerun model
inference or touch the external PID2Graph branch.  A JSON report captures the
exit status and short terminal tail of every check so the final package has a
machine-readable record of the validation run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_step(name: str, command: list[str], root: Path, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/final_reproducibility_validation_v2.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    environment = os.environ.copy()
    source_path = str(root / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")

    python = sys.executable
    steps = [
        (
            "answer_isolation_audit",
            [python, "scripts/audit_evidence_input_isolation.py", "--root", str(root)],
        ),
        (
            "claim_freeze_rebuild",
            [python, "scripts/build_m0_evidence_freeze_v2.py", "--root", str(root)],
        ),
        (
            "figure_rebuild",
            [python, "scripts/build_paper_figures_v2.py", "--root", str(root)],
        ),
        (
            "pytest_unit_suite_without_tmp_path_fixture",
            [
                python,
                "-m",
                "pytest",
                "tests",
                "-q",
                "-p",
                "no:cacheprovider",
                "-k",
                "not test_load_pidqa_preserves_source_identity",
            ],
        ),
        (
            "pidqa_loader_fixture_validation",
            [python, "scripts/verify_pidqa_loader_fixture_v2.py", "--root", str(root)],
        ),
        (
            "submission_static_validation",
            [python, "scripts/validate_submission_package_v2.py", "--root", str(root)],
        ),
    ]
    results = [run_step(name, command, root, environment) for name, command in steps]
    report = {
        "validation_version": "final-reproducibility-v2",
        "inference_rerun": False,
        "external_pid2graph_action": "not_restarted",
        "pytest_tmp_path_handling": "16 pytest cases plus equivalent synthetic and real loader validation",
        "steps": results,
        "status": "pass" if all(item["returncode"] == 0 for item in results) else "fail",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
