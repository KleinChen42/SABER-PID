"""Run deterministic v3 evidence and submission checks without rerunning inference."""

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
        "stdout_tail": completed.stdout[-2400:],
        "stderr_tail": completed.stderr[-2400:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/final_reproducibility_validation_v3.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    python = sys.executable
    steps = (
        (
            "answer_isolation_audit_e2_to_e8",
            [
                python,
                "scripts/audit_evidence_input_isolation.py",
                "--root",
                str(root),
                "--output",
                "reports/generated/evidence_input_answer_isolation_audit_v2.json",
            ],
        ),
        (
            "e7_mapping_control_rescore",
            [python, "scripts/score_positive_narrative_controls.py", "--root", str(root), "--experiment", "e7"],
        ),
        (
            "e8_text_only_control_rescore",
            [python, "scripts/score_positive_narrative_controls.py", "--root", str(root), "--experiment", "e8"],
        ),
        ("figure_rebuild", [python, "scripts/build_paper_figures_v3.py", "--root", str(root)]),
        (
            "manuscript_artifact_rebuild",
            [python, "scripts/build_positive_narrative_submission_v3.py", "--root", str(root)],
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
            [python, "scripts/validate_submission_package_v3.py", "--root", str(root)],
        ),
    )
    results = [run_step(name, command, root, environment) for name, command in steps]
    report = {
        "validation_version": "final-reproducibility-v3",
        "inference_rerun": False,
        "external_pid2graph_action": "not_restarted_or_extracted",
        "pytest_tmp_path_handling": "unit suite omits one tmp-path-dependent loader case and executes the equivalent fixture validator separately",
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
