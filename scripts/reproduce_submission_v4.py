"""Rebuild the frozen v4 submission evidence layer with one command.

This command is deliberately inference-free.  It rescales/rescores immutable
outputs, rebuilds deterministic analyses and figures, renders the manuscript
and supplement, and checks every PDF page mechanically.  The separately
recorded all-page visual inspection is not fabricated or overwritten here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/reproduction_validation_v4.json",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Rebuild analyses, figures, and TeX without invoking Tectonic.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root / "scripts"), env.get("PYTHONPATH", ""))
    )
    # Freeze PDF metadata so a repeated rebuild does not change only because
    # the command was executed on another date.
    env["SOURCE_DATE_EPOCH"] = "1786406400"
    env["TZ"] = "UTC"
    python = sys.executable
    steps: list[tuple[str, list[str]]] = [
        (
            "answer_isolation_audit",
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
        (
            "editorial_extension_rescore",
            [python, "scripts/score_editorial_extension_experiments_v4.py", "--root", str(root)],
        ),
        (
            "editorial_analysis_rebuild",
            [python, "scripts/build_editorial_revision_analysis_v4.py", "--root", str(root)],
        ),
        (
            "figure_rebuild",
            [python, "scripts/build_paper_figures_v4.py", "--root", str(root)],
        ),
        (
            "manuscript_rebuild",
            [python, "scripts/build_editorial_revision_submission_v4.py", "--root", str(root)],
        ),
    ]
    if not args.skip_pdf:
        tectonic = shutil.which("tectonic")
        if tectonic is None:
            raise FileNotFoundError("Tectonic is required unless --skip-pdf is supplied")
        for document in ("manuscript", "supplementary"):
            command = [
                tectonic,
                "--keep-logs",
                "--outdir",
                "output/pdf",
                f"paper/{document}.tex",
            ]
            steps.extend(
                (
                    (f"{document}_pdf_pass_1", command),
                    (f"{document}_pdf_pass_2", command),
                )
            )
        steps.append(
            (
                "pdf_render_validation",
                [python, "scripts/build_pdf_render_validation_v4.py", "--root", str(root)],
            )
        )
    # A release archive contains the already completed all-page visual record.
    # When that record is present, include the complete static package contract;
    # during an initial build it remains a deliberately separate final step.
    if (root / "reports/generated/pdf_visual_inspection_v4.json").is_file():
        steps.append(
            (
                "submission_package_validation",
                [python, "scripts/validate_submission_package_v4.py", "--root", str(root)],
            )
        )

    results: list[dict[str, Any]] = []
    for name, command in steps:
        result = run_step(name, command, root, env)
        results.append(result)
        if result["returncode"] != 0:
            break
    status = "pass" if len(results) == len(steps) and all(row["returncode"] == 0 for row in results) else "fail"
    report = {
        "validation_version": "submission-reproduction-v4",
        "status": status,
        "inference_rerun": False,
        "human_or_model_judging_used": False,
        "original_pidqa_tree_required": False,
        "pdf_rebuilt": not args.skip_pdf,
        "visual_inspection_overwritten": False,
        "package_validation_included": any(name == "submission_package_validation" for name, _ in steps),
        "steps": results,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "output": str(output)}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
