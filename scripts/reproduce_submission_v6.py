"""Rebuild and validate the inference-free Results in Engineering v6 package.

The command re-scores immutable artifacts, rebuilds deterministic analyses and
figures, runs tests, compiles both PDFs with frozen metadata, renders every
page, and validates the current submission. It never reruns model inference,
retries external downloads, or overwrites the separately recorded visual
inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from reproduce_submission_v4 import run_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument(
        "--output",
        default="reports/generated/reproduction_validation_v6.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root / "scripts"), env.get("PYTHONPATH", ""))
    )
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
            "rineng_revision_analysis_v6",
            [
                python,
                "scripts/build_rineng_revision_analysis_v6.py",
                "--root",
                str(root),
            ],
        ),
        (
            "rineng_analysis_unit_tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_rineng_revision_analysis_v6.py",
                "tests/test_ocr_geometry_join.py",
                "tests/test_positive_narrative_fusion.py",
            ],
        ),
        (
            "full_unit_test_suite",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
        (
            "supplementary_boundary_figure_rebuild",
            [python, "scripts/build_paper_figures_v4.py", "--root", str(root)],
        ),
        (
            "rineng_revision_figures_v6",
            [
                python,
                "scripts/build_rineng_revision_figures_v6.py",
                "--root",
                str(root),
            ],
        ),
    ]

    if not args.skip_pdf:
        tectonic = shutil.which("tectonic")
        if tectonic is None:
            raise FileNotFoundError("Tectonic is required unless --skip-pdf is supplied")
        (root / "output/pdf/v6").mkdir(parents=True, exist_ok=True)
        for document in ("manuscript", "supplementary"):
            command = [
                tectonic,
                "--keep-logs",
                "--outdir",
                "output/pdf/v6",
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
                "pdf_render_validation_v6",
                [
                    python,
                    "scripts/build_pdf_render_validation_v4.py",
                    "--root",
                    str(root),
                    "--pdf-dir",
                    "output/pdf/v6",
                    "--render-dir",
                    "tmp/pdfs/final_validation_v6",
                    "--output",
                    "reports/generated/pdf_render_validation_v6.json",
                    "--validation-version",
                    "pdf-render-v6",
                    "--visual-record",
                    "reports/generated/pdf_visual_inspection_v6.json",
                ],
            )
        )
        steps.append(
            (
                "rineng_submission_validation_v6",
                [
                    python,
                    "scripts/validate_rineng_submission_v6.py",
                    "--root",
                    str(root),
                ],
            )
        )

    results = []
    for name, command in steps:
        result = run_step(name, command, root, env)
        results.append(result)
        if result["returncode"] != 0:
            break

    status = (
        "pass"
        if len(results) == len(steps)
        and all(result["returncode"] == 0 for result in results)
        else "fail"
    )
    report = {
        "validation_version": "rineng-submission-reproduction-v6",
        "status": status,
        "source_date_epoch": env["SOURCE_DATE_EPOCH"],
        "inference_rerun": False,
        "external_download_retried": False,
        "visual_inspection_overwritten": False,
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
