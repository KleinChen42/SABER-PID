"""Rebuild and validate the inference-free SABER-PID RINENG V10 package."""

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
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-submission-validation", action="store_true")
    parser.add_argument(
        "--output", default="reports/generated/reproduction_validation_v10.json"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    python = sys.executable
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root / "scripts"), environment.get("PYTHONPATH", ""))
    )
    environment["SOURCE_DATE_EPOCH"] = "1786492800"
    environment["TZ"] = "UTC"

    steps: list[tuple[str, list[str]]] = [
        (
            "v10_publication_figures",
            [python, "scripts/build_rineng_v10_publication_figures.py", "--root", str(root)],
        )
    ]
    if not args.skip_tests:
        steps.append(
            (
                "v10_relevant_tests",
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_build_rineng_v10_publication_figures.py",
                    "tests/test_build_rineng_v9_editorial_assets.py",
                    "tests/test_cost_sensitive_operating_modes_v8.py",
                    "tests/test_score_rineng_v8_extensions.py",
                    "tests/test_score_dexpi_external_v8.py",
                    "tests/test_validate_rineng_v8_extensions.py",
                    "tests/test_rineng_revision_analysis_v6.py",
                    "tests/test_ocr_geometry_join.py",
                ],
            )
        )

    if not args.skip_pdf:
        tectonic = shutil.which("tectonic")
        if tectonic is None:
            raise FileNotFoundError("Tectonic is required unless --skip-pdf is supplied")
        (root / "output/pdf/v10").mkdir(parents=True, exist_ok=True)
        for document in ("manuscript", "supplementary"):
            steps.append(
                (
                    f"{document}_pdf",
                    [
                        tectonic,
                        "--keep-logs",
                        "--outdir",
                        "output/pdf/v10",
                        f"paper/{document}.tex",
                    ],
                )
            )
        steps.append(
            (
                "pdf_render_validation_v10",
                [
                    python,
                    "scripts/build_pdf_render_validation_v4.py",
                    "--root",
                    str(root),
                    "--pdf-dir",
                    "output/pdf/v10",
                    "--render-dir",
                    "tmp/pdfs/final_validation_v10",
                    "--output",
                    "reports/generated/pdf_render_validation_v10.json",
                    "--validation-version",
                    "pdf-render-v10",
                    "--visual-record",
                    "reports/generated/pdf_visual_inspection_v10.json",
                ],
            )
        )

    if not args.skip_submission_validation:
        steps.append(
            (
                "rineng_submission_validation_v10",
                [python, "scripts/validate_rineng_submission_v10.py", "--root", str(root)],
            )
        )

    results = []
    for name, command in steps:
        result = run_step(name, command, root, environment)
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
        "version": "rineng-submission-reproduction-v10",
        "status": status,
        "mode": "inference-free",
        "source_date_epoch": environment["SOURCE_DATE_EPOCH"],
        "tests_skipped": args.skip_tests,
        "pdf_rebuild_skipped": args.skip_pdf,
        "steps": results,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": args.output}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
