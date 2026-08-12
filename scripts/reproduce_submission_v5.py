"""Rebuild the positive-narrative submission without rerunning inference."""

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
    args = parser.parse_args()
    root = Path(args.root).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "scripts"), env.get("PYTHONPATH", "")))
    env["SOURCE_DATE_EPOCH"] = "1786406400"
    env["TZ"] = "UTC"
    python = sys.executable
    steps: list[tuple[str, list[str]]] = [
        (
            "answer_isolation_audit",
            [python, "scripts/audit_evidence_input_isolation.py", "--root", str(root), "--output", "reports/generated/evidence_input_answer_isolation_audit_v2.json"],
        ),
        ("e7_mapping_control_rescore", [python, "scripts/score_positive_narrative_controls.py", "--root", str(root), "--experiment", "e7"]),
        ("e8_text_only_control_rescore", [python, "scripts/score_positive_narrative_controls.py", "--root", str(root), "--experiment", "e8"]),
        ("editorial_extension_rescore", [python, "scripts/score_editorial_extension_experiments_v4.py", "--root", str(root)]),
        ("editorial_analysis_rebuild", [python, "scripts/build_editorial_revision_analysis_v4.py", "--root", str(root)]),
        ("hybrid_analysis_rebuild", [python, "scripts/build_positive_narrative_hybrid_analysis_v5.py", "--root", str(root)]),
        ("frozen_fusion_validation_rescore", [python, "scripts/score_positive_narrative_fusion_validation_v5.py", "--root", str(root)]),
        ("supplementary_figure_rebuild", [python, "scripts/build_paper_figures_v4.py", "--root", str(root)]),
        ("positive_main_figure_rebuild", [python, "scripts/build_positive_narrative_figures_v5.py", "--root", str(root)]),
        ("manuscript_rebuild", [python, "scripts/build_editorial_revision_submission_v4.py", "--root", str(root)]),
    ]
    if not args.skip_pdf:
        tectonic = shutil.which("tectonic")
        if tectonic is None:
            raise FileNotFoundError("Tectonic is required unless --skip-pdf is supplied")
        pdf_dir = root / "output/pdf/v5"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for document in ("manuscript", "supplementary"):
            command = [tectonic, "--keep-logs", "--outdir", "output/pdf/v5", f"paper/{document}.tex"]
            steps.extend(((f"{document}_pdf_pass_1", command), (f"{document}_pdf_pass_2", command)))
        steps.append(
            (
                "pdf_render_validation",
                [
                    python,
                    "scripts/build_pdf_render_validation_v4.py",
                    "--root",
                    str(root),
                    "--pdf-dir",
                    "output/pdf/v5",
                    "--render-dir",
                    "tmp/pdfs/final_validation_v5",
                    "--output",
                    "reports/generated/pdf_render_validation_v5.json",
                    "--validation-version",
                    "pdf-render-v5",
                    "--visual-record",
                    "reports/generated/pdf_visual_inspection_v5.json",
                ],
            )
        )

    results = []
    for name, command in steps:
        result = run_step(name, command, root, env)
        results.append(result)
        if result["returncode"] != 0:
            break
    status = "pass" if len(results) == len(steps) and all(row["returncode"] == 0 for row in results) else "fail"
    report = {
        "validation_version": "positive-narrative-reproduction-v5",
        "status": status,
        "inference_rerun": False,
        "external_download_retried": False,
        "visual_inspection_overwritten": False,
        "steps": results,
    }
    output = root / "reports/generated/reproduction_validation_v5.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output)}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
