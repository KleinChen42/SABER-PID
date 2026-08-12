"""Inventory and integrity-check the locally recovered overnight-v7 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def category(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("outputs/rineng_overnight_v7/"):
        return "raw_prediction" if path.suffix == ".jsonl" else "run_control"
    if value.startswith("reports/remote_logs/rineng_overnight_v7/"):
        return "remote_log"
    if value.startswith("paper/figures/"):
        return "paper_figure"
    if value.startswith("paper/tables/"):
        return "paper_table"
    if value.startswith("scripts/"):
        return "code"
    if value.startswith("data/manifests/"):
        return "frozen_plan"
    return "derived_result"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.json")
    parser.add_argument("--csv", default="reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    score_path = root / "reports/generated/rineng_overnight_v7_score.json"
    validation_path = root / "reports/generated/rineng_overnight_v7_validation.json"
    score = json_value(score_path)
    validation = json_value(validation_path)
    if score.get("status") != "pass" or validation.get("status") != "pass":
        raise SystemExit("Refusing to inventory an unvalidated V7 run")

    raw_paths = sorted((root / "outputs/rineng_overnight_v7").rglob("*.jsonl"))
    if len(raw_paths) != 54:
        raise SystemExit(f"Expected 54 raw JSONL files, found {len(raw_paths)}")
    raw_rows = sum(line_count(path) for path in raw_paths)
    if raw_rows != 16_560:
        raise SystemExit(f"Expected 16560 raw rows, found {raw_rows}")

    score_hashes = {
        str(cell["path"]): str(cell["sha256"])
        for cell in score["cells"].values()
    }
    for path in raw_paths:
        relative = path.relative_to(root).as_posix()
        if score_hashes.get(relative) != sha256(path):
            raise SystemExit(f"Recovered raw hash mismatch: {relative}")

    explicit_paths = [
        "21_RINENG_OVERNIGHT_HIGH_VALUE_GPU_EXECUTION_V7.md",
        "README.md",
        "data/manifests/rineng_overnight_v7_public_plan.json",
        "reports/generated/rineng_overnight_v7_preflight.json",
        "reports/generated/rineng_overnight_v7_score.json",
        "reports/generated/rineng_overnight_v7_score.csv",
        "reports/generated/rineng_overnight_v7_validation.json",
        "reports/generated/rineng_overnight_v7_counterfactual_table.csv",
        "reports/generated/rineng_overnight_v7_task_table.csv",
        "reports/generated/rineng_overnight_v7_prompt_sensitivity.csv",
        "reports/generated/rineng_overnight_v7_paper_summary.json",
        "reports/generated/pdf_render_validation_v7.json",
        "reports/generated/pdf_visual_inspection_v7.json",
        "reports/RINENG_OVERNIGHT_V7_CLOSEOUT.md",
        "paper/manuscript.tex",
        "paper/supplementary.tex",
        "paper/figure_manifest.md",
        "paper/figure_captions.md",
        "paper/highlights.md",
        "paper/cover_letter.md",
        "paper/data_availability.md",
        "output/pdf/v7/manuscript.pdf",
        "output/pdf/v7/supplementary.pdf",
        "paper/figures/figure_v7_cross_model_counterfactual_replication.pdf",
        "paper/figures/figure_v7_cross_model_counterfactual_replication.png",
        "paper/figures/figure_s_v7_prompt_sensitivity.pdf",
        "paper/figures/figure_s_v7_prompt_sensitivity.png",
        "paper/figures/figure_metadata_v7.json",
        "paper/tables/table_rineng_overnight_v7_counterfactual.tex",
        "paper/tables/table_rineng_overnight_v7_task_accuracy.tex",
        "scripts/prepare_rineng_overnight_v7.py",
        "scripts/preflight_rineng_overnight_v7.py",
        "scripts/run_qwen_counterfactual_prompt_matrix_v7.py",
        "scripts/run_internvl_counterfactual_prompt_matrix_v7.py",
        "scripts/score_rineng_overnight_v7.py",
        "scripts/validate_rineng_overnight_v7.py",
        "scripts/build_rineng_overnight_v7_paper_artifacts.py",
        "scripts/build_rineng_overnight_v7_artifact_manifest.py",
        "scripts/reproduce_rineng_overnight_v7.py",
        "scripts/build_pdf_render_validation_v4.py",
        "scripts/record_pdf_visual_inspection_v4.py",
        "scripts/run_remote_rineng_overnight_qwen_v7.sh",
        "scripts/run_remote_rineng_overnight_internvl_v7.sh",
        "scripts/run_remote_rineng_overnight_score_watch_v7.sh",
        "scripts/run_remote_rineng_overnight_score_v7.sh",
        "scripts/run_e1_evidence_audit.py",
        "connect_h200.ps1",
    ]
    paths = set(raw_paths)
    paths.update((root / "outputs/rineng_overnight_v7").rglob("run_summary.json"))
    paths.update((root / "outputs/rineng_overnight_v7").rglob("COMPLETE"))
    paths.update((root / "outputs/rineng_overnight_v7").rglob("FINISHED"))
    paths.update((root / "reports/remote_logs/rineng_overnight_v7").glob("*"))
    for relative in explicit_paths:
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Required artifact is missing: {relative}")
        paths.add(path)

    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "path": relative.as_posix(),
                "category": category(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": line_count(path) if path.suffix == ".jsonl" else "",
            }
        )

    report = {
        "version": "rineng-overnight-v7-artifact-manifest",
        "status": "pass",
        "remote_source_root": "/home/hera/pid_reliability_benchmark",
        "frozen_plan_sha256": score["plan_sha256"],
        "integrity": {
            "validation_status": validation["status"],
            "validation_error_count": validation["error_count"],
            "raw_file_count": len(raw_paths),
            "raw_row_count": raw_rows,
            "raw_bytes": sum(path.stat().st_size for path in raw_paths),
            "cell_count": validation["scope"]["cell_count"],
            "comparison_count": validation["scope"]["comparison_count"],
            "prompt_sensitivity_count": validation["scope"]["prompt_sensitivity_count"],
            "max_counterfactual_absolute_error": validation["numeric_agreement"][
                "max_counterfactual_absolute_error"
            ],
            "max_prompt_sensitivity_absolute_error": validation["numeric_agreement"][
                "max_prompt_sensitivity_absolute_error"
            ],
        },
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = root / args.csv
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_count": len(rows),
                "raw_file_count": len(raw_paths),
                "raw_row_count": raw_rows,
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
