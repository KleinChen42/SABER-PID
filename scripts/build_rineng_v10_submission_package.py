"""Build deterministic clean source and editorial packages for RINENG V10."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ZIP_TIME = (2026, 8, 13, 0, 0, 0)
STEM = "saber_pid_rineng_v10_submission_source"

MAIN_FIGURES = (
    "figure_1_saber_pid_overview_v10.pdf",
    "figure_2_quality_and_budget_v10.pdf",
    "figure_3_dexpi_external_v10.pdf",
    "figure_4_cost_aware_operation_v10.pdf",
)
SUPPLEMENT_FIGURES = (
    "figure_s1_boundary_controls_v10.pdf",
    "figure_s2_qualification_effects_v10.pdf",
    "figure_s3_operating_modes_v10.pdf",
    "figure_s4_cross_model_replication_v10.pdf",
    "figure_s5_prompt_sensitivity_v10.pdf",
)
TABLES = (
    "table_rineng_v9_qualification_scorecard.tex",
    "table_rineng_v9_operating_modes.tex",
    "table_rineng_overnight_v7_counterfactual.tex",
    "table_rineng_overnight_v7_task_accuracy.tex",
    "table_rineng_v8_quality_by_subset.tex",
    "table_rineng_v8_internvl_budget54.tex",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compresslevel=9)


def source_files() -> list[str]:
    return [
        "paper/manuscript.tex",
        "paper/supplementary.tex",
        *(f"paper/figures/{name}" for name in (*MAIN_FIGURES, *SUPPLEMENT_FIGURES)),
        *(f"paper/tables/{name}" for name in TABLES),
    ]


def editorial_files() -> list[str]:
    return [
        "paper/title_page.md",
        "paper/highlights.md",
        "paper/cover_letter.md",
        "paper/declarations.md",
        "paper/data_availability.md",
        "paper/Declaration_of_Interests.docx",
        "output/pdf/v10/manuscript.pdf",
        "output/pdf/v10/supplementary.pdf",
        "CITATION.cff",
        "LICENSE",
        "LICENSES.md",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=f"release/{STEM}.zip")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    files = source_files() + editorial_files()
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing submission files: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    payloads = {}
    for name in sorted(files):
        data = (root / name).read_bytes()
        payloads[name] = data
        rows.append({"path": name, "size_bytes": len(data), "sha256": digest(data)})
    readme = (
        "# SABER-PID RINENG V10 submission package\n\n"
        "`paper/manuscript.tex` and `paper/supplementary.tex` are the editable "
        "sources. Only the nine active V10 PDF figures and six referenced TeX "
        "tables are included. Editorial files and final reference PDFs are "
        "provided separately in their original paths.\n"
    ).encode("utf-8")
    manifest = {
        "version": "saber-pid-rineng-submission-v10",
        "status": "pass",
        "file_count": len(rows),
        "editable_source_count": 2,
        "active_figure_count": 9,
        "active_table_count": 6,
        "files": rows,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        add_bytes(archive, "PACKAGE_README.md", readme)
        for name, data in sorted(payloads.items()):
            add_bytes(archive, name, data)
        add_bytes(
            archive,
            "PACKAGE_MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files) + 2:
            raise RuntimeError("Submission package CRC or member-count validation failed")
    report = {
        "version": "rineng-v10-submission-package",
        "status": "pass",
        "archive": output.relative_to(root).as_posix(),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": digest(output.read_bytes()),
        "member_count": len(files) + 2,
        "active_dependency_count": len(source_files()),
    }
    report_path = root / "reports/generated/rineng_v10_submission_package.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
