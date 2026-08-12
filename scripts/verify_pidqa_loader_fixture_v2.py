"""Validate PIDQA source identity on synthetic and real deterministic inputs.

This is the exact behavioral coverage of the one ``tmp_path`` pytest case,
implemented without pytest's Windows temporary-directory cleanup hook. It also
checks the real PIDQA loader summary used by the evidence route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pidbench.pidqa import load_pidqa, summarize_pidqa


FIXTURE_FILES = {
    "Simple Counting/simple_counting.csv": (
        "P&ID_number,Type,Symbol_Class,Q_id,Question,GT,Cypher\n"
        "7,Count,1,1,How many class 1?,2,MATCH (n) RETURN count(n)\n"
    ),
    "Spatial Counting/spatial_counting.csv": (
        "P&ID_number,Type,Symbol_XX,Symbol_YY,Q_id,Question,GT,Cypher\n"
        "7,Spatial-Count,1,2,1,How many connected?,1,MATCH (n) RETURN count(n)\n"
    ),
    "Spatial Connections/spatial_connectivity.csv": (
        "P&ID_number,Type,Symbol_XX,Symbol_YY,Symbol_ZZ,Q_id,Question,GT,Cypher\n"
        "7,Spatial-Connection,1,2,3,1,Connected?,True,MATCH (n) RETURN true\n"
    ),
    "Value/value_based.csv": (
        "P&ID_number,Type,Symbol_Class,Prefix,Q_id,Question,GT,Cypher\n"
        "7,Value-Based,1,A,1,List tags,['A-1'],MATCH (n) RETURN n.tag\n"
    ),
}


def ensure_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for relative, content in FIXTURE_FILES.items():
        file_path = path / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and file_path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Fixture content mismatch: {file_path}")
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/pidqa_loader_validation_v2.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    fixture = root / "tmp" / "pidqa_loader_fixture_validation_v2"
    ensure_fixture(fixture)
    synthetic_records = load_pidqa(fixture)
    synthetic_summary = summarize_pidqa(synthetic_records)
    if {record["source_id"] for record in synthetic_records} != {"pidqa-sheet-007"}:
        raise AssertionError("Synthetic source identity was not preserved")
    if synthetic_summary["record_count"] != 4:
        raise AssertionError("Synthetic fixture record count is not four")

    real_records = load_pidqa(root / "data" / "raw" / "PIDQA")
    real_summary = summarize_pidqa(real_records)
    expected_tasks = {
        "connectivity": 16000,
        "count": 16000,
        "spatial_count": 16000,
        "value": 16000,
    }
    if real_summary["record_count"] != 64000 or real_summary["source_count"] != 500:
        raise AssertionError("Real PIDQA record/source counts differ from the frozen contract")
    if real_summary["task_counts"] != expected_tasks:
        raise AssertionError("Real PIDQA task counts differ from the frozen contract")
    if not all(record["source_id"].startswith("pidqa-sheet-") for record in real_records):
        raise AssertionError("Real PIDQA source identifiers are malformed")

    report = {
        "validation_version": "pidqa-loader-v2",
        "pytest_tmp_path_equivalent": "test_load_pidqa_preserves_source_identity",
        "synthetic_fixture": {
            "path": str(fixture.relative_to(root)),
            "source_ids": sorted({record["source_id"] for record in synthetic_records}),
            "summary": synthetic_summary,
        },
        "real_pidqa": real_summary,
        "status": "pass",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
