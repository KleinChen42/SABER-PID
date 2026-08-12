"""Build the final machine-readable mainline artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ARTIFACTS = [
    "17_FIXED_NEXT_EXPERIMENT_ROUTE.md",
    "data/processed/main400_source_test_diverse_public.jsonl",
    "data/answer_store/main400_source_test_diverse_hidden.jsonl",
    "reports/R1_INPUT_RETRIEVAL_CLOSEOUT.md",
    "reports/R5_CROSS_FAMILY_STATUS.md",
    "reports/R7_EXTERNAL_STATUS.md",
    "reports/MAINLINE_FINAL_CLOSEOUT.md",
    "reports/MANUSCRIPT_RESULTS_DRAFT_V2.md",
    "reports/SUBMISSION_PACKAGE_INDEX_V2.md",
    "reports/generated/main400_source_test_diverse_summary.json",
    "reports/generated/main400_image_binding_summary.json",
    "reports/generated/all500_image_download_tolerant_summary.json",
    "reports/generated/pidqa_input_retrieval_seed_sweep.json",
    "reports/generated/qwen3vl8b_source400_resolution_table.json",
    "reports/generated/qwen3vl8b_source400_resolution_table.csv",
    "reports/generated/qwen3vl8b_source400_resolution_bootstrap.json",
    "reports/generated/qwen3vl8b_source400_resolution_bootstrap_from768.json",
    "reports/generated/qwen3vl8b_source400_resolution_frontier.svg",
    "reports/generated/qwen3vl32b_source400_resolution_table.json",
    "reports/generated/qwen3vl32b_source400_resolution_table.csv",
    "reports/generated/qwen3vl_cross_scale_resolution_bootstrap.json",
    "reports/generated/qwen3vl32b_minus_8b_3072_bootstrap.json",
    "reports/generated/qwen3vl32b_resolution_bootstrap.json",
    "reports/generated/qwen3vl8b_source400_degradation_table.json",
    "reports/generated/qwen3vl8b_source400_degradation_table.csv",
    "reports/generated/qwen3vl8b_source400_degradation_bootstrap.json",
    "reports/generated/qwen3vl8b_source400_task_condition_heatmap.svg",
    "reports/generated/main_efficiency_frontier.json",
    "reports/generated/main_efficiency_frontier.csv",
    "reports/generated/main_efficiency_frontier.svg",
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for relative in ARTIFACTS:
        path = root / relative
        exists = path.exists() and path.is_file()
        rows.append(
            {
                "path": relative,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": digest(path) if exists else "",
            }
        )
    payload = {
        "manifest_version": "mainline-v2",
        "root": str(root),
        "artifact_count": len(rows),
        "missing_count": sum(not row["exists"] for row in rows),
        "claim_boundary": {
            "primary": "source-disjoint PIDQA resolution/latency trade-off for Qwen3-VL",
            "not_supported": [
                "universal cross-family generalization",
                "external PID2Graph score without a verified archive",
                "hardware-independent energy efficiency",
            ],
        },
        "rows": rows,
    }
    json_path, csv_path = Path(args.json), Path(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "exists", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["missing_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
