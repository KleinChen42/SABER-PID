"""Recompute manuscript headline numbers and write an evidence map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    score_files = {
        "8b_768": "reports/generated/qwen3vl8b_source400_clean_768_score.json",
        "8b_1536": "reports/generated/qwen3vl8b_source400_clean_1536_score.json",
        "8b_2304": "reports/generated/qwen3vl8b_source400_clean_2304_score.json",
        "8b_3072": "reports/generated/qwen3vl8b_source400_clean_3072_score.json",
        "32b_1536": "reports/generated/qwen3vl32b_source400_clean_1536_score.json",
        "32b_3072": "reports/generated/qwen3vl32b_source400_clean_3072_score.json",
        "blur": "reports/generated/qwen3vl8b_source400_blur_1536_score.json",
        "jpeg35": "reports/generated/qwen3vl8b_source400_jpeg35_1536_score.json",
        "center_crop": "reports/generated/qwen3vl8b_source400_center_crop_1536_score.json",
    }
    scores = {label: load(root, relative) for label, relative in score_files.items()}
    resolution = [scores[label]["overall_accuracy"] for label in ("8b_768", "8b_1536", "8b_2304", "8b_3072")]
    scale = [scores[label]["overall_accuracy"] for label in ("32b_1536", "32b_3072")]
    degradation = [scores[label]["overall_accuracy"] for label in ("blur", "jpeg35", "center_crop")]
    resolution_bootstrap = load(root, "reports/generated/qwen3vl8b_source400_resolution_bootstrap.json")
    cross_bootstrap = load(root, "reports/generated/qwen3vl_cross_scale_resolution_bootstrap.json")
    degradation_bootstrap = load(root, "reports/generated/qwen3vl8b_source400_degradation_bootstrap.json")

    text_paths = [root / "reports/MANUSCRIPT_RESULTS_DRAFT_V2.md", root / "reports/MAINLINE_FINAL_CLOSEOUT.md"]
    manuscript_text = "\n".join(path.read_text(encoding="utf-8") for path in text_paths)
    expected_strings = ["21.50%", "22.25%", "25.00%", "27.25%", "22.00%", "29.75%", "21.25%", "22.75%", "25.25%", "21.397%", "20.2 GiB", "68.3 GiB"]
    text_checks = {value: value in manuscript_text for value in expected_strings}
    rows = [
        {"claim": "Input-side same-drawing retrieval exposure", "evidence": ["reports/R1_INPUT_RETRIEVAL_CLOSEOUT.md", "reports/generated/pidqa_input_retrieval_seed_sweep.json"], "status": "supported"},
        {"claim": "8B source-disjoint resolution/latency curve", "evidence": [*score_files.values(), "reports/generated/qwen3vl8b_source400_resolution_bootstrap.json", "reports/generated/qwen3vl8b_source400_resolution_frontier.svg"], "status": "supported"},
        {"claim": "32B scale and resolution interaction", "evidence": [score_files["32b_1536"], score_files["32b_3072"], "reports/generated/qwen3vl_cross_scale_resolution_bootstrap.json"], "status": "bounded"},
        {"claim": "Controlled degradation boundary", "evidence": [score_files["blur"], score_files["jpeg35"], score_files["center_crop"], "reports/generated/qwen3vl8b_source400_degradation_bootstrap.json", "reports/generated/qwen3vl8b_source400_task_condition_heatmap.svg"], "status": "bounded"},
        {"claim": "Efficiency and memory cost", "evidence": ["reports/generated/main_efficiency_frontier.json", "reports/generated/main_efficiency_frontier.svg"], "status": "observed-run only"},
        {"claim": "Release and external-data limitations", "evidence": ["LICENSES.md", "reports/P9_LICENSE_RELEASE_AUDIT.md", "reports/R5_CROSS_FAMILY_STATUS.md", "reports/R7_EXTERNAL_STATUS.md"], "status": "explicit limitation"},
    ]
    passed = all(text_checks.values()) and resolution == [0.215, 0.2225, 0.25, 0.2725] and scale == [0.22, 0.2975] and degradation == [0.2125, 0.2275, 0.2525] and len(resolution_bootstrap["rows"]) == 3 and len(cross_bootstrap["rows"]) == 3 and len(degradation_bootstrap["rows"]) == 3
    payload = {
        "status": "pass" if passed else "fail",
        "recomputed": {"resolution_accuracy": [percent(value) for value in resolution], "scale_accuracy": [percent(value) for value in scale], "degradation_accuracy": [percent(value) for value in degradation]},
        "text_checks": text_checks,
        "bootstrap_row_counts": {"resolution": len(resolution_bootstrap["rows"]), "scale": len(cross_bootstrap["rows"]), "degradation": len(degradation_bootstrap["rows"])},
        "evidence_map": rows,
    }
    json_path, markdown_path = Path(args.json), Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# P9 claim-to-evidence map", "", f"Status: **{payload['status']}**", "", "| Claim | Status | Evidence |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['claim']} | {row['status']} | " + "; ".join(f"`{item}`" for item in row["evidence"]) + " |")
    lines += ["", "## Recomputed headline values", "", "- 8B resolution: " + ", ".join(payload["recomputed"]["resolution_accuracy"]), "- 32B scale/resolution: " + ", ".join(payload["recomputed"]["scale_accuracy"]), "- Degradation: " + ", ".join(payload["recomputed"]["degradation_accuracy"]), "", "All values above are read from scorer JSON files; no manuscript value is manually re-entered into the calculation."]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
