"""Deterministic consistency checks for the completed mainline package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    public_lines = (root / "data/processed/main400_source_test_diverse_public.jsonl").read_text(encoding="utf-8").splitlines()
    hidden_lines = (root / "data/answer_store/main400_source_test_diverse_hidden.jsonl").read_text(encoding="utf-8").splitlines()
    check("public_400_records", len(public_lines) == 400, str(len(public_lines)))
    check("hidden_400_records", len(hidden_lines) == 400, str(len(hidden_lines)))
    public = [json.loads(line) for line in public_lines]
    hidden = [json.loads(line) for line in hidden_lines]
    check("public_100_sources", len({row["source_id"] for row in public}) == 100, str(len({row["source_id"] for row in public})))
    check("public_hidden_ids_match", [row["instance_id"] for row in public] == [row["instance_id"] for row in hidden], "ordered IDs match")
    check("public_answer_isolated", all("answer" not in row and "cypher" not in row for row in public), "answer/cypher absent")

    score_paths = {
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
    scores = {}
    for label, relative in score_paths.items():
        payload = load(root, relative)
        scores[label] = payload
        check(f"{label}_record_count", payload["record_count"] == payload["prediction_count"] == 400, str(payload))
        check(f"{label}_coverage", payload["coverage"] == 1.0, str(payload["coverage"]))
        check(f"{label}_no_missing", payload["missing_prediction_count"] == 0 and not payload["extra_prediction_ids"], str(payload))
        check(f"{label}_all_ok", payload["status_counts"] == {"ok": 400}, str(payload["status_counts"]))

    check("8b_resolution_direction", scores["8b_3072"]["overall_accuracy"] > scores["8b_768"]["overall_accuracy"], str(scores))
    check("32b_resolution_direction", scores["32b_3072"]["overall_accuracy"] > scores["32b_1536"]["overall_accuracy"], str(scores))

    for relative, expected_rows in [
        ("reports/generated/qwen3vl8b_source400_resolution_bootstrap.json", 3),
        ("reports/generated/qwen3vl_cross_scale_resolution_bootstrap.json", 3),
        ("reports/generated/qwen3vl8b_source400_degradation_bootstrap.json", 3),
    ]:
        payload = load(root, relative)
        check(f"bootstrap_rows:{relative}", len(payload["rows"]) == expected_rows, str(len(payload["rows"])))
        check(f"bootstrap_source_count:{relative}", all(row["source_count"] == 100 for row in payload["rows"]), str(payload))

    manifest = load(root, "reports/generated/final_mainline_manifest.json")
    check("final_manifest_complete", manifest["missing_count"] == 0, str(manifest["missing_count"]))
    check("final_manifest_artifact_count", manifest["artifact_count"] >= 30, str(manifest["artifact_count"]))
    result = {"status": "pass", "check_count": len(checks), "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
