"""Deterministic consistency checks for the completed mainline package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[dict[str, object]] = []

    def load(relative: str) -> dict:
        return json.loads((root / relative).read_text(encoding="utf-8"))

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    public = [json.loads(line) for line in (root / "data/processed/main400_source_test_diverse_public.jsonl").read_text(encoding="utf-8").splitlines()]
    hidden = [json.loads(line) for line in (root / "data/answer_store/main400_source_test_diverse_hidden.jsonl").read_text(encoding="utf-8").splitlines()]
    public_ids, hidden_ids = {row["instance_id"] for row in public}, {row["instance_id"] for row in hidden}
    check("public_400_records", len(public) == 400, str(len(public)))
    check("hidden_400_records", len(hidden) == 400, str(len(hidden)))
    check("public_100_sources", len({row["source_id"] for row in public}) == 100, str(len({row["source_id"] for row in public})))
    check("public_hidden_ids_match", public_ids == hidden_ids, f"public={len(public_ids)} hidden={len(hidden_ids)} intersection={len(public_ids & hidden_ids)}")
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
        payload = load(relative)
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
        payload = load(relative)
        check(f"bootstrap_rows:{relative}", len(payload["rows"]) == expected_rows, str(len(payload["rows"])))
        check(f"bootstrap_source_count:{relative}", all(row["source_count"] == 100 for row in payload["rows"]), str(payload))
    manifest = load("reports/generated/final_mainline_manifest.json")
    check("final_manifest_complete", manifest["missing_count"] == 0, str(manifest["missing_count"]))
    check("final_manifest_artifact_count", manifest["artifact_count"] >= 30, str(manifest["artifact_count"]))
    print(json.dumps({"status": "pass", "check_count": len(checks), "checks": checks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
