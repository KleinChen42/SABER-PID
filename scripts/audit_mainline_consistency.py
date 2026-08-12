"""Recompute and assert the key numerical boundaries of the mainline package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = load(root / "reports/generated/mainline_evidence_manifest.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    entries = manifest["entries"]
    ids = [entry["id"] for entry in entries]
    check("manifest_entry_count", len(entries) == 11, f"observed={len(entries)}")
    check("manifest_ids_unique", len(ids) == len(set(ids)), "entry ids are unique")

    for entry in entries:
        for key in ("score_artifact", "prediction_artifact"):
            if key not in entry:
                continue
            item = entry[key]
            check(f"artifact_exists:{entry['id']}:{key}", item["exists"], item["path"])
        if entry.get("kind") == "scored_vlm_run":
            score = entry["score"]
            observed = entry["prediction_artifact"].get("nonempty_jsonl_lines")
            check(
                f"prediction_count:{entry['id']}",
                score["record_count"] == observed == score["prediction_count"],
                f"record={score['record_count']} prediction={score['prediction_count']} lines={observed}",
            )
            check(
                f"complete_status:{entry['id']}",
                entry.get("status", "complete") == "complete",
                f"status={entry.get('status', 'complete')}",
            )

    normalization = load(root / "reports/generated/pidqa_normalization_summary.json")
    check("pidqa_record_count", normalization["record_count"] == 64000, str(normalization["record_count"]))
    check("pidqa_source_count", normalization["source_count"] == 500, str(normalization["source_count"]))
    check(
        "pidqa_task_balance",
        set(normalization["task_counts"].values()) == {16000},
        str(normalization["task_counts"]),
    )

    normalizer = load(root / "reports/generated/pidqa_question_normalizer_audit.json")
    check("question_normalizer_complete", normalizer["match_count"] == normalizer["record_count"] == 64000, str(normalizer))

    exposure = load(root / "reports/generated/pidqa_split_exposure_seed_sweep.json")
    random_summary = exposure["aggregate"]["random"]
    source_summary = exposure["aggregate"]["source"]
    check("random_same_source_complete", close(random_summary["mean_same_source_test_rate"], 1.0), str(random_summary))
    check("random_semantic_exposure", close(random_summary["mean_unambiguous_cache_hit_rate"], 0.21396875), str(random_summary))
    check("source_semantic_exposure_zero", close(source_summary["mean_unambiguous_cache_hit_rate"], 0.0), str(source_summary))

    highres_bootstrap = load(root / "reports/generated/qwen3vl8b_source200_highres_condition_bootstrap.json")
    by_condition = {row["condition"]: row for row in highres_bootstrap["rows"]}
    lowres = by_condition["lowres_clean"]
    crop = by_condition["center_crop_1536"]
    check("lowres_accuracy", close(lowres["condition_accuracy"], 0.23), str(lowres))
    check("lowres_delta", close(lowres["difference_condition_minus_baseline"], -0.06), str(lowres))
    check("crop_accuracy", close(crop["condition_accuracy"], 0.235), str(crop))
    check("crop_delta", close(crop["difference_condition_minus_baseline"], -0.055), str(crop))

    crop_bootstrap = load(root / "reports/generated/qwen3vl8b_source200_crop_vs_lowres_bootstrap.json")
    crop_lowres = crop_bootstrap["rows"][0]
    check("crop_vs_lowres_delta", close(crop_lowres["difference_condition_minus_baseline"], 0.005), str(crop_lowres))
    check("crop_vs_lowres_ci_crosses_zero", crop_lowres["bootstrap_ci95_low"] < 0 < crop_lowres["bootstrap_ci95_high"], str(crop_lowres))

    scale_bootstrap = load(root / "reports/generated/qwen3vl_scale_bootstrap.json")["rows"][0]
    check("scale_delta", close(scale_bootstrap["difference_condition_minus_baseline"], 1 / 12), str(scale_bootstrap))
    check("scale_ci_crosses_zero", scale_bootstrap["bootstrap_ci95_low"] < 0 < scale_bootstrap["bootstrap_ci95_high"], str(scale_bootstrap))

    e4 = next(entry for entry in entries if entry["id"] == "E4.external_transfer")
    check("e4_no_score_claim", e4["summary"]["external_score_reported"] is False, str(e4["summary"]))
    check("e4_blocked_is_explicit", e4["status"] == "blocked_by_transfer_infrastructure", e4["status"])

    print(json.dumps({"status": "pass", "check_count": len(checks), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
