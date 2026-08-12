"""Build a machine-readable evidence manifest for the fixed PID reliability mainline.

The manifest intentionally separates primary results, negative ablations, and
descriptive pilots.  It is a packaging/consistency aid; it does not invent
new scores or promote exploratory runs to confirmatory claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def artifact(root: Path, relative_path: str, *, with_hash: bool = True) -> dict[str, Any]:
    path = root / relative_path
    item: dict[str, Any] = {
        "path": relative_path.replace("\\", "/"),
        "exists": path.exists(),
    }
    if path.exists():
        item["bytes"] = path.stat().st_size
        if with_hash:
            item["sha256"] = sha256(path)
    return item


def score_entry(
    root: Path,
    *,
    entry_id: str,
    stage: str,
    split_scope: str,
    condition: str,
    model: str,
    score_relative: str,
    prediction_relative: str,
    source_count: int | None,
    claim_role: str,
    caveat: str,
) -> dict[str, Any]:
    score_path = root / score_relative
    score = read_json(score_path)
    prediction_path = root / prediction_relative
    expected = score.get("record_count")
    observed = line_count(prediction_path)
    if expected is not None and observed is not None and expected != observed:
        raise AssertionError(
            f"{entry_id}: score record_count={expected} but prediction lines={observed}"
        )
    return {
        "id": entry_id,
        "stage": stage,
        "kind": "scored_vlm_run",
        "split_scope": split_scope,
        "condition": condition,
        "model": model,
        "claim_role": claim_role,
        "caveat": caveat,
        "score": {
            "label": score.get("label"),
            "record_count": score.get("record_count"),
            "prediction_count": score.get("prediction_count"),
            "answered_count": score.get("answered_count"),
            "answered_correct_count": score.get("answered_correct_count"),
            "coverage": score.get("coverage"),
            "overall_accuracy": score.get("overall_accuracy"),
            "answered_accuracy": score.get("answered_accuracy"),
            "mean_latency_seconds": score.get("mean_latency_seconds"),
            "task_accuracy": score.get("task_accuracy"),
            "missing_prediction_count": score.get("missing_prediction_count"),
            "extra_prediction_ids": score.get("extra_prediction_ids"),
        },
        "source_count": source_count,
        "score_artifact": artifact(root, score_relative),
        "prediction_artifact": {
            **artifact(root, prediction_relative),
            "nonempty_jsonl_lines": observed,
        },
    }


def mainline_manifest(root: Path) -> dict[str, Any]:
    normalization = read_json(root / "reports/generated/pidqa_normalization_summary.json")
    exposure = read_json(root / "reports/generated/pidqa_split_exposure_seed_sweep.json")
    normalizer = read_json(root / "reports/generated/pidqa_question_normalizer_audit.json")

    entries: list[dict[str, Any]] = [
        {
            "id": "E1.split_exposure",
            "stage": "E1",
            "kind": "diagnostic_audit",
            "split_scope": "random_vs_source_disjoint",
            "claim_role": "primary",
            "caveat": exposure.get("method_role"),
            "summary": {
                "random_mean_same_source_rate": exposure["aggregate"]["random"]["mean_same_source_test_rate"],
                "random_mean_unambiguous_cache_hit_rate": exposure["aggregate"]["random"]["mean_unambiguous_cache_hit_rate"],
                "random_min_unambiguous_cache_hit_rate": exposure["aggregate"]["random"]["min_unambiguous_cache_hit_rate"],
                "random_max_unambiguous_cache_hit_rate": exposure["aggregate"]["random"]["max_unambiguous_cache_hit_rate"],
                "source_mean_unambiguous_cache_hit_rate": exposure["aggregate"]["source"]["mean_unambiguous_cache_hit_rate"],
                "seed_count": len(exposure.get("rows", [])),
            },
            "artifacts": [
                artifact(root, "reports/generated/pidqa_split_exposure_seed_sweep.json"),
                artifact(root, "reports/generated/pidqa_split_exposure_seed_sweep.csv"),
                artifact(root, "reports/generated/pidqa_random_semantic_overlap_examples.json"),
                artifact(root, "reports/generated/pidqa_random_semantic_overlap_examples.csv"),
            ],
        },
        {
            "id": "E1.question_normalization",
            "stage": "E1",
            "kind": "deterministic_audit",
            "split_scope": "all_pidqa_records",
            "claim_role": "supporting",
            "caveat": normalizer.get("scope"),
            "summary": {
                "record_count": normalizer.get("record_count"),
                "match_count": normalizer.get("match_count"),
                "match_rate": normalizer.get("match_rate"),
                "parse_error_count": normalizer.get("parse_error_count"),
            },
            "artifacts": [artifact(root, "reports/generated/pidqa_question_normalizer_audit.json")],
        },
        score_entry(
            root,
            entry_id="E3.source200.highres",
            stage="E3",
            split_scope="50_source_test_sheets",
            condition="clean_3072_max_side",
            model="Qwen3-VL-8B-Instruct",
            score_relative="reports/generated/qwen3vl8b_source200_clean_highres_direct_pidqa_score.json",
            prediction_relative="outputs/pilot/qwen3vl8b_source200_clean_highres_direct.jsonl",
            source_count=50,
            claim_role="primary_baseline",
            caveat="Frozen-model direct answering; source-disjoint main set.",
        ),
        score_entry(
            root,
            entry_id="E3.source200.lowres",
            stage="E3",
            split_scope="50_source_test_sheets",
            condition="clean_1536_max_side",
            model="Qwen3-VL-8B-Instruct",
            score_relative="reports/generated/qwen3vl8b_source200_clean_lowres_direct_pidqa_score.json",
            prediction_relative="outputs/pilot/qwen3vl8b_source200_clean_lowres_direct.jsonl",
            source_count=50,
            claim_role="primary_condition",
            caveat="Paired source-sheet condition; resolution and latency trade-off.",
        ),
        score_entry(
            root,
            entry_id="E3.source200.center_crop",
            stage="E3",
            split_scope="50_source_test_sheets",
            condition="70_percent_center_crop_at_1536",
            model="Qwen3-VL-8B-Instruct",
            score_relative="reports/generated/qwen3vl8b_source200_center_crop_1536_direct_pidqa_score.json",
            prediction_relative="outputs/pilot/qwen3vl8b_source200_center_crop_1536_direct.jsonl",
            source_count=50,
            claim_role="primary_context_sensitivity_condition",
            caveat="Crop comparison is not promoted to an independent context-loss claim.",
        ),
        {
            "id": "E3.source200.bootstrap",
            "stage": "E3",
            "kind": "paired_source_bootstrap",
            "split_scope": "50_source_test_sheets",
            "claim_role": "primary_inference",
            "caveat": "10,000 paired source-sheet resamples; intervals support the bounded resolution claim.",
            "artifacts": [
                artifact(root, "reports/generated/qwen3vl8b_source200_highres_condition_bootstrap.json"),
                artifact(root, "reports/generated/qwen3vl8b_source200_highres_condition_bootstrap.csv"),
                artifact(root, "reports/generated/qwen3vl8b_source200_crop_vs_lowres_bootstrap.json"),
                artifact(root, "reports/generated/qwen3vl8b_source200_crop_vs_lowres_bootstrap.csv"),
                artifact(root, "reports/generated/qwen3vl8b_source200_resolution_context_table.md"),
                artifact(root, "reports/generated/qwen3vl8b_source200_resolution_context_table.csv"),
            ],
        },
        score_entry(
            root,
            entry_id="E2.structured_negative",
            stage="E2",
            split_scope="12_source_test_sheets",
            condition="answer_first_structured_prompt",
            model="Qwen3-VL-8B-Instruct",
            score_relative="reports/generated/qwen3vl8b_source_test_diverse_highres_answerfirst_pidqa_score.json",
            prediction_relative="outputs/pilot/qwen3vl8b_source_test_diverse_highres_answerfirst.jsonl",
            source_count=12,
            claim_role="negative_ablation",
            caveat="Low coverage and lower overall accuracy; branch stopped at minimum useful evidence.",
        ),
        score_entry(
            root,
            entry_id="E3.scale.8b",
            stage="E3",
            split_scope="12_source_test_sheets",
            condition="clean_3072_max_side",
            model="Qwen3-VL-8B-Instruct",
            score_relative="reports/generated/qwen3vl8b_source_test_diverse_highres_direct_pidqa_score.json",
            prediction_relative="outputs/pilot/qwen3vl8b_source_test_diverse_highres_direct.jsonl",
            source_count=12,
            claim_role="descriptive_scale_baseline",
            caveat="Small pilot; used only for a descriptive scale/latency trend.",
        ),
        score_entry(
            root,
            entry_id="E3.scale.32b",
            stage="E3",
            split_scope="12_source_test_sheets",
            condition="clean_3072_max_side",
            model="Qwen3-VL-32B-Instruct",
            score_relative="reports/generated/qwen3vl32b_source_test_diverse_highres_direct_pidqa_score.json",
            prediction_relative="outputs/pilot/qwen3vl32b_source_test_diverse_highres_direct.jsonl",
            source_count=12,
            claim_role="descriptive_scale_condition",
            caveat="Small pilot; paired bootstrap interval crosses zero; not a definitive ranking.",
        ),
        {
            "id": "E3.scale.bootstrap",
            "stage": "E3",
            "kind": "paired_source_bootstrap",
            "split_scope": "12_source_test_sheets",
            "claim_role": "descriptive_inference",
            "caveat": "Scale difference is reported as a trend because the 95% interval crosses zero.",
            "artifacts": [
                artifact(root, "reports/generated/qwen3vl_scale_bootstrap.json"),
                artifact(root, "reports/generated/qwen3vl_scale_bootstrap.csv"),
                artifact(root, "reports/generated/qwen3vl_scale_pilot_table.md"),
                artifact(root, "reports/generated/qwen3vl_scale_pilot_table.csv"),
            ],
        },
        {
            "id": "E4.external_transfer",
            "stage": "E4",
            "kind": "external_source_status",
            "split_scope": "PID2Graph/OPEN100",
            "claim_role": "blocked_non_result",
            "status": "blocked_by_transfer_infrastructure",
            "caveat": "Official Zenodo archive is reachable, but the observed transfer rate implied roughly 25 hours; the incomplete partial archive was not extracted or scored.",
            "summary": {
                "official_record": "https://zenodo.org/records/14803338",
                "archive_size_gb": 9.3,
                "observed_rate_mb_per_s": 0.1,
                "external_score_reported": False,
            },
            "artifacts": [artifact(root, "scripts/download_remote_pid2graph.sh")],
        },
    ]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "PID reliability benchmark",
        "mainline": ["E0 data/splits/scoring", "E1 source exposure", "E2 negative structured branch", "E3 resolution and scale", "E4 external transfer status"],
        "data": {
            "record_count": normalization.get("record_count"),
            "source_count": normalization.get("source_count"),
            "task_counts": normalization.get("task_counts"),
            "questions_per_source": normalization.get("questions_per_source"),
            "normalization_artifact": artifact(root, "reports/generated/pidqa_normalization_summary.json"),
        },
        "entries": entries,
        "claim_boundary": {
            "supported": [
                "Random question-level splits retain a material same-drawing semantic-query exposure pathway.",
                "Source-disjoint Qwen3-VL 8B performance on this PIDQA main set improves at the higher tested resolution with a measured latency cost.",
                "The tested structured-answer branch is a negative ablation with low coverage.",
            ],
            "descriptive_only": [
                "The 32B versus 8B accuracy difference is a small-pilot scale/latency trend.",
                "Crop versus low-resolution behavior is not promoted to a standalone context-loss claim.",
            ],
            "not_supported": [
                "Universal VLM inflation under random splits.",
                "Structured checker or deployment safety improvement.",
                "External PID2Graph transfer score.",
            ],
        },
    }
    return manifest


def write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        score = entry.get("score", {})
        rows.append(
            {
                "id": entry.get("id"),
                "stage": entry.get("stage"),
                "kind": entry.get("kind"),
                "split_scope": entry.get("split_scope"),
                "claim_role": entry.get("claim_role"),
                "status": entry.get("status", "complete"),
                "model": entry.get("model", ""),
                "condition": entry.get("condition", ""),
                "source_count": entry.get("source_count", ""),
                "record_count": score.get("record_count", entry.get("summary", {}).get("record_count", "")),
                "overall_accuracy": score.get("overall_accuracy", ""),
                "coverage": score.get("coverage", ""),
                "mean_latency_seconds": score.get("mean_latency_seconds", ""),
                "caveat": entry.get("caveat", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = mainline_manifest(root)
    json_path = (args.json or root / "reports/generated/mainline_evidence_manifest.json").resolve()
    csv_path = (args.csv or root / "reports/generated/mainline_evidence_manifest.csv").resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, manifest["entries"])
    print(json.dumps({"json": rel(root, json_path), "csv": rel(root, csv_path), "entry_count": len(manifest["entries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
