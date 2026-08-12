"""Score the v8 quality and budget-matched cross-family extensions.

This CPU-only scorer reads hidden references only after inference has stopped.
It validates row membership and provenance, computes paired source-bootstrap
intervals, and reports a paired difference-in-differences for whether image
degradation changes requested-drawing dependence relative to clean images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from run_e1_evidence_audit import (
    bootstrap_f1_difference,
    evaluate,
    f1_from_counts,
    source_accuracy,
    source_tag_counts,
)
from score_rineng_overnight_v7 import compact_metrics, paired_mean_difference_ci, pooled_f1, quantile


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pooled_for_sample(
    counts: dict[str, tuple[int, int, int]], sampled_sources: list[str]
) -> float:
    totals = [0, 0, 0]
    for source in sampled_sources:
        values = counts[source]
        for index, value in enumerate(values):
            totals[index] += value
    return f1_from_counts(*totals)


def paired_f1_difference_in_differences(
    clean_correct: dict[str, tuple[int, int, int]],
    clean_shuffled: dict[str, tuple[int, int, int]],
    degraded_correct: dict[str, tuple[int, int, int]],
    degraded_shuffled: dict[str, tuple[int, int, int]],
    *,
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    sources = sorted(
        set(clean_correct)
        & set(clean_shuffled)
        & set(degraded_correct)
        & set(degraded_shuffled)
    )
    if not sources:
        raise ValueError("No common sources for paired difference-in-differences")

    def estimate(sampled: list[str]) -> float:
        clean_effect = pooled_for_sample(clean_correct, sampled) - pooled_for_sample(
            clean_shuffled, sampled
        )
        degraded_effect = pooled_for_sample(
            degraded_correct, sampled
        ) - pooled_for_sample(degraded_shuffled, sampled)
        return degraded_effect - clean_effect

    point = estimate(sources)
    rng = random.Random(seed)
    samples = [estimate([rng.choice(sources) for _ in sources]) for _ in range(reps)]
    samples.sort()
    return point, [
        samples[round((reps - 1) * 0.025)],
        samples[round((reps - 1) * 0.975)],
    ]


def score_cell(
    *,
    root: Path,
    path: Path,
    expected_ids: list[str],
    references_by_id: dict[str, dict[str, Any]],
    expected_plan_sha256: str,
    expected_input_elements: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = read_jsonl(path)
    ids = [str(row.get("instance_id")) for row in predictions]
    duplicate_count = len(ids) - len(set(ids))
    error_count = sum(str(row.get("status")) != "ok" for row in predictions)
    answer_leak_flags = sum(row.get("test_answer_used") is True for row in predictions)
    plan_mismatch_count = sum(
        str(row.get("plan_sha256")) != expected_plan_sha256 for row in predictions
    )
    budget_mismatch_count = 0
    if expected_input_elements is not None:
        budget_mismatch_count = sum(
            int(row.get("actual_input_pixel_count", row.get("input_pixel_count", -1)))
            != expected_input_elements
            for row in predictions
        )
    scorer_records = [references_by_id[instance_id] for instance_id in expected_ids]
    scored = evaluate(scorer_records, predictions)
    metrics = scored["metrics"]
    complete = (
        len(predictions) == len(expected_ids)
        and set(ids) == set(expected_ids)
        and duplicate_count == 0
        and error_count == 0
        and answer_leak_flags == 0
        and plan_mismatch_count == 0
        and budget_mismatch_count == 0
        and not metrics["duplicate_prediction_ids"]
        and not metrics["extra_prediction_ids"]
    )
    latencies = [
        float(row["latency_seconds"])
        for row in predictions
        if row.get("latency_seconds") is not None
    ]
    cell = {
        "status": "pass" if complete else "fail",
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "row_count": len(predictions),
        "expected_row_count": len(expected_ids),
        "duplicate_count": duplicate_count,
        "error_count": error_count,
        "test_answer_used_true_count": answer_leak_flags,
        "plan_sha256_mismatch_count": plan_mismatch_count,
        "input_budget_mismatch_count": budget_mismatch_count,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": quantile(latencies, 0.95),
        },
        "metrics": compact_metrics(metrics),
    }
    return cell, scored["events"]


def csv_metric_row(
    *,
    experiment: str,
    model: str,
    dataset: str,
    quality: str | None,
    condition: str,
    cell: dict[str, Any],
) -> dict[str, Any]:
    metrics = cell["metrics"]
    tags = metrics["strict_value_tags"]
    return {
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "quality": quality,
        "condition": condition,
        "status": cell["status"],
        "records": cell["row_count"],
        "strict_accuracy": metrics["strict_accuracy"],
        "source_macro_accuracy": metrics["strict_source_macro_accuracy"],
        "value_exact_accuracy": metrics["task"]["value"]["strict_accuracy"],
        "value_tp": tags["tp"],
        "value_fp": tags["fp"],
        "value_fn": tags["fn"],
        "value_precision": tags["precision"],
        "value_recall": tags["recall"],
        "value_f1": tags["f1"],
        "latency_mean_seconds": cell["latency_seconds"]["mean"],
        "latency_p95_seconds": cell["latency_seconds"]["p95"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--quality-plan", default="data/manifests/rineng_v8_quality_robustness_plan.json"
    )
    parser.add_argument(
        "--internvl-plan",
        default="data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json",
    )
    parser.add_argument(
        "--quality-output-root", default="outputs/rineng_v8/qwen3vl8b_quality"
    )
    parser.add_argument(
        "--internvl-output-root", default="outputs/rineng_v8/internvl35_8b_budget54"
    )
    parser.add_argument(
        "--v7-internvl-root", default="outputs/rineng_overnight_v7/internvl35_8b"
    )
    parser.add_argument(
        "--output", default="reports/generated/rineng_v8_extension_score.json"
    )
    parser.add_argument("--csv", default="reports/generated/rineng_v8_extension_score.csv")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    references_path = root / "data/processed/pidqa_records.jsonl"
    references = read_jsonl(references_path)
    references_by_id = {str(row["instance_id"]): row for row in references}
    failures: list[str] = []
    cells: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    events: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    quality_plan_path = root / args.quality_plan
    quality_plan = json.loads(quality_plan_path.read_text(encoding="utf-8"))
    quality_plan_hash = sha256(quality_plan_path)
    quality_output_root = root / args.quality_output_root
    for spec in quality_plan["datasets"]:
        dataset_id = str(spec["dataset_id"])
        base_dataset = str(spec["base_dataset_id"])
        quality = str(spec["quality_condition"])
        expected_ids = [
            str(row["instance_id"]) for row in read_jsonl(root / spec["correct_input"])
        ]
        for condition in ("correct", "shuffled"):
            pattern = f"qwen3vl8b_{dataset_id}_p0_{condition}_3072.jsonl"
            path = quality_output_root / pattern
            cell_id = f"quality|qwen3vl8b|{base_dataset}|{quality}|{condition}"
            if not path.is_file():
                failures.append(f"{cell_id}: missing {path}")
                cells[cell_id] = {"status": "missing", "path": str(path)}
                continue
            cell, cell_events = score_cell(
                root=root,
                path=path,
                expected_ids=expected_ids,
                references_by_id=references_by_id,
                expected_plan_sha256=quality_plan_hash,
                expected_input_elements=35_979_264,
            )
            cells[cell_id] = cell
            events[("quality", base_dataset, quality, condition)] = cell_events
            csv_rows.append(
                csv_metric_row(
                    experiment="quality",
                    model="qwen3vl8b",
                    dataset=base_dataset,
                    quality=quality,
                    condition=condition,
                    cell=cell,
                )
            )
            if cell["status"] != "pass":
                failures.append(f"{cell_id}: validation failed")

    quality_comparisons = []
    comparison_index = 0
    quality_names = list(quality_plan["quality_conditions"])
    base_datasets = sorted({str(spec["base_dataset_id"]) for spec in quality_plan["datasets"]})
    for dataset in base_datasets:
        for quality in quality_names:
            correct_key = ("quality", dataset, quality, "correct")
            shuffled_key = ("quality", dataset, quality, "shuffled")
            if correct_key not in events or shuffled_key not in events:
                continue
            correct_counts = source_tag_counts(events[correct_key], "strict")
            shuffled_counts = source_tag_counts(events[shuffled_key], "strict")
            correct_accuracy = source_accuracy(events[correct_key], "strict_correct")
            shuffled_accuracy = source_accuracy(events[shuffled_key], "strict_correct")
            macro_difference, macro_ci = paired_mean_difference_ci(
                shuffled_accuracy,
                correct_accuracy,
                args.bootstrap_reps,
                18100 + comparison_index,
            )
            quality_comparisons.append(
                {
                    "dataset": dataset,
                    "quality": quality,
                    "contrast": "correct_minus_shuffled",
                    "value_f1_difference": pooled_f1(correct_counts) - pooled_f1(shuffled_counts),
                    "value_f1_source_bootstrap_ci95": list(
                        bootstrap_f1_difference(
                            shuffled_counts,
                            correct_counts,
                            args.bootstrap_reps,
                            17100 + comparison_index,
                        )
                    ),
                    "source_macro_strict_accuracy_difference": macro_difference,
                    "source_macro_source_bootstrap_ci95": macro_ci,
                }
            )
            comparison_index += 1

        clean_correct_key = ("quality", dataset, "clean", "correct")
        clean_shuffled_key = ("quality", dataset, "clean", "shuffled")
        if clean_correct_key not in events or clean_shuffled_key not in events:
            continue
        clean_correct = source_tag_counts(events[clean_correct_key], "strict")
        clean_shuffled = source_tag_counts(events[clean_shuffled_key], "strict")
        for quality in quality_names:
            if quality == "clean":
                continue
            degraded_correct_key = ("quality", dataset, quality, "correct")
            degraded_shuffled_key = ("quality", dataset, quality, "shuffled")
            if degraded_correct_key not in events or degraded_shuffled_key not in events:
                continue
            degraded_correct = source_tag_counts(events[degraded_correct_key], "strict")
            degraded_shuffled = source_tag_counts(events[degraded_shuffled_key], "strict")
            difference, interval = paired_f1_difference_in_differences(
                clean_correct,
                clean_shuffled,
                degraded_correct,
                degraded_shuffled,
                reps=args.bootstrap_reps,
                seed=19100 + comparison_index,
            )
            quality_comparisons.append(
                {
                    "dataset": dataset,
                    "quality": quality,
                    "contrast": "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1",
                    "value_f1_difference_in_differences": difference,
                    "source_bootstrap_ci95": interval,
                }
            )
            comparison_index += 1

    # The three frozen subsets are pairwise source-disjoint.  A pooled row is
    # therefore a genuine additional-source estimate rather than repeated
    # scoring of the same drawings.  Dataset-specific rows remain available
    # to expose heterogeneity.
    pooled_dataset_label = "pooled_three_source_disjoint_subsets"
    for quality in quality_names:
        pooled_correct_events = [
            event
            for dataset in base_datasets
            for event in events.get(("quality", dataset, quality, "correct"), [])
        ]
        pooled_shuffled_events = [
            event
            for dataset in base_datasets
            for event in events.get(("quality", dataset, quality, "shuffled"), [])
        ]
        if not pooled_correct_events or not pooled_shuffled_events:
            continue
        correct_counts = source_tag_counts(pooled_correct_events, "strict")
        shuffled_counts = source_tag_counts(pooled_shuffled_events, "strict")
        correct_accuracy = source_accuracy(pooled_correct_events, "strict_correct")
        shuffled_accuracy = source_accuracy(pooled_shuffled_events, "strict_correct")
        macro_difference, macro_ci = paired_mean_difference_ci(
            shuffled_accuracy,
            correct_accuracy,
            args.bootstrap_reps,
            23100 + comparison_index,
        )
        quality_comparisons.append(
            {
                "dataset": pooled_dataset_label,
                "quality": quality,
                "contrast": "correct_minus_shuffled",
                "value_f1_difference": pooled_f1(correct_counts) - pooled_f1(shuffled_counts),
                "value_f1_source_bootstrap_ci95": list(
                    bootstrap_f1_difference(
                        shuffled_counts,
                        correct_counts,
                        args.bootstrap_reps,
                        24100 + comparison_index,
                    )
                ),
                "source_macro_strict_accuracy_difference": macro_difference,
                "source_macro_source_bootstrap_ci95": macro_ci,
                "source_count": len(correct_counts),
            }
        )
        comparison_index += 1

    pooled_clean_correct_events = [
        event
        for dataset in base_datasets
        for event in events.get(("quality", dataset, "clean", "correct"), [])
    ]
    pooled_clean_shuffled_events = [
        event
        for dataset in base_datasets
        for event in events.get(("quality", dataset, "clean", "shuffled"), [])
    ]
    if pooled_clean_correct_events and pooled_clean_shuffled_events:
        pooled_clean_correct = source_tag_counts(pooled_clean_correct_events, "strict")
        pooled_clean_shuffled = source_tag_counts(pooled_clean_shuffled_events, "strict")
        for quality in quality_names:
            if quality == "clean":
                continue
            degraded_correct_events = [
                event
                for dataset in base_datasets
                for event in events.get(("quality", dataset, quality, "correct"), [])
            ]
            degraded_shuffled_events = [
                event
                for dataset in base_datasets
                for event in events.get(("quality", dataset, quality, "shuffled"), [])
            ]
            if not degraded_correct_events or not degraded_shuffled_events:
                continue
            difference, interval = paired_f1_difference_in_differences(
                pooled_clean_correct,
                pooled_clean_shuffled,
                source_tag_counts(degraded_correct_events, "strict"),
                source_tag_counts(degraded_shuffled_events, "strict"),
                reps=args.bootstrap_reps,
                seed=25100 + comparison_index,
            )
            quality_comparisons.append(
                {
                    "dataset": pooled_dataset_label,
                    "quality": quality,
                    "contrast": "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1",
                    "value_f1_difference_in_differences": difference,
                    "source_bootstrap_ci95": interval,
                    "source_count": len(pooled_clean_correct),
                }
            )
            comparison_index += 1

    internvl_plan_path = root / args.internvl_plan
    internvl_plan = json.loads(internvl_plan_path.read_text(encoding="utf-8"))
    internvl_plan_hash = sha256(internvl_plan_path)
    internvl_output_root = root / args.internvl_output_root
    expected_elements = int(internvl_plan["frozen_inference"]["total_input_tensor_elements"])
    internvl_label = str(internvl_plan["models"][0]["model_label"])
    for spec in internvl_plan["datasets"]:
        dataset_id = str(spec["dataset_id"])
        expected_ids = [
            str(row["instance_id"]) for row in read_jsonl(root / spec["correct_input"])
        ]
        for condition in internvl_plan["conditions"]:
            path = internvl_output_root / (
                f"{internvl_label}_{dataset_id}_p0_{condition}_letterbox54.jsonl"
            )
            cell_id = f"internvl_budget54|{dataset_id}|{condition}"
            if not path.is_file():
                failures.append(f"{cell_id}: missing {path}")
                cells[cell_id] = {"status": "missing", "path": str(path)}
                continue
            cell, cell_events = score_cell(
                root=root,
                path=path,
                expected_ids=expected_ids,
                references_by_id=references_by_id,
                expected_plan_sha256=internvl_plan_hash,
                expected_input_elements=None if condition == "text_only" else expected_elements,
            )
            cells[cell_id] = cell
            events[("internvl_budget54", dataset_id, condition)] = cell_events
            csv_rows.append(
                csv_metric_row(
                    experiment="internvl_budget54",
                    model=internvl_label,
                    dataset=dataset_id,
                    quality=None,
                    condition=condition,
                    cell=cell,
                )
            )
            if cell["status"] != "pass":
                failures.append(f"{cell_id}: validation failed")

    internvl_comparisons = []
    v7_root = root / args.v7_internvl_root
    internvl_datasets = [str(spec["dataset_id"]) for spec in internvl_plan["datasets"]]
    v7_events_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in internvl_datasets:
        correct_key = ("internvl_budget54", dataset, "correct")
        if correct_key not in events:
            continue
        correct_counts = source_tag_counts(events[correct_key], "strict")
        for control in ("shuffled", "text_only"):
            control_key = ("internvl_budget54", dataset, control)
            if control_key not in events:
                continue
            control_counts = source_tag_counts(events[control_key], "strict")
            internvl_comparisons.append(
                {
                    "dataset": dataset,
                    "contrast": f"correct_minus_{control}",
                    "value_f1_difference": pooled_f1(correct_counts) - pooled_f1(control_counts),
                    "value_f1_source_bootstrap_ci95": list(
                        bootstrap_f1_difference(
                            control_counts,
                            correct_counts,
                            args.bootstrap_reps,
                            21100 + len(internvl_comparisons),
                        )
                    ),
                }
            )
        v7_path = v7_root / f"internvl35_8b_{dataset}_p0_correct_tiles12.jsonl"
        if v7_path.is_file():
            expected_ids = [
                str(row["instance_id"])
                for row in read_jsonl(
                    root
                    / next(
                        spec["correct_input"]
                        for spec in internvl_plan["datasets"]
                        if spec["dataset_id"] == dataset
                    )
                )
            ]
            v7_predictions = read_jsonl(v7_path)
            v7_events = evaluate(
                [references_by_id[instance_id] for instance_id in expected_ids], v7_predictions
            )["events"]
            v7_events_by_dataset[dataset] = v7_events
            v7_counts = source_tag_counts(v7_events, "strict")
            internvl_comparisons.append(
                {
                    "dataset": dataset,
                    "contrast": "budget54_minus_native_tiles12_correct",
                    "value_f1_difference": pooled_f1(correct_counts) - pooled_f1(v7_counts),
                    "value_f1_source_bootstrap_ci95": list(
                        bootstrap_f1_difference(
                            v7_counts,
                            correct_counts,
                            args.bootstrap_reps,
                            22100 + len(internvl_comparisons),
                        )
                    ),
                    "v7_path": v7_path.relative_to(root).as_posix(),
                    "v7_sha256": sha256(v7_path),
                }
            )

    pooled_budget_correct_events = [
        event
        for dataset in internvl_datasets
        for event in events.get(("internvl_budget54", dataset, "correct"), [])
    ]
    if pooled_budget_correct_events:
        pooled_budget_correct = source_tag_counts(pooled_budget_correct_events, "strict")
        for control in ("shuffled", "text_only"):
            pooled_control_events = [
                event
                for dataset in internvl_datasets
                for event in events.get(("internvl_budget54", dataset, control), [])
            ]
            if not pooled_control_events:
                continue
            pooled_control = source_tag_counts(pooled_control_events, "strict")
            internvl_comparisons.append(
                {
                    "dataset": pooled_dataset_label,
                    "contrast": f"correct_minus_{control}",
                    "value_f1_difference": pooled_f1(pooled_budget_correct)
                    - pooled_f1(pooled_control),
                    "value_f1_source_bootstrap_ci95": list(
                        bootstrap_f1_difference(
                            pooled_control,
                            pooled_budget_correct,
                            args.bootstrap_reps,
                            26100 + len(internvl_comparisons),
                        )
                    ),
                    "source_count": len(pooled_budget_correct),
                }
            )
        pooled_v7_events = [
            event
            for dataset in internvl_datasets
            for event in v7_events_by_dataset.get(dataset, [])
        ]
        if pooled_v7_events and len(v7_events_by_dataset) == len(internvl_datasets):
            pooled_v7_counts = source_tag_counts(pooled_v7_events, "strict")
            internvl_comparisons.append(
                {
                    "dataset": pooled_dataset_label,
                    "contrast": "budget54_minus_native_tiles12_correct",
                    "value_f1_difference": pooled_f1(pooled_budget_correct)
                    - pooled_f1(pooled_v7_counts),
                    "value_f1_source_bootstrap_ci95": list(
                        bootstrap_f1_difference(
                            pooled_v7_counts,
                            pooled_budget_correct,
                            args.bootstrap_reps,
                            27100 + len(internvl_comparisons),
                        )
                    ),
                    "source_count": len(pooled_budget_correct),
                    "v7_dataset_count": len(v7_events_by_dataset),
                }
            )

    status = "pass" if not failures else "fail"
    report = {
        "version": "rineng-v8-extension-score",
        "status": status,
        "failure_reasons": failures,
        "reference": {
            "path": references_path.relative_to(root).as_posix(),
            "sha256": sha256(references_path),
            "role": "scorer-only; never read by inference runners",
        },
        "quality_plan": {"path": args.quality_plan, "sha256": quality_plan_hash},
        "internvl_plan": {"path": args.internvl_plan, "sha256": internvl_plan_hash},
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "unit": "source_id",
            "interval": "95% percentile",
            "pairing": "paired within dataset and intervention contrast",
        },
        "cells": cells,
        "quality_comparisons": quality_comparisons,
        "internvl_comparisons": internvl_comparisons,
        "integrity_boundary": (
            "All quality and InternVL experiments remain inside PIDQA. Budget matching uses input tensor "
            "elements and does not assert equivalent encoders or effective information throughput."
        ),
    }
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(root / args.csv, csv_rows)
    print(
        json.dumps(
            {
                "status": status,
                "cell_count": len(cells),
                "failure_count": len(failures),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
