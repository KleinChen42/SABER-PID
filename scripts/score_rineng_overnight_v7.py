"""Score the completed v7 overnight matrices after inference has stopped."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def pooled_f1(counts: dict[str, tuple[int, int, int]]) -> float:
    totals = [0, 0, 0]
    for values in counts.values():
        for index, value in enumerate(values):
            totals[index] += value
    return f1_from_counts(*totals)


def paired_mean_difference_ci(
    baseline: dict[str, float],
    condition: dict[str, float],
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    sources = sorted(set(baseline) & set(condition))
    if not sources:
        return 0.0, [0.0, 0.0]
    differences = [condition[source] - baseline[source] for source in sources]
    point = statistics.mean(differences)
    rng = random.Random(seed)
    samples = [
        statistics.mean(rng.choice(differences) for _ in differences)
        for _ in range(reps)
    ]
    samples.sort()
    return point, [
        samples[round((reps - 1) * 0.025)],
        samples[round((reps - 1) * 0.975)],
    ]


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_count": metrics["record_count"],
        "prediction_count": metrics["prediction_count"],
        "missing_prediction_count": metrics["missing_prediction_count"],
        "invalid_prediction_count": metrics["invalid_prediction_count"],
        "duplicate_prediction_ids": metrics["duplicate_prediction_ids"],
        "extra_prediction_ids": metrics["extra_prediction_ids"],
        "strict_accuracy": metrics["strict_accuracy"],
        "strict_source_macro_accuracy": metrics["strict_source_macro_accuracy"],
        "task": metrics["task"],
        "strict_value_tags": metrics["strict_value_tags"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--plan", default="data/manifests/rineng_overnight_v7_public_plan.json"
    )
    parser.add_argument("--output-root", default="outputs/rineng_overnight_v7")
    parser.add_argument(
        "--output", default="reports/generated/rineng_overnight_v7_score.json"
    )
    parser.add_argument(
        "--csv", default="reports/generated/rineng_overnight_v7_score.csv"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan_path = root / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reference_path = root / "data/processed/pidqa_records.jsonl"
    references = read_jsonl(reference_path)
    reference_by_id = {str(row["instance_id"]): row for row in references}
    prompts = [str(row["prompt_id"]) for row in plan["prompts"]]
    conditions = [str(value) for value in plan["conditions"]]
    output_root = root / args.output_root
    failures: list[str] = []
    cells: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    events_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}

    for model in plan["models"]:
        model_label = str(model["model_label"])
        model_dir = output_root / model_label
        for dataset in plan["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            correct_manifest = read_jsonl(root / dataset["correct_input"])
            expected_ids = [str(row["instance_id"]) for row in correct_manifest]
            scorer_records = [reference_by_id[instance_id] for instance_id in expected_ids]
            for prompt_id in prompts:
                for condition in conditions:
                    pattern = f"{model_label}_{dataset_id}_{prompt_id}_{condition}_*.jsonl"
                    matches = sorted(model_dir.glob(pattern))
                    cell_id = f"{model_label}|{dataset_id}|{prompt_id}|{condition}"
                    if len(matches) != 1:
                        failures.append(f"{cell_id}: expected one output, found {len(matches)}")
                        cells[cell_id] = {"status": "missing_or_ambiguous", "matches": [str(path) for path in matches]}
                        continue
                    path = matches[0]
                    predictions = read_jsonl(path)
                    ids = [str(row.get("instance_id")) for row in predictions]
                    duplicate_count = len(ids) - len(set(ids))
                    error_count = sum(str(row.get("status")) != "ok" for row in predictions)
                    answer_leak_flags = sum(row.get("test_answer_used") is True for row in predictions)
                    scored = evaluate(scorer_records, predictions)
                    metrics = scored["metrics"]
                    complete = (
                        len(predictions) == len(expected_ids)
                        and set(ids) == set(expected_ids)
                        and duplicate_count == 0
                        and error_count == 0
                        and answer_leak_flags == 0
                        and not metrics["duplicate_prediction_ids"]
                        and not metrics["extra_prediction_ids"]
                    )
                    if not complete:
                        failures.append(f"{cell_id}: incomplete or invalid output")
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
                        "latency_seconds": {
                            "mean": statistics.mean(latencies) if latencies else None,
                            "median": statistics.median(latencies) if latencies else None,
                            "p95": quantile(latencies, 0.95),
                        },
                        "metrics": compact_metrics(metrics),
                    }
                    cells[cell_id] = cell
                    events_by_key[(model_label, dataset_id, prompt_id, condition)] = scored["events"]
                    strict_tags = metrics["strict_value_tags"]
                    csv_rows.append(
                        {
                            "model": model_label,
                            "dataset": dataset_id,
                            "prompt": prompt_id,
                            "condition": condition,
                            "records": len(predictions),
                            "strict_accuracy": metrics["strict_accuracy"],
                            "source_macro_accuracy": metrics["strict_source_macro_accuracy"],
                            "connectivity_accuracy": metrics["task"]["connectivity"]["strict_accuracy"],
                            "count_accuracy": metrics["task"]["count"]["strict_accuracy"],
                            "spatial_count_accuracy": metrics["task"]["spatial_count"]["strict_accuracy"],
                            "value_exact_accuracy": metrics["task"]["value"]["strict_accuracy"],
                            "value_tp": strict_tags["tp"],
                            "value_fp": strict_tags["fp"],
                            "value_fn": strict_tags["fn"],
                            "value_precision": strict_tags["precision"],
                            "value_recall": strict_tags["recall"],
                            "value_f1": strict_tags["f1"],
                            "latency_mean_seconds": cell["latency_seconds"]["mean"],
                            "latency_p95_seconds": cell["latency_seconds"]["p95"],
                        }
                    )

    comparisons: list[dict[str, Any]] = []
    comparison_index = 0
    for model in plan["models"]:
        model_label = str(model["model_label"])
        for dataset in plan["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            for prompt_id in prompts:
                correct_key = (model_label, dataset_id, prompt_id, "correct")
                if correct_key not in events_by_key:
                    continue
                correct_events = events_by_key[correct_key]
                correct_counts = source_tag_counts(correct_events, "strict")
                correct_accuracy = source_accuracy(correct_events, "strict_correct")
                for control in ("shuffled", "text_only"):
                    control_key = (model_label, dataset_id, prompt_id, control)
                    if control_key not in events_by_key:
                        continue
                    control_events = events_by_key[control_key]
                    control_counts = source_tag_counts(control_events, "strict")
                    control_accuracy = source_accuracy(control_events, "strict_correct")
                    value_difference = pooled_f1(correct_counts) - pooled_f1(control_counts)
                    value_ci = bootstrap_f1_difference(
                        control_counts,
                        correct_counts,
                        args.bootstrap_reps,
                        7200 + comparison_index,
                    )
                    accuracy_difference, accuracy_ci = paired_mean_difference_ci(
                        control_accuracy,
                        correct_accuracy,
                        args.bootstrap_reps,
                        8200 + comparison_index,
                    )
                    comparisons.append(
                        {
                            "model": model_label,
                            "dataset": dataset_id,
                            "prompt": prompt_id,
                            "contrast": f"correct_minus_{control}",
                            "value_f1_difference": value_difference,
                            "value_f1_source_bootstrap_ci95": list(value_ci),
                            "source_macro_strict_accuracy_difference": accuracy_difference,
                            "source_bootstrap_ci95": accuracy_ci,
                        }
                    )
                    comparison_index += 1

    prompt_sensitivity: list[dict[str, Any]] = []
    if set(("p0", "p1")) <= set(prompts):
        for model in plan["models"]:
            model_label = str(model["model_label"])
            for dataset in plan["datasets"]:
                dataset_id = str(dataset["dataset_id"])
                p0_key = (model_label, dataset_id, "p0", "correct")
                p1_key = (model_label, dataset_id, "p1", "correct")
                if p0_key not in events_by_key or p1_key not in events_by_key:
                    continue
                p0_counts = source_tag_counts(events_by_key[p0_key], "strict")
                p1_counts = source_tag_counts(events_by_key[p1_key], "strict")
                prompt_sensitivity.append(
                    {
                        "model": model_label,
                        "dataset": dataset_id,
                        "contrast": "p1_minus_p0_correct_value_f1",
                        "difference": pooled_f1(p1_counts) - pooled_f1(p0_counts),
                        "source_bootstrap_ci95": list(
                            bootstrap_f1_difference(
                                p0_counts,
                                p1_counts,
                                args.bootstrap_reps,
                                9200 + len(prompt_sensitivity),
                            )
                        ),
                        "selection_note": "Both pre-existing prompts are reported; no best-prompt selection.",
                    }
                )

    status = "pass" if not failures else "fail"
    report = {
        "version": "rineng-overnight-v7-score",
        "status": status,
        "failure_reasons": failures,
        "plan": args.plan,
        "plan_sha256": sha256(plan_path),
        "reference": {
            "path": reference_path.relative_to(root).as_posix(),
            "role": "scorer-only; not read by inference runners",
            "sha256": sha256(reference_path),
        },
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "unit": "source_id",
            "interval": "95% percentile",
            "pairing": "paired within model/dataset/prompt contrast",
        },
        "cells": cells,
        "counterfactual_comparisons": comparisons,
        "prompt_sensitivity": prompt_sensitivity,
        "boundary": (
            "All three subsets are source-disjoint, but all belong to PIDQA; "
            "the report cannot establish external-family transport."
        ),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(root / args.csv, csv_rows)
    print(
        json.dumps(
            {
                "status": status,
                "output": output.relative_to(root).as_posix(),
                "cell_count": len(cells),
                "comparison_count": len(comparisons),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

