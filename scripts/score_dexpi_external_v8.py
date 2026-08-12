"""Score the answer-isolated DEXPI external correct/control/OCR matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from prepare_dexpi_external_v8 import tag_prefix, tags_in_text


CONDITIONS = ("correct", "shuffled", "text_only", "paddleocr_full_image")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_path(path: Path, root: Path) -> str:
    """Prefer a portable repository-relative path, retaining external paths."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 1.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn else 1.0


def predicted_tags(row: dict[str, Any], prefix: str) -> set[str]:
    raw = row.get("raw", row.get("answer", ""))
    return {tag for tag in tags_in_text(str(raw or "")) if tag_prefix(tag) == prefix.casefold()}


def score_prediction_rows(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, tuple[int, int, int]], list[dict[str, Any]]]:
    pred_ids = [str(row.get("instance_id")) for row in predictions]
    pred_by_id = {str(row.get("instance_id")): row for row in predictions}
    ref_ids = [str(row["instance_id"]) for row in references]
    if len(pred_ids) != len(set(pred_ids)):
        raise ValueError("Duplicate prediction instance IDs")
    if set(pred_ids) != set(ref_ids):
        raise ValueError("Prediction/reference membership mismatch")
    if any(str(row.get("status")) != "ok" for row in predictions):
        raise ValueError("Prediction file contains failed rows")
    if any(row.get("test_answer_used") is True for row in predictions):
        raise ValueError("Prediction file declares hidden-answer use")

    totals = [0, 0, 0]
    exacts: list[int] = []
    group_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    group_exacts: dict[str, list[int]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    for reference in references:
        instance_id = str(reference["instance_id"])
        prediction = pred_by_id[instance_id]
        prefix = str(reference["fields"]["Prefix"]).casefold()
        truth = {str(tag).casefold() for tag in reference["answer"]}
        predicted = predicted_tags(prediction, prefix)
        tp = len(truth & predicted)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        exact = int(truth == predicted)
        group = str(reference["source_sheet"])
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        group_counts[group][0] += tp
        group_counts[group][1] += fp
        group_counts[group][2] += fn
        exacts.append(exact)
        group_exacts[group].append(exact)
        events.append(
            {
                "instance_id": instance_id,
                "source_id": reference["source_id"],
                "source_sheet": group,
                "prefix": prefix,
                "truth": sorted(truth),
                "predicted": sorted(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact": exact,
            }
        )
    tp, fp, fn = totals
    metrics = {
        "records": len(references),
        "logical_groups": len(group_counts),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
        "f1": f1(tp, fp, fn),
        "exact_set_accuracy": statistics.mean(exacts) if exacts else 0.0,
        "logical_group_macro_exact_accuracy": statistics.mean(
            statistics.mean(values) for values in group_exacts.values()
        )
        if group_exacts
        else 0.0,
    }
    return metrics, {key: tuple(value) for key, value in group_counts.items()}, events


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty list")
    return sorted_values[round((len(sorted_values) - 1) * probability)]


def paired_group_bootstrap(
    baseline: dict[str, tuple[int, int, int]],
    condition: dict[str, tuple[int, int, int]],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    groups = sorted(set(baseline) & set(condition))
    if set(groups) != set(baseline) or set(groups) != set(condition):
        raise ValueError("Bootstrap groups are not paired")

    def pooled(counts: dict[str, tuple[int, int, int]], sample: list[str]) -> float:
        values = [0, 0, 0]
        for group in sample:
            for index, value in enumerate(counts[group]):
                values[index] += value
        return f1(*values)

    point = pooled(condition, groups) - pooled(baseline, groups)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        sample = [rng.choice(groups) for _ in groups]
        samples.append(pooled(condition, sample) - pooled(baseline, sample))
    samples.sort()
    return {
        "difference": point,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "bootstrap_reps": reps,
        "unit": "logical DEXPI test case (source_sheet)",
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="data/manifests/rineng_v8_dexpi_external_plan.json")
    parser.add_argument(
        "--qwen-output-root", default="outputs/rineng_v8/dexpi_external_qwen"
    )
    parser.add_argument(
        "--ocr-output", default="outputs/rineng_v8/dexpi_external_ocr.jsonl"
    )
    parser.add_argument(
        "--output", default="reports/generated/rineng_v8_dexpi_external_score.json"
    )
    parser.add_argument(
        "--csv", default="reports/generated/rineng_v8_dexpi_external_score.csv"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan_path = (root / args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_inference":
        raise ValueError("Plan is not frozen_before_inference")
    plan_hash = sha256(plan_path)
    hidden_path = root / str(plan["answer_isolation"]["hidden_reference"])
    if sha256(hidden_path) != str(plan["answer_isolation"]["hidden_reference_sha256"]):
        raise ValueError("Hidden-reference hash mismatch")
    references = read_jsonl(hidden_path)
    dataset = plan["datasets"][0]
    if len(references) != int(dataset["record_count"]):
        raise ValueError("Hidden-reference membership mismatch")

    qwen_root = (root / args.qwen_output_root).resolve()
    paths = {
        condition: qwen_root / f"qwen3vl8b_dexpi_external_v8_p0_{condition}_3072.jsonl"
        for condition in ("correct", "shuffled", "text_only")
    }
    paths["paddleocr_full_image"] = (root / args.ocr_output).resolve()
    metrics: dict[str, Any] = {}
    group_counts: dict[str, dict[str, tuple[int, int, int]]] = {}
    events: dict[str, list[dict[str, Any]]] = {}
    integrity: dict[str, Any] = {}
    for condition in CONDITIONS:
        path = paths[condition]
        if not path.is_file():
            raise FileNotFoundError(path)
        predictions = read_jsonl(path)
        expected_plan_hash_mismatches = sum(
            str(row.get("plan_sha256")) != plan_hash for row in predictions
        )
        if expected_plan_hash_mismatches:
            raise ValueError(f"{condition}: plan hash mismatch")
        cell_metrics, counts, cell_events = score_prediction_rows(references, predictions)
        metrics[condition] = cell_metrics
        group_counts[condition] = counts
        events[condition] = cell_events
        integrity[condition] = {
            "path": report_path(path, root),
            "sha256": sha256(path),
            "rows": len(predictions),
            "plan_sha256_mismatch_count": expected_plan_hash_mismatches,
            "test_answer_used_true_count": sum(row.get("test_answer_used") is True for row in predictions),
            "error_count": sum(str(row.get("status")) != "ok" for row in predictions),
        }

    comparisons = {}
    for index, baseline in enumerate(("shuffled", "text_only", "paddleocr_full_image")):
        comparisons[f"correct_minus_{baseline}"] = paired_group_bootstrap(
            group_counts[baseline],
            group_counts["correct"],
            reps=args.bootstrap_reps,
            seed=28120 + index,
        )
    comparisons["ocr_minus_text_only"] = paired_group_bootstrap(
        group_counts["text_only"],
        group_counts["paddleocr_full_image"],
        reps=args.bootstrap_reps,
        seed=28130,
    )

    shuffled_mapping = dict(plan["shuffled_control"]["mapping"])
    report = {
        "version": "rineng-v8-dexpi-external-score",
        "status": "pass",
        "plan": {"path": args.plan, "sha256": plan_hash},
        "external_source": plan["external_source"],
        "selection": plan["selection"],
        "metrics": metrics,
        "comparisons": comparisons,
        "integrity": {
            "cells": integrity,
            "one_to_one_shuffled_mapping": len(set(shuffled_mapping.values())) == len(shuffled_mapping),
            "shuffled_fixed_point_count": sum(key == value for key, value in shuffled_mapping.items()),
            "same_logical_test_case_count": int(plan["shuffled_control"]["same_logical_test_case_count"]),
            "answer_isolation": plan["answer_isolation"],
        },
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "unit": "logical DEXPI test case (source_sheet)",
            "interval": "paired 95% percentile",
        },
        "scope_boundary": plan["analysis_boundary"],
    }
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        root / args.csv,
        [
            {"condition": condition, **values}
            for condition, values in metrics.items()
        ],
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "conditions": len(metrics),
                "records": len(references),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
