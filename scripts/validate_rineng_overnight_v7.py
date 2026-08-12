"""Independently validate the frozen RINENG overnight-v7 score report.

The validator does not import the scorer or its audit helper. It recomputes
cell metrics, paired counterfactual effects, source-level percentile bootstrap
intervals, prompt sensitivity, hashes, and the flat CSV from immutable inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pidbench.pidqa_metrics import normalize_pidqa_answer  # noqa: E402


TASKS = ("connectivity", "count", "spatial_count", "value")
TOLERANCE = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_text(row: dict[str, Any]) -> Any:
    return row["answer"] if "answer" in row else row.get("raw")


def is_answer(row: dict[str, Any]) -> bool:
    if "action" in row:
        return str(row.get("action")) == "ANSWER"
    return str(row.get("status", "ok")) == "ok"


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def close(left: float | int, right: float | int, tolerance: float = TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def percentile_interval(values: list[float]) -> list[float]:
    values.sort()
    return [values[round((len(values) - 1) * 0.025)], values[round((len(values) - 1) * 0.975)]]


def pooled_f1(by_source: dict[str, tuple[int, int, int]]) -> float:
    totals = [0, 0, 0]
    for counts in by_source.values():
        for index, value in enumerate(counts):
            totals[index] += value
    return f1(*totals)


def bootstrap_f1_delta(
    baseline: dict[str, tuple[int, int, int]],
    treatment: dict[str, tuple[int, int, int]],
    reps: int,
    seed: int,
) -> list[float]:
    sources = sorted(set(baseline) & set(treatment))
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(reps):
        baseline_totals = [0, 0, 0]
        treatment_totals = [0, 0, 0]
        for source in (rng.choice(sources) for _ in sources):
            for index in range(3):
                baseline_totals[index] += baseline[source][index]
                treatment_totals[index] += treatment[source][index]
        draws.append(f1(*treatment_totals) - f1(*baseline_totals))
    return percentile_interval(draws)


def bootstrap_mean_delta(
    baseline: dict[str, float],
    treatment: dict[str, float],
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    sources = sorted(set(baseline) & set(treatment))
    deltas = [treatment[source] - baseline[source] for source in sources]
    point = statistics.mean(deltas)
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choice(deltas) for _ in deltas) for _ in range(reps)]
    return point, percentile_interval(draws)


def build_events(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    predictions_by_id = {str(row["instance_id"]): row for row in predictions}
    events: list[dict[str, Any]] = []
    for reference in references:
        instance_id = str(reference["instance_id"])
        prediction = predictions_by_id[instance_id]
        task = str(reference["task"])
        truth = normalize_pidqa_answer(reference.get("answer"), task)
        predicted = normalize_pidqa_answer(prediction_text(prediction), task)
        action = is_answer(prediction)
        truth_tags = set(truth or ()) if task == "value" else set()
        predicted_tags = set(predicted or ()) if task == "value" and action else set()
        events.append(
            {
                "instance_id": instance_id,
                "source_id": str(reference["source_id"]),
                "task": task,
                "correct": int(action and predicted == truth),
                "truth_tags": truth_tags,
                "predicted_tags": predicted_tags,
            }
        )
    return events


def source_accuracy(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for event in events:
        groups[event["source_id"]].append(event["correct"])
    return {key: sum(values) / len(values) for key, values in sorted(groups.items())}


def source_value_counts(events: Iterable[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for event in events:
        if event["task"] != "value":
            continue
        truth = event["truth_tags"]
        predicted = event["predicted_tags"]
        counts = groups[event["source_id"]]
        counts[0] += len(truth & predicted)
        counts[1] += len(predicted - truth)
        counts[2] += len(truth - predicted)
    return {key: tuple(values) for key, values in sorted(groups.items())}


def cell_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    task_accuracy = {
        task: sum(event["correct"] for event in events if event["task"] == task)
        / sum(event["task"] == task for event in events)
        for task in TASKS
    }
    sources = source_accuracy(events)
    counts = source_value_counts(events)
    tp = sum(value[0] for value in counts.values())
    fp = sum(value[1] for value in counts.values())
    fn = sum(value[2] for value in counts.values())
    return {
        "strict_accuracy": sum(event["correct"] for event in events) / len(events),
        "source_macro_accuracy": statistics.mean(sources.values()),
        "task_accuracy": task_accuracy,
        "value_tp": tp,
        "value_fp": fp,
        "value_fn": fn,
        "value_precision": tp / (tp + fp) if tp + fp else 0.0,
        "value_recall": tp / (tp + fn) if tp + fn else 0.0,
        "value_f1": f1(tp, fp, fn),
    }


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="data/manifests/rineng_overnight_v7_public_plan.json")
    parser.add_argument("--output-root", default="outputs/rineng_overnight_v7")
    parser.add_argument("--score", default="reports/generated/rineng_overnight_v7_score.json")
    parser.add_argument("--csv", default="reports/generated/rineng_overnight_v7_score.csv")
    parser.add_argument("--output", default="reports/generated/rineng_overnight_v7_validation.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    plan_path = root / args.plan
    score_path = root / args.score
    csv_path = root / args.csv
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    references_path = root / "data/processed/pidqa_records.jsonl"
    reference_by_id = {str(row["instance_id"]): row for row in read_jsonl(references_path)}
    errors: list[str] = []

    add_error(errors, score.get("status") == "pass", "score report status is not pass")
    add_error(errors, score.get("plan_sha256") == file_sha256(plan_path), "plan hash mismatch")
    add_error(
        errors,
        score.get("reference", {}).get("sha256") == file_sha256(references_path),
        "reference hash mismatch",
    )
    add_error(errors, score.get("bootstrap", {}).get("reps") == 10_000, "bootstrap reps are not 10000")

    events_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    expected_csv: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    row_total = 0

    for model in plan["models"]:
        model_label = str(model["model_label"])
        model_dir = root / args.output_root / model_label
        for dataset in plan["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            manifest = read_jsonl(root / dataset["correct_input"])
            expected_ids = [str(row["instance_id"]) for row in manifest]
            reference_rows = [reference_by_id[instance_id] for instance_id in expected_ids]
            for prompt in plan["prompts"]:
                prompt_id = str(prompt["prompt_id"])
                for condition in plan["conditions"]:
                    key = (model_label, dataset_id, prompt_id, str(condition))
                    cell_id = "|".join(key)
                    matches = sorted(model_dir.glob(f"{model_label}_{dataset_id}_{prompt_id}_{condition}_*.jsonl"))
                    add_error(errors, len(matches) == 1, f"{cell_id}: expected one output, found {len(matches)}")
                    if len(matches) != 1:
                        continue
                    path = matches[0]
                    predictions = read_jsonl(path)
                    prediction_ids = [str(row.get("instance_id")) for row in predictions]
                    row_total += len(predictions)
                    add_error(errors, set(prediction_ids) == set(expected_ids), f"{cell_id}: prediction ID set differs")
                    add_error(errors, len(prediction_ids) == len(set(prediction_ids)), f"{cell_id}: duplicate IDs")
                    add_error(errors, all(row.get("status") == "ok" for row in predictions), f"{cell_id}: non-ok row")
                    add_error(
                        errors,
                        all(row.get("test_answer_used") is False for row in predictions),
                        f"{cell_id}: answer isolation flag failed",
                    )
                    add_error(
                        errors,
                        all(row.get("plan_sha256") == score["plan_sha256"] for row in predictions),
                        f"{cell_id}: row plan hash mismatch",
                    )
                    report_cell = score["cells"].get(cell_id, {})
                    add_error(errors, report_cell.get("status") == "pass", f"{cell_id}: report cell not pass")
                    add_error(errors, report_cell.get("sha256") == file_sha256(path), f"{cell_id}: output hash mismatch")
                    events = build_events(reference_rows, predictions)
                    events_by_key[key] = events
                    metrics = cell_metrics(events)
                    reported = report_cell.get("metrics", {})
                    add_error(errors, close(reported.get("strict_accuracy", -1), metrics["strict_accuracy"]), f"{cell_id}: strict accuracy mismatch")
                    add_error(errors, close(reported.get("strict_source_macro_accuracy", -1), metrics["source_macro_accuracy"]), f"{cell_id}: source macro accuracy mismatch")
                    for task in TASKS:
                        actual = reported.get("task", {}).get(task, {}).get("strict_accuracy", -1)
                        add_error(errors, close(actual, metrics["task_accuracy"][task]), f"{cell_id}: {task} accuracy mismatch")
                    tag_metrics = reported.get("strict_value_tags", {})
                    for name in ("tp", "fp", "fn", "precision", "recall", "f1"):
                        expected = metrics[f"value_{name}"]
                        add_error(errors, close(tag_metrics.get(name, -1), expected), f"{cell_id}: value {name} mismatch")
                    expected_csv[key] = metrics

    comparisons = {
        (row["model"], row["dataset"], row["prompt"], row["contrast"]): row
        for row in score.get("counterfactual_comparisons", [])
    }
    comparison_index = 0
    max_comparison_error = 0.0
    for model in plan["models"]:
        model_label = str(model["model_label"])
        for dataset in plan["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            for prompt in plan["prompts"]:
                prompt_id = str(prompt["prompt_id"])
                correct_events = events_by_key[(model_label, dataset_id, prompt_id, "correct")]
                correct_counts = source_value_counts(correct_events)
                correct_accuracy = source_accuracy(correct_events)
                for control in ("shuffled", "text_only"):
                    control_events = events_by_key[(model_label, dataset_id, prompt_id, control)]
                    control_counts = source_value_counts(control_events)
                    control_accuracy = source_accuracy(control_events)
                    key = (model_label, dataset_id, prompt_id, f"correct_minus_{control}")
                    reported = comparisons[key]
                    value_delta = pooled_f1(correct_counts) - pooled_f1(control_counts)
                    value_ci = bootstrap_f1_delta(control_counts, correct_counts, 10_000, 7200 + comparison_index)
                    accuracy_delta, accuracy_ci = bootstrap_mean_delta(control_accuracy, correct_accuracy, 10_000, 8200 + comparison_index)
                    values = [
                        (reported["value_f1_difference"], value_delta, "value delta"),
                        (reported["source_macro_strict_accuracy_difference"], accuracy_delta, "accuracy delta"),
                        *zip(reported["value_f1_source_bootstrap_ci95"], value_ci, ("value CI low", "value CI high")),
                        *zip(reported["source_bootstrap_ci95"], accuracy_ci, ("accuracy CI low", "accuracy CI high")),
                    ]
                    for actual, expected, label in values:
                        difference = abs(float(actual) - float(expected))
                        max_comparison_error = max(max_comparison_error, difference)
                        add_error(errors, difference <= TOLERANCE, f"{'|'.join(key)}: {label} mismatch")
                    comparison_index += 1

    sensitivities = {
        (row["model"], row["dataset"]): row for row in score.get("prompt_sensitivity", [])
    }
    max_prompt_error = 0.0
    sensitivity_index = 0
    for model in plan["models"]:
        model_label = str(model["model_label"])
        for dataset in plan["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            p0 = source_value_counts(events_by_key[(model_label, dataset_id, "p0", "correct")])
            p1 = source_value_counts(events_by_key[(model_label, dataset_id, "p1", "correct")])
            delta = pooled_f1(p1) - pooled_f1(p0)
            interval = bootstrap_f1_delta(p0, p1, 10_000, 9200 + sensitivity_index)
            reported = sensitivities[(model_label, dataset_id)]
            values = [
                (reported["difference"], delta, "prompt delta"),
                *zip(reported["source_bootstrap_ci95"], interval, ("prompt CI low", "prompt CI high")),
            ]
            for actual, expected, label in values:
                difference = abs(float(actual) - float(expected))
                max_prompt_error = max(max_prompt_error, difference)
                add_error(errors, difference <= TOLERANCE, f"{model_label}|{dataset_id}: {label} mismatch")
            sensitivity_index += 1

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    add_error(errors, len(csv_rows) == len(expected_csv), "CSV row count mismatch")
    for row in csv_rows:
        key = (row["model"], row["dataset"], row["prompt"], row["condition"])
        expected = expected_csv[key]
        field_map = {
            "strict_accuracy": "strict_accuracy",
            "source_macro_accuracy": "source_macro_accuracy",
            "connectivity_accuracy": "task_accuracy.connectivity",
            "count_accuracy": "task_accuracy.count",
            "spatial_count_accuracy": "task_accuracy.spatial_count",
            "value_exact_accuracy": "task_accuracy.value",
            "value_tp": "value_tp",
            "value_fp": "value_fp",
            "value_fn": "value_fn",
            "value_precision": "value_precision",
            "value_recall": "value_recall",
            "value_f1": "value_f1",
        }
        for csv_field, metric_field in field_map.items():
            if metric_field.startswith("task_accuracy."):
                expected_value = expected["task_accuracy"][metric_field.split(".", 1)[1]]
            else:
                expected_value = expected[metric_field]
            add_error(errors, close(float(row[csv_field]), expected_value), f"CSV {'|'.join(key)}: {csv_field} mismatch")

    status = "pass" if not errors else "fail"
    report = {
        "version": "rineng-overnight-v7-independent-validation",
        "status": status,
        "error_count": len(errors),
        "errors": errors,
        "scope": {
            "cell_count": len(events_by_key),
            "row_count": row_total,
            "comparison_count": comparison_index,
            "prompt_sensitivity_count": sensitivity_index,
            "csv_row_count": len(csv_rows),
        },
        "checks": {
            "raw_output_hashes": "recomputed",
            "answer_isolation": "recomputed",
            "cell_metrics": "independent aggregation",
            "counterfactual_effects": "independent aggregation",
            "source_bootstrap_ci95": "independent seeded resampling",
            "prompt_sensitivity": "independent seeded resampling",
            "flat_csv": "recomputed from raw predictions",
        },
        "numeric_agreement": {
            "tolerance": TOLERANCE,
            "max_counterfactual_absolute_error": max_comparison_error,
            "max_prompt_sensitivity_absolute_error": max_prompt_error,
        },
        "inputs": {
            "plan": {"path": args.plan, "sha256": file_sha256(plan_path)},
            "score": {"path": args.score, "sha256": file_sha256(score_path)},
            "csv": {"path": args.csv, "sha256": file_sha256(csv_path)},
            "reference": {"path": "data/processed/pidqa_records.jsonl", "sha256": file_sha256(references_path)},
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "errors": len(errors), "output": args.output}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
