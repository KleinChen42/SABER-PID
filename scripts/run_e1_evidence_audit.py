"""Run the E1 PIDQA evidence-strengthening audit from immutable outputs.

This script does not run inference and never overwrites historical F2/F3/F5/F6
prediction files.  It creates a parallel strict-versus-semantic scoring audit,
a matched Set-B task-prior baseline, visual/output-budget inventories, and
degradation transition tables.  All reported aggregates are deterministically
recomputable from local JSONL inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pidbench.io import read_jsonl, write_json, write_jsonl
from pidbench.pidqa_metrics import normalize_pidqa_answer
from pidbench.semantic_answer_parser import SemanticParse, parse_semantic_answer


TASKS = ("connectivity", "count", "spatial_count", "value")
PROMPTS = ("p0", "p1", "p2")
SIDES = (768, 3072)


def action_is_answer(prediction: dict[str, Any] | None) -> bool:
    if prediction is None:
        return False
    if "action" in prediction:
        return str(prediction.get("action")) == "ANSWER"
    return str(prediction.get("status", "ok")) == "ok"


def prediction_text(prediction: dict[str, Any] | None) -> Any:
    if prediction is None:
        return None
    return prediction.get("answer", prediction.get("raw"))


def tag_set(value: Any) -> set[str]:
    parsed = normalize_pidqa_answer(value, "value")
    return set(parsed or ())


def event_for(record: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    task = str(record["task"])
    action = action_is_answer(prediction)
    truth_strict = normalize_pidqa_answer(record.get("answer"), task)
    pred_strict = normalize_pidqa_answer(prediction_text(prediction), task) if prediction else None
    truth_semantic = parse_semantic_answer(record.get("answer"), task)
    pred_semantic = parse_semantic_answer(prediction_text(prediction), task)
    strict_correct = int(action and pred_strict == truth_strict)
    semantic_correct = int(
        action
        and truth_semantic.parsed
        and pred_semantic.parsed
        and pred_semantic.value == truth_semantic.value
    )
    return {
        "instance_id": str(record["instance_id"]),
        "source_id": str(record["source_id"]),
        "task": task,
        "prediction_present": prediction is not None,
        "action_is_answer": action,
        "strict_correct": strict_correct,
        "semantic_correct": semantic_correct,
        "format_compliant": int(action and pred_semantic.format_compliant),
        "strict_truth": truth_strict,
        "strict_prediction": pred_strict,
        "semantic_truth": truth_semantic.value,
        "semantic_prediction": pred_semantic.value,
        "semantic_truth_parsed": truth_semantic.parsed,
        "semantic_prediction_parsed": pred_semantic.parsed,
        "semantic_parser_rule": pred_semantic.parser_rule,
        "strict_truth_tags": tag_set(record.get("answer")) if task == "value" else set(),
        "strict_prediction_tags": tag_set(prediction_text(prediction)) if prediction and action and task == "value" else set(),
        "semantic_truth_tags": set(truth_semantic.value or ()) if task == "value" and truth_semantic.parsed else set(),
        "semantic_prediction_tags": set(pred_semantic.value or ()) if task == "value" and action and pred_semantic.parsed else set(),
        "raw_output": str(prediction.get("raw", prediction_text(prediction))) if prediction else None,
        "output_token_count": prediction.get("output_token_count") if prediction else None,
    }


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def aggregate_tag_metrics(events: Iterable[dict[str, Any]], mode: str) -> dict[str, float | int]:
    truth_key = f"{mode}_truth_tags"
    pred_key = f"{mode}_prediction_tags"
    tp = fp = fn = 0
    for event in events:
        if event["task"] != "value":
            continue
        truth = set(event[truth_key])
        prediction = set(event[pred_key])
        tp += len(truth & prediction)
        fp += len(prediction - truth)
        fn += len(truth - prediction)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_from_counts(tp, fp, fn),
    }


def source_accuracy(events: Iterable[dict[str, Any]], metric: str, task: str | None = None) -> dict[str, float]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if task is None or event["task"] == task:
            by_source[event["source_id"]].append(float(event[metric]))
    return {source: sum(values) / len(values) for source, values in sorted(by_source.items())}


def source_tag_counts(events: Iterable[dict[str, Any]], mode: str) -> dict[str, tuple[int, int, int]]:
    truth_key = f"{mode}_truth_tags"
    pred_key = f"{mode}_prediction_tags"
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for event in events:
        if event["task"] != "value":
            continue
        truth = set(event[truth_key])
        prediction = set(event[pred_key])
        values = by_source[event["source_id"]]
        values[0] += len(truth & prediction)
        values[1] += len(prediction - truth)
        values[2] += len(truth - prediction)
    return {source: tuple(values) for source, values in sorted(by_source.items())}


def bootstrap_mean(values: dict[str, float], reps: int, seed: int) -> tuple[float, float]:
    ordered = [values[key] for key in sorted(values)]
    if not ordered:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        samples.append(sum(rng.choice(ordered) for _ in ordered) / len(ordered))
    samples.sort()
    return samples[round((reps - 1) * 0.025)], samples[round((reps - 1) * 0.975)]


def bootstrap_f1_difference(
    low: dict[str, tuple[int, int, int]],
    high: dict[str, tuple[int, int, int]],
    reps: int,
    seed: int,
) -> tuple[float, float]:
    sources = sorted(set(low) & set(high))
    if not sources:
        return 0.0, 0.0
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        low_counts = [0, 0, 0]
        high_counts = [0, 0, 0]
        for source in (rng.choice(sources) for _ in sources):
            for index, value in enumerate(low[source]):
                low_counts[index] += value
            for index, value in enumerate(high[source]):
                high_counts[index] += value
        values.append(
            f1_from_counts(*high_counts) - f1_from_counts(*low_counts)
        )
    values.sort()
    return values[round((reps - 1) * 0.025)], values[round((reps - 1) * 0.975)]


def evaluate(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for prediction in predictions:
        instance_id = str(prediction.get("instance_id"))
        if instance_id in by_id:
            duplicate_ids.append(instance_id)
        by_id[instance_id] = prediction
    events = [event_for(record, by_id.get(str(record["instance_id"]))) for record in records]
    record_ids = {str(record["instance_id"]) for record in records}
    task_rows: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        subset = [event for event in events if event["task"] == task]
        task_rows[task] = {
            "record_count": len(subset),
            "strict_accuracy": sum(event["strict_correct"] for event in subset) / max(1, len(subset)),
            "semantic_accuracy": sum(event["semantic_correct"] for event in subset) / max(1, len(subset)),
            "format_compliance_rate": sum(event["format_compliant"] for event in subset) / max(1, len(subset)),
            "semantic_parse_rate": sum(event["semantic_prediction_parsed"] for event in subset) / max(1, len(subset)),
            "strict_to_semantic_gain_count": sum(
                int(event["semantic_correct"] and not event["strict_correct"]) for event in subset
            ),
        }
    strict_sources = source_accuracy(events, "strict_correct")
    semantic_sources = source_accuracy(events, "semantic_correct")
    format_sources = source_accuracy(events, "format_compliant")
    return {
        "events": events,
        "metrics": {
            "record_count": len(records),
            "prediction_count": len(predictions),
            "missing_prediction_count": sum(not event["prediction_present"] for event in events),
            "extra_prediction_ids": sorted(set(by_id) - record_ids),
            "duplicate_prediction_ids": sorted(set(duplicate_ids)),
            "invalid_prediction_count": sum(not event["action_is_answer"] for event in events),
            "strict_accuracy": sum(event["strict_correct"] for event in events) / max(1, len(events)),
            "semantic_accuracy": sum(event["semantic_correct"] for event in events) / max(1, len(events)),
            "format_compliance_rate": sum(event["format_compliant"] for event in events) / max(1, len(events)),
            "semantic_parse_rate": sum(event["semantic_prediction_parsed"] for event in events) / max(1, len(events)),
            "strict_to_semantic_gain_count": sum(
                int(event["semantic_correct"] and not event["strict_correct"]) for event in events
            ),
            "task": task_rows,
            "strict_source_macro_accuracy": sum(strict_sources.values()) / max(1, len(strict_sources)),
            "semantic_source_macro_accuracy": sum(semantic_sources.values()) / max(1, len(semantic_sources)),
            "format_source_macro_rate": sum(format_sources.values()) / max(1, len(format_sources)),
            "strict_value_tags": aggregate_tag_metrics(events, "strict"),
            "semantic_value_tags": aggregate_tag_metrics(events, "semantic"),
        },
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove internal event arrays before serialising a cell."""

    return metrics


def canonical_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=list)


def render_prior_answer(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return value


def build_task_prior(
    records: list[dict[str, Any]], split_rows: list[dict[str, Any]], set_b: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_by_id = {str(row["instance_id"]): str(row["split"]) for row in split_rows}
    counters: dict[str, Counter[str]] = {task: Counter() for task in TASKS}
    representatives: dict[str, dict[str, Any]] = {task: {} for task in TASKS}
    for row in records:
        if split_by_id.get(str(row["instance_id"])) != "train":
            continue
        task = str(row["task"])
        if task not in counters:
            continue
        value = normalize_pidqa_answer(row.get("answer"), task)
        key = canonical_key(value)
        counters[task][key] += 1
        representatives[task][key] = value
    priors: dict[str, Any] = {}
    frequencies: dict[str, int] = {}
    for task in TASKS:
        chosen = min(counters[task], key=lambda key: (-counters[task][key], key))
        priors[task] = representatives[task][chosen]
        frequencies[task] = counters[task][chosen]
    predictions = [
        {
            "instance_id": str(row["instance_id"]),
            "source_id": str(row["source_id"]),
            "task": str(row["task"]),
            "action": "ANSWER",
            "answer": render_prior_answer(priors[str(row["task"])]),
            "raw": str(render_prior_answer(priors[str(row["task"])])),
            "run_id": "e1_set_b_task_prior_v2",
            "status": "ok",
            "baseline": "training_task_majority",
        }
        for row in set_b
    ]
    return predictions, {
        "seed": 17,
        "split": "source",
        "train_record_count": sum(value for counter in counters.values() for value in counter.values()),
        "task_priors": {task: render_prior_answer(priors[task]) for task in TASKS},
        "task_prior_normalized": {task: canonical_key(priors[task]) for task in TASKS},
        "task_prior_train_frequency": frequencies,
    }


def comparison_rows(
    label: str,
    baseline_events: list[dict[str, Any]],
    condition_events: list[dict[str, Any]],
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ("strict_correct", "semantic_correct", "format_compliant"):
        for task in ("overall", *TASKS):
            low = source_accuracy(baseline_events, metric, None if task == "overall" else task)
            high = source_accuracy(condition_events, metric, None if task == "overall" else task)
            sources = sorted(set(low) & set(high))
            diffs = {source: high[source] - low[source] for source in sources}
            ci_low, ci_high = bootstrap_mean(diffs, reps, seed + len(rows))
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "task": task,
                    "baseline_mean": sum(low.values()) / max(1, len(low)),
                    "condition_mean": sum(high.values()) / max(1, len(high)),
                    "difference_condition_minus_baseline": sum(diffs.values()) / max(1, len(diffs)),
                    "source_bootstrap_ci95_low": ci_low,
                    "source_bootstrap_ci95_high": ci_high,
                    "source_count": len(diffs),
                    "bootstrap_reps": reps,
                }
            )
    for mode in ("strict", "semantic"):
        low_counts = source_tag_counts(baseline_events, mode)
        high_counts = source_tag_counts(condition_events, mode)
        sources = sorted(set(low_counts) & set(high_counts))
        low_total = [sum(low_counts[source][index] for source in sources) for index in range(3)]
        high_total = [sum(high_counts[source][index] for source in sources) for index in range(3)]
        ci_low, ci_high = bootstrap_f1_difference(low_counts, high_counts, reps, seed + len(rows))
        rows.append(
            {
                "comparison": label,
                "metric": f"{mode}_value_tag_f1",
                "task": "value",
                "baseline_mean": f1_from_counts(*low_total),
                "condition_mean": f1_from_counts(*high_total),
                "difference_condition_minus_baseline": f1_from_counts(*high_total) - f1_from_counts(*low_total),
                "source_bootstrap_ci95_low": ci_low,
                "source_bootstrap_ci95_high": ci_high,
                "source_count": len(sources),
                "bootstrap_reps": reps,
            }
        )
    return rows


def cell_table_rows(group: str, label: str, metadata: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = result["metrics"]
    rows = []
    for task in TASKS:
        task_metrics = metrics["task"][task]
        rows.append(
            {
                "group": group,
                "label": label,
                "task": task,
                **metadata,
                "record_count": task_metrics["record_count"],
                "strict_accuracy": task_metrics["strict_accuracy"],
                "semantic_accuracy": task_metrics["semantic_accuracy"],
                "format_compliance_rate": task_metrics["format_compliance_rate"],
                "semantic_parse_rate": task_metrics["semantic_parse_rate"],
                "strict_to_semantic_gain_count": task_metrics["strict_to_semantic_gain_count"],
                "strict_value_tag_f1": metrics["strict_value_tags"]["f1"] if task == "value" else None,
                "semantic_value_tag_f1": metrics["semantic_value_tags"]["f1"] if task == "value" else None,
            }
        )
    return rows


def qwen_f2_paths(root: Path) -> dict[tuple[str, str, int], Path]:
    paths: dict[tuple[str, str, int], Path] = {}
    for set_id in ("A", "B"):
        for prompt in PROMPTS:
            for side in SIDES:
                if set_id == "A" and prompt == "p0":
                    paths[(set_id, prompt, side)] = root / f"outputs/main/qwen3vl8b_source400_clean_{side}.jsonl"
                else:
                    paths[(set_id, prompt, side)] = root / f"outputs/final_replication/qwen8_{set_id.lower()}_{prompt}_{side}.jsonl"
    return paths


def output_budget_rows(
    cells: dict[str, dict[str, Any]],
    rows_by_label: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for label, metadata in cells.items():
        predictions = rows_by_label[label]
        events = metadata["result"]["events"]
        event_by_id = {event["instance_id"]: event for event in events}
        per_task: dict[str, Any] = {}
        for task in TASKS:
            subset = [prediction for prediction in predictions if str(prediction.get("task")) == task]
            output_tokens = [int(prediction["output_token_count"]) for prediction in subset if prediction.get("output_token_count") is not None]
            capped = [prediction for prediction in subset if prediction.get("output_token_count") == 192]
            raw_values = [str(prediction.get("raw", prediction.get("answer", ""))) for prediction in subset]
            task_events = [event_by_id[str(prediction["instance_id"])] for prediction in subset if str(prediction["instance_id"]) in event_by_id]
            tag_cardinality = [len(event["strict_truth_tags"]) for event in task_events if task == "value"]
            per_task[task] = {
                "record_count": len(subset),
                "output_token_count_recorded": len(output_tokens),
                "output_token_mean": statistics.mean(output_tokens) if output_tokens else None,
                "output_token_median": statistics.median(output_tokens) if output_tokens else None,
                "output_token_p95": sorted(output_tokens)[round((len(output_tokens) - 1) * 0.95)] if output_tokens else None,
                "token_cap": 192 if output_tokens else None,
                "token_capped_count": len(capped),
                "token_capped_rate": len(capped) / len(subset) if subset else 0.0,
                "trailing_comma_count": sum(value.rstrip().endswith(",") for value in raw_values),
                "unbalanced_bracket_count": sum(value.count("[") != value.count("]") for value in raw_values),
                "output_character_mean": statistics.mean([len(value) for value in raw_values]) if raw_values else 0.0,
                "strict_accuracy": sum(event["strict_correct"] for event in task_events) / max(1, len(task_events)),
                "semantic_accuracy": sum(event["semantic_correct"] for event in task_events) / max(1, len(task_events)),
                "truth_value_cardinality_mean": statistics.mean(tag_cardinality) if tag_cardinality else None,
                "truth_value_cardinality_max": max(tag_cardinality) if tag_cardinality else None,
            }
            rows.append({"label": label, **{key: value for key, value in metadata.items() if key != "result"}, "task": task, **per_task[task]})
        details[label] = per_task
    return rows, details


def visual_budget_rows(root: Path, f2_predictions: dict[str, list[dict[str, Any]]], f3_predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def summarise(group: str, label: str, predictions: list[dict[str, Any]], status: str) -> None:
        sizes = [tuple(prediction["resized_image_size"]) for prediction in predictions if prediction.get("resized_image_size")]
        originals = [tuple(prediction["original_image_size"]) for prediction in predictions if prediction.get("original_image_size")]
        actual_pixels = sorted({int(prediction["input_pixel_count"]) for prediction in predictions if prediction.get("input_pixel_count") is not None})
        tiles = sorted({int(prediction["dynamic_tile_count"]) for prediction in predictions if prediction.get("dynamic_tile_count") is not None})
        pixel_proxy = [width * height * 3 for width, height in sizes]
        rows.append(
            {
                "group": group,
                "label": label,
                "record_count": len(predictions),
                "original_size_examples": json.dumps(sorted(set(originals))[:5]),
                "resized_size_examples": json.dumps(sorted(set(sizes))[:5]),
                "resized_pixel_proxy_mean": statistics.mean(pixel_proxy) if pixel_proxy else None,
                "actual_input_pixel_count_values": json.dumps(actual_pixels),
                "actual_dynamic_tile_count_values": json.dumps(tiles),
                "actual_input_budget_status": status,
            }
        )

    for label, predictions in f2_predictions.items():
        summarise("F2_Qwen", label, predictions, "not_recorded_in_f2_legacy_rows")
    for label, predictions in f3_predictions.items():
        summarise("F3_InternVL", label, predictions, "recorded_in_prediction_rows")

    telemetry = read_rows(root / "outputs/telemetry/efficiency_repeats_v2.jsonl")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in telemetry:
        if row.get("phase") == "measure":
            grouped[(str(row.get("family")), str(row.get("condition")), int(row.get("max_image_side")))].append(row)
    for (family, condition, side), predictions in sorted(grouped.items()):
        summarise("F6_telemetry", f"{family}_{condition}_{side}", predictions, "recorded_in_telemetry")
    return rows


def degradation_transitions(
    records: list[dict[str, Any]],
    prior_events: list[dict[str, Any]],
    clean_predictions: list[dict[str, Any]],
    condition_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean = evaluate(records, clean_predictions)["events"]
    clean_by_id = {event["instance_id"]: event for event in clean}
    prior_by_id = {event["instance_id"]: event for event in prior_events}
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for condition, path in sorted(condition_paths.items()):
        predictions = read_rows(path)
        events = evaluate(records, predictions)["events"]
        by_id = {event["instance_id"]: event for event in events}
        condition_detail: dict[str, Any] = {}
        for task in TASKS:
            pairs = [(clean_by_id[event["instance_id"]], event, prior_by_id[event["instance_id"]]) for event in events if event["task"] == task]
            for mode, metric in (("strict", "strict_correct"), ("semantic", "semantic_correct")):
                transition = Counter(
                    f"clean_{int(clean_event[metric])}_to_condition_{int(event[metric])}"
                    for clean_event, event, _ in pairs
                )
                prior_matches = sum(
                    event[f"{mode}_prediction"] == prior_event[f"{mode}_prediction"]
                    for _, event, prior_event in pairs
                )
                raw_lengths = [len(event["raw_output"] or "") for _, event, _ in pairs]
                row = {
                    "condition": condition,
                    "task": task,
                    "mode": mode,
                    "record_count": len(pairs),
                    "clean_correct_to_condition_wrong": transition["clean_1_to_condition_0"],
                    "clean_wrong_to_condition_correct": transition["clean_0_to_condition_1"],
                    "both_correct": transition["clean_1_to_condition_1"],
                    "both_wrong": transition["clean_0_to_condition_0"],
                    "condition_prediction_matches_task_prior_count": prior_matches,
                    "condition_prediction_matches_task_prior_rate": prior_matches / max(1, len(pairs)),
                    "condition_output_character_mean": statistics.mean(raw_lengths) if raw_lengths else 0.0,
                }
                rows.append(row)
                condition_detail[f"{task}|{mode}"] = row
        details[condition] = condition_detail
    return rows, details


def f6_semantic_cells(records_by_id: dict[str, dict[str, Any]], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    """Score each F6 repeat independently before exposing any aggregate.

    F6 deliberately repeats the same 100 instance IDs three times.  Passing all
    300 rows into the ordinary evaluator silently collapses duplicate IDs to
    their final prediction.  This helper instead preserves each repeat as a
    self-contained score cell and reports an explicit mean/range summary.
    """

    grouped: dict[tuple[str, str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in telemetry:
        if row.get("phase") == "measure":
            key = (str(row.get("family")), str(row.get("condition")), int(row.get("max_image_side")))
            grouped[key][int(row.get("repeat", 0))].append(row)
    cells: dict[str, Any] = {}
    for key, repeat_rows in sorted(grouped.items()):
        per_repeat: dict[str, dict[str, Any]] = {}
        for repeat, rows in sorted(repeat_rows.items()):
            relevant = [row for row in rows if str(row.get("instance_id")) in records_by_id]
            records = [records_by_id[str(row["instance_id"])] for row in relevant]
            per_repeat[str(repeat)] = evaluate(records, relevant)["metrics"]
        summary: dict[str, Any] = {"repeat_count": len(per_repeat), "per_repeat_metrics": per_repeat}
        for metric in ("strict_accuracy", "semantic_accuracy", "format_compliance_rate", "semantic_parse_rate"):
            values = [float(metrics[metric]) for metrics in per_repeat.values()]
            summary[f"{metric}_mean_across_repeats"] = statistics.mean(values) if values else None
            summary[f"{metric}_min_across_repeats"] = min(values) if values else None
            summary[f"{metric}_max_across_repeats"] = max(values) if values else None
        cells["|".join(map(str, key))] = summary
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    hidden_a = read_rows(root / "data/answer_store/main400_source_test_diverse_hidden.jsonl")
    hidden_b = read_rows(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl")
    hidden = {"A": hidden_a, "B": hidden_b}

    # E1.1: matched task prior.
    records = read_rows(root / "data/processed/pidqa_records.jsonl")
    split_rows = read_rows(root / "data/manifests/pidqa_source_split_seed17.jsonl")
    prior_predictions, prior_manifest = build_task_prior(records, split_rows, hidden_b)
    prior_path = root / "outputs/evidence_strengthening/set_b_task_prior_v2.jsonl"
    write_jsonl(prior_path, prior_predictions)
    prior_result = evaluate(hidden_b, prior_predictions)
    prior_payload = {
        "status": "pass",
        "description": "Source-seed-17 training-only task-majority baseline evaluated on the exact Set-B instances.",
        "manifest": prior_manifest,
        "metrics": prior_result["metrics"],
        "prediction_path": str(prior_path.relative_to(root)).replace("\\", "/"),
    }
    write_json(outdir / "set_b_task_prior_v2.json", prior_payload)

    # E1.2: score F2 and F3 with both contracts.
    semantic_cells: dict[str, Any] = {"f2": {}, "f3": {}}
    score_rows: list[dict[str, Any]] = []
    f2_results: dict[str, dict[str, Any]] = {}
    f2_predictions: dict[str, list[dict[str, Any]]] = {}
    f2_metadata: dict[str, dict[str, Any]] = {}
    for (set_id, prompt, side), path in qwen_f2_paths(root).items():
        if not path.exists():
            raise FileNotFoundError(path)
        label = f"qwen8_{set_id.lower()}_{prompt}_{side}"
        predictions = read_rows(path)
        result = evaluate(hidden[set_id], predictions)
        metadata = {"family": "Qwen3-VL-8B", "set_id": set_id, "prompt_id": prompt, "max_image_side": side, "prediction_path": str(path.relative_to(root)).replace("\\", "/")}
        semantic_cells["f2"][label] = {**metadata, "metrics": result["metrics"]}
        f2_results[label] = result
        f2_predictions[label] = predictions
        f2_metadata[label] = {**metadata, "result": result}
        score_rows.extend(cell_table_rows("F2", label, metadata, result))

    f3_results: dict[str, dict[str, Any]] = {}
    f3_predictions: dict[str, list[dict[str, Any]]] = {}
    for set_id in ("A", "B"):
        for side in SIDES:
            path = root / f"outputs/final_replication/internvl35_{set_id.lower()}_p0_{side}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            label = f"internvl35_{set_id.lower()}_p0_{side}"
            predictions = read_rows(path)
            result = evaluate(hidden[set_id], predictions)
            metadata = {"family": "InternVL3.5-8B", "set_id": set_id, "prompt_id": "p0", "max_image_side": side, "prediction_path": str(path.relative_to(root)).replace("\\", "/")}
            semantic_cells["f3"][label] = {**metadata, "metrics": result["metrics"]}
            f3_results[label] = result
            f3_predictions[label] = predictions
            score_rows.extend(cell_table_rows("F3", label, metadata, result))

    score_rows.extend(cell_table_rows("E1_prior", "set_b_training_task_prior", {"family": "non_visual_task_prior", "set_id": "B", "prompt_id": "none", "max_image_side": None, "prediction_path": str(prior_path.relative_to(root)).replace("\\", "/")}, prior_result))

    # F5 and F6 are also re-scored with the semantic parser.
    f5_clean_path = root / "outputs/final_degradation/qwen8_set_b_clean.jsonl"
    f5_clean_predictions = read_rows(f5_clean_path)
    f5_clean_result = evaluate(hidden_b, f5_clean_predictions)
    semantic_cells["f5_clean"] = {"prediction_path": str(f5_clean_path.relative_to(root)).replace("\\", "/"), "metrics": f5_clean_result["metrics"]}
    score_rows.extend(cell_table_rows("F5", "qwen8_set_b_clean", {"family": "Qwen3-VL-8B", "set_id": "B", "prompt_id": "p0", "max_image_side": 1536, "prediction_path": str(f5_clean_path.relative_to(root)).replace("\\", "/")}, f5_clean_result))
    degradation_paths: dict[str, Path] = {}
    for path in sorted((root / "outputs/final_degradation").glob("qwen8_set_b_*.jsonl")):
        condition = path.stem.removeprefix("qwen8_set_b_")
        if condition == "clean":
            continue
        degradation_paths[condition] = path
        result = evaluate(hidden_b, read_rows(path))
        semantic_cells.setdefault("f5", {})[condition] = {"prediction_path": str(path.relative_to(root)).replace("\\", "/"), "metrics": result["metrics"]}
        score_rows.extend(cell_table_rows("F5", f"qwen8_set_b_{condition}", {"family": "Qwen3-VL-8B", "set_id": "B", "prompt_id": "p0", "max_image_side": 1536, "prediction_path": str(path.relative_to(root)).replace("\\", "/")}, result))

    telemetry = read_rows(root / "outputs/telemetry/efficiency_repeats_v2.jsonl")
    semantic_cells["f6"] = f6_semantic_cells({str(row["instance_id"]): row for row in hidden_b}, telemetry)

    semantic_payload = {
        "status": "pass",
        "parser_contract": {
            "strict": "Existing pidbench.pidqa_metrics normalization, retained unchanged.",
            "semantic": "Conservative task parser: boolean prefix, one integer token, and frozen P&ID tag grammar.",
            "raw_output_mutated": False,
        },
        "cells": semantic_cells,
        "f6_measurement_cells_are_repeated_rows": True,
        "f6_scored_per_repeat_before_aggregation": True,
    }
    write_json(outdir / "semantic_scoring_audit_v1.json", semantic_payload)
    write_csv(outdir / "strict_semantic_score_table_v1.csv", score_rows)

    # Pairwise task-level comparisons for F2/F3.
    comparison_rows_all: list[dict[str, Any]] = []
    for set_index, set_id in enumerate(("A", "B")):
        for prompt_index, prompt in enumerate(PROMPTS):
            low = f2_results[f"qwen8_{set_id.lower()}_{prompt}_768"]["events"]
            high = f2_results[f"qwen8_{set_id.lower()}_{prompt}_3072"]["events"]
            comparison_rows_all.extend(comparison_rows(f"F2_Qwen8_{set_id}_{prompt}_3072_minus_768", low, high, args.bootstrap_reps, 1000 + 100 * set_index + 10 * prompt_index))
        low = f3_results[f"internvl35_{set_id.lower()}_p0_768"]["events"]
        high = f3_results[f"internvl35_{set_id.lower()}_p0_3072"]["events"]
        comparison_rows_all.extend(comparison_rows(f"F3_InternVL_{set_id}_p0_3072_minus_768", low, high, args.bootstrap_reps, 2000 + 100 * set_index))
    write_json(outdir / "task_level_bootstrap_v2.json", {"status": "pass", "bootstrap_method": "paired source-cluster bootstrap", "bootstrap_reps": args.bootstrap_reps, "comparisons": comparison_rows_all})

    # Matched B task-prior comparisons.
    prior_comparisons: list[dict[str, Any]] = []
    for label, result in sorted(f2_results.items()):
        if label.startswith("qwen8_b_"):
            prior_comparisons.extend(comparison_rows(f"{label}_minus_set_b_task_prior", prior_result["events"], result["events"], args.bootstrap_reps, 3000 + len(prior_comparisons)))
    for label, result in sorted(f3_results.items()):
        if label.startswith("internvl35_b_"):
            prior_comparisons.extend(comparison_rows(f"{label}_minus_set_b_task_prior", prior_result["events"], result["events"], args.bootstrap_reps, 4000 + len(prior_comparisons)))
    write_json(outdir / "set_b_model_vs_prior_bootstrap_v2.json", {"status": "pass", "bootstrap_method": "paired source-cluster bootstrap", "bootstrap_reps": args.bootstrap_reps, "comparisons": prior_comparisons})
    write_csv(outdir / "set_b_model_vs_prior_table_v2.csv", prior_comparisons)

    # E1.3 visual budget inventory.
    visual_rows = visual_budget_rows(root, f2_predictions, f3_predictions)
    write_csv(outdir / "effective_visual_budget_audit_v1.csv", visual_rows)
    write_json(outdir / "effective_visual_budget_audit_v1.json", {"status": "pass", "rows": visual_rows, "interpretation": "Rows without actual processor pixels retain only a resized-image proxy and are explicitly labelled not_recorded."})

    # E1.4 output budget audit uses F2, which has per-output token telemetry.
    output_rows, output_detail = output_budget_rows(f2_metadata, f2_predictions)
    write_csv(outdir / "output_budget_by_task_v1.csv", output_rows)
    write_json(outdir / "output_budget_audit_v1.json", {"status": "pass", "token_cap_contract": {"f2_max_new_tokens": 192}, "cells": output_detail})

    # E1.5 degradation transitions relative to the same matched task prior.
    transition_rows, transition_details = degradation_transitions(hidden_b, prior_result["events"], f5_clean_predictions, degradation_paths)
    write_csv(outdir / "degradation_transition_table_v1.csv", transition_rows)
    write_json(outdir / "degradation_transition_analysis_v1.json", {"status": "pass", "baseline": "qwen8_set_b_clean", "task_prior": prior_payload["manifest"]["task_priors"], "conditions": transition_details})

    validation = {
        "status": "pass",
        "checks": {
            "prior_prediction_count": len(prior_predictions) == len(hidden_b),
            "f2_cell_count": len(f2_results) == 12,
            "f3_cell_count": len(f3_results) == 4,
            "f5_condition_count": len(degradation_paths) == 9,
            "semantic_table_nonempty": bool(score_rows),
            "visual_budget_rows_nonempty": bool(visual_rows),
            "output_budget_rows_nonempty": bool(output_rows),
            "transition_rows_nonempty": bool(transition_rows),
        },
    }
    validation["status"] = "pass" if all(validation["checks"].values()) else "fail"
    write_json(outdir / "e1_evidence_audit_validation_v1.json", validation)
    print(json.dumps({"status": validation["status"], "f2_cells": len(f2_results), "f3_cells": len(f3_results), "f5_conditions": len(degradation_paths), "output_dir": str(outdir)}, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
