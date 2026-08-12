"""Dependency-free answer normalization and pilot-level scoring."""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from typing import Any, Iterable


def normalize_answer(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None and math.isfinite(number):
        return int(number) if number.is_integer() else number
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return tuple(sorted(str(item).strip() for item in parsed))
    return " ".join(lowered.split())


def score_predictions(
    records: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    record_by_id = {str(row["instance_id"]): row for row in records}
    prediction_by_id = {str(row["instance_id"]): row for row in predictions}
    per_task: dict[str, list[int]] = defaultdict(list)
    per_source: dict[str, list[int]] = defaultdict(list)
    missing = 0
    extra = sorted(set(prediction_by_id) - set(record_by_id))
    for instance_id, record in record_by_id.items():
        prediction = prediction_by_id.get(instance_id)
        if prediction is None:
            missing += 1
            correct = 0
        else:
            action = str(prediction.get("action", "ANSWER"))
            correct = int(
                action == "ANSWER"
                and normalize_answer(prediction.get("answer"))
                == normalize_answer(record.get("answer"))
            )
        per_task[str(record["task"])].append(correct)
        per_source[str(record["source_id"])].append(correct)

    def average(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    task_accuracy = {task: average(values) for task, values in sorted(per_task.items())}
    source_accuracy = {source: average(values) for source, values in sorted(per_source.items())}
    return {
        "record_count": len(record_by_id),
        "prediction_count": len(prediction_by_id),
        "missing_prediction_count": missing,
        "extra_prediction_ids": extra[:20],
        "overall_accuracy": average([item for values in per_task.values() for item in values]),
        "task_accuracy": task_accuracy,
        "source_accuracy": source_accuracy,
        "source_macro_accuracy": average(list(source_accuracy.values())),
    }
