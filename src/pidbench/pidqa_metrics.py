"""Task-aware normalization and scoring for the heterogeneous PIDQA answers."""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from typing import Any, Iterable


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) if isinstance(value, float) and value.is_integer() else value
    text = " ".join(str(value).strip().split())
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        number = float(text)
    except ValueError:
        return lowered
    return int(number) if math.isfinite(number) and number.is_integer() else number


def normalize_pidqa_answer(value: Any, task: str) -> Any:
    """Normalize answer form using the known PIDQA task schema.

    ``value`` questions have set/list semantics. Models often produce a plain
    comma-separated tag list while the released labels use Python-list syntax;
    those equivalent forms need to compare as the same set.
    """

    if task != "value":
        return _scalar(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if text in {"", "[]"}:
            return ()
        parsed = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, (list, tuple, set)):
            parts = [str(item).strip() for item in parsed]
        else:
            parts = [part.strip().strip("'\"") for part in text.split(",")]
    return tuple(sorted(part.lower() for part in parts if part))


def score_pidqa_predictions(
    records: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    record_by_id = {str(row["instance_id"]): row for row in records}
    prediction_by_id = {str(row["instance_id"]): row for row in predictions}
    per_task: dict[str, list[int]] = defaultdict(list)
    per_source: dict[str, list[int]] = defaultdict(list)
    missing = 0
    for instance_id, record in record_by_id.items():
        task = str(record["task"])
        prediction = prediction_by_id.get(instance_id)
        if prediction is None:
            missing += 1
            correct = 0
        else:
            correct = int(
                str(prediction.get("action", "ANSWER")) == "ANSWER"
                and normalize_pidqa_answer(prediction.get("answer"), task)
                == normalize_pidqa_answer(record.get("answer"), task)
            )
        per_task[task].append(correct)
        per_source[str(record["source_id"])].append(correct)

    def average(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    task_accuracy = {task: average(values) for task, values in sorted(per_task.items())}
    source_accuracy = {source: average(values) for source, values in sorted(per_source.items())}
    return {
        "record_count": len(record_by_id),
        "prediction_count": len(prediction_by_id),
        "missing_prediction_count": missing,
        "extra_prediction_ids": sorted(set(prediction_by_id) - set(record_by_id))[:20],
        "overall_accuracy": average([item for values in per_task.values() for item in values]),
        "task_accuracy": task_accuracy,
        "source_accuracy": source_accuracy,
        "source_macro_accuracy": average(list(source_accuracy.values())),
    }
