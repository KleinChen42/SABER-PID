"""Score the answer-blind F2 set/prompt/resolution matrix.

The scorer reads hidden answers locally and treats direct-mode rows with
``status=ok`` as ANSWER.  It reports exact accuracy, value-task tag F1,
task/source summaries, and paired source-cluster bootstrap differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pidbench.pidqa_metrics import normalize_pidqa_answer


TASKS = ("connectivity", "count", "spatial_count", "value")
PROMPTS = ("p0", "p1", "p2")
SIDES = (768, 3072)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def correct(record: dict[str, Any], prediction: dict[str, Any] | None) -> int:
    if prediction is None:
        return 0
    action = str(prediction.get("action", "ANSWER"))
    if "action" not in prediction and str(prediction.get("status", "ok")) == "ok":
        action = "ANSWER"
    return int(action == "ANSWER" and normalize_pidqa_answer(prediction.get("answer"), str(record["task"])) == normalize_pidqa_answer(record.get("answer"), str(record["task"])))


def tag_counts(record: dict[str, Any], prediction: dict[str, Any] | None) -> tuple[int, int, int]:
    truth = set(normalize_pidqa_answer(record.get("answer"), "value") or ())
    pred = set(normalize_pidqa_answer(prediction.get("answer"), "value") or ()) if prediction else set()
    if prediction and "action" in prediction and str(prediction.get("action")) != "ANSWER":
        pred = set()
    return len(truth & pred), len(pred - truth), len(truth - pred)


def source_bootstrap(
    source_diffs: dict[str, float], reps: int = 10000, seed: int = 17
) -> tuple[float, float]:
    sources = sorted(source_diffs)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(reps):
        sampled = [rng.choice(sources) for _ in sources]
        values.append(sum(source_diffs[source] for source in sampled) / len(sampled))
    values.sort()
    return values[round((len(values) - 1) * 0.025)], values[round((len(values) - 1) * 0.975)]


def cell_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["instance_id"]): row for row in predictions}
    correct_rows = {str(record["instance_id"]): correct(record, by_id.get(str(record["instance_id"]))) for record in records}
    task_accuracy = {
        task: sum(correct_rows[str(record["instance_id"])] for record in records if str(record["task"]) == task) / max(1, sum(str(record["task"]) == task for record in records))
        for task in TASKS
    }
    by_source: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_id"])].append(correct_rows[str(record["instance_id"])])
    source_accuracy = {source: sum(values) / len(values) for source, values in sorted(by_source.items())}
    value_records = [record for record in records if str(record["task"]) == "value"]
    tp = fp = fn = 0
    for record in value_records:
        a, b, c = tag_counts(record, by_id.get(str(record["instance_id"])))
        tp += a; fp += b; fn += c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "record_count": len(records),
        "prediction_count": len(by_id),
        "missing_prediction_count": len(set(str(row["instance_id"]) for row in records) - set(by_id)),
        "overall_accuracy": sum(correct_rows.values()) / max(1, len(records)),
        "task_accuracy": task_accuracy,
        "source_macro_accuracy": sum(source_accuracy.values()) / max(1, len(source_accuracy)),
        "source_accuracy": source_accuracy,
        "value_exact_set_accuracy": task_accuracy["value"],
        "value_tag_tp": tp,
        "value_tag_fp": fp,
        "value_tag_fn": fn,
        "value_tag_precision": precision,
        "value_tag_recall": recall,
        "value_tag_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--prediction-dir", default="outputs/final_replication")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    outdir = root / args.output_dir
    pred_dir = root / args.prediction_dir
    hidden = {
        "A": read_jsonl(root / "data/answer_store/main400_source_test_diverse_hidden.jsonl"),
        "B": read_jsonl(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl"),
    }
    paths: dict[tuple[str, str, int], Path] = {}
    for set_id in ("A", "B"):
        for prompt_id in PROMPTS:
            for side in SIDES:
                if set_id == "A" and prompt_id == "p0":
                    paths[(set_id, prompt_id, side)] = root / f"outputs/main/qwen3vl8b_source400_clean_{side}.jsonl"
                else:
                    paths[(set_id, prompt_id, side)] = pred_dir / f"qwen8_{set_id.lower()}_{prompt_id}_{side}.jsonl"
    cells: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for (set_id, prompt_id, side), path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        metrics = cell_metrics(hidden[set_id], read_jsonl(path))
        label = f"qwen8_{set_id.lower()}_{prompt_id}_{side}"
        cells[label] = {"set_id": set_id, "prompt_id": prompt_id, "max_image_side": side, "prediction_path": str(path.relative_to(root)).replace("\\", "/"), **metrics}
        for task in TASKS:
            rows.append({"set_id": set_id, "prompt_id": prompt_id, "max_image_side": side, "label": label, "task": task, "accuracy": metrics["task_accuracy"][task], "overall_accuracy": metrics["overall_accuracy"], "source_macro_accuracy": metrics["source_macro_accuracy"], "value_tag_f1": metrics["value_tag_f1"], "record_count": metrics["record_count"], "missing_prediction_count": metrics["missing_prediction_count"]})
    csv_path = outdir / "qwen8_selection_prompt_resolution_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(outdir / "qwen8_selection_prompt_resolution_matrix.json", {"status": "pass", "cells": cells, "cell_count": len(cells), "records_per_cell": {key: value["record_count"] for key, value in cells.items()}})

    bootstrap_rows: list[dict[str, Any]] = []
    for set_id in ("A", "B"):
        records = hidden[set_id]
        for prompt_id in PROMPTS:
            low_path = paths[(set_id, prompt_id, 768)]
            high_path = paths[(set_id, prompt_id, 3072)]
            low = {str(row["instance_id"]): row for row in read_jsonl(low_path)}
            high = {str(row["instance_id"]): row for row in read_jsonl(high_path)}
            by_source: dict[str, list[float]] = defaultdict(list)
            for record in records:
                instance_id = str(record["instance_id"])
                by_source[str(record["source_id"])].append(correct(record, high.get(instance_id)) - correct(record, low.get(instance_id)))
            source_diffs = {source: sum(values) / len(values) for source, values in by_source.items()}
            low_metrics = cells[f"qwen8_{set_id.lower()}_{prompt_id}_768"]
            high_metrics = cells[f"qwen8_{set_id.lower()}_{prompt_id}_3072"]
            ci_low, ci_high = source_bootstrap(source_diffs, args.bootstrap_reps, 17)
            row = {"set_id": set_id, "prompt_id": prompt_id, "baseline_side": 768, "condition_side": 3072, "overall_baseline_accuracy": low_metrics["overall_accuracy"], "overall_condition_accuracy": high_metrics["overall_accuracy"], "overall_difference": high_metrics["overall_accuracy"] - low_metrics["overall_accuracy"], "value_f1_baseline": low_metrics["value_tag_f1"], "value_f1_condition": high_metrics["value_tag_f1"], "value_f1_difference": high_metrics["value_tag_f1"] - low_metrics["value_tag_f1"], "source_bootstrap_ci95_low": ci_low, "source_bootstrap_ci95_high": ci_high, "source_count": len(source_diffs), "bootstrap_reps": args.bootstrap_reps, "seed": 17}
            row["task_differences"] = {task: high_metrics["task_accuracy"][task] - low_metrics["task_accuracy"][task] for task in TASKS}
            bootstrap_rows.append(row)
    write_json(outdir / "qwen8_selection_prompt_resolution_bootstrap.json", {"status": "pass", "comparisons": bootstrap_rows, "bootstrap_method": "paired source-cluster bootstrap", "bootstrap_reps": args.bootstrap_reps, "seed": 17})
    print(json.dumps({"status": "pass", "cell_count": len(cells), "comparisons": len(bootstrap_rows), "matrix": str(csv_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
