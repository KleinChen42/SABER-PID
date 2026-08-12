"""Score the frozen cross-family F3 resolution matrix.

Predictions are scored only against the local hidden answer stores.  The
script deliberately uses the same answer normalisation and source-cluster
bootstrap as the Qwen matrix so the family comparison is on an identical
metric contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pidbench.pidqa_metrics import normalize_pidqa_answer


TASKS = ("connectivity", "count", "spatial_count", "value")
SIDES = (768, 3072)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correct(record: dict[str, Any], prediction: dict[str, Any] | None) -> int:
    if prediction is None:
        return 0
    action = str(prediction.get("action", "ANSWER"))
    if "action" not in prediction and str(prediction.get("status", "ok")) == "ok":
        action = "ANSWER"
    truth = normalize_pidqa_answer(record.get("answer"), str(record["task"]))
    pred = normalize_pidqa_answer(prediction.get("answer"), str(record["task"]))
    return int(action == "ANSWER" and pred == truth)


def tag_counts(record: dict[str, Any], prediction: dict[str, Any] | None) -> tuple[int, int, int]:
    truth = set(normalize_pidqa_answer(record.get("answer"), "value") or ())
    pred = set(normalize_pidqa_answer(prediction.get("answer"), "value") or ()) if prediction else set()
    if prediction and str(prediction.get("action", "ANSWER")) != "ANSWER":
        pred = set()
    return len(truth & pred), len(pred - truth), len(truth - pred)


def cell_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("instance_id")): row for row in predictions}
    correct_rows = {
        str(record["instance_id"]): correct(record, by_id.get(str(record["instance_id"])))
        for record in records
    }
    task_accuracy = {
        task: sum(correct_rows[str(r["instance_id"])] for r in records if str(r["task"]) == task)
        / max(1, sum(str(r["task"]) == task for r in records))
        for task in TASKS
    }
    by_source: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_id"])].append(correct_rows[str(record["instance_id"])])
    source_accuracy = {s: sum(v) / len(v) for s, v in sorted(by_source.items())}
    tp = fp = fn = 0
    for record in records:
        if str(record["task"]) != "value":
            continue
        a, b, c = tag_counts(record, by_id.get(str(record["instance_id"])))
        tp += a
        fp += b
        fn += c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "record_count": len(records),
        "prediction_count": len(by_id),
        "missing_prediction_count": len(set(str(r["instance_id"]) for r in records) - set(by_id)),
        "invalid_prediction_count": sum(str(p.get("action", "ANSWER")) != "ANSWER" for p in predictions),
        "overall_accuracy": sum(correct_rows.values()) / max(1, len(records)),
        "task_accuracy": task_accuracy,
        "source_macro_accuracy": sum(source_accuracy.values()) / max(1, len(source_accuracy)),
        "source_accuracy": source_accuracy,
        "value_tag_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "value_tag_precision": precision,
        "value_tag_recall": recall,
        "value_tag_tp": tp,
        "value_tag_fp": fp,
        "value_tag_fn": fn,
    }


def bootstrap(source_diffs: dict[str, float], reps: int, seed: int) -> tuple[float, float]:
    keys = sorted(source_diffs)
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [rng.choice(keys) for _ in keys]
        values.append(sum(source_diffs[k] for k in sample) / len(sample))
    values.sort()
    return values[round((len(values) - 1) * 0.025)], values[round((len(values) - 1) * 0.975)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--prediction-dir", default="outputs/final_replication")
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    prediction_dir = root / args.prediction_dir
    outdir = root / args.output_dir
    hidden = {
        "A": read_jsonl(root / "data/answer_store/main400_source_test_diverse_hidden.jsonl"),
        "B": read_jsonl(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl"),
    }
    cells: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []
    paths: dict[tuple[str, int], Path] = {}
    for set_id in ("A", "B"):
        for side in SIDES:
            path = prediction_dir / f"internvl35_{set_id.lower()}_p0_{side}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            paths[(set_id, side)] = path
            metrics = cell_metrics(hidden[set_id], read_jsonl(path))
            label = f"internvl35_{set_id.lower()}_p0_{side}"
            cells[label] = {
                "family": "InternVL3.5-8B",
                "set_id": set_id,
                "prompt_id": "p0",
                "max_image_side": side,
                "prediction_path": str(path.relative_to(root)).replace("\\", "/"),
                **metrics,
            }
            for task in TASKS:
                table_rows.append({
                    "family": "InternVL3.5-8B",
                    "set_id": set_id,
                    "prompt_id": "p0",
                    "max_image_side": side,
                    "task": task,
                    "accuracy": metrics["task_accuracy"][task],
                    "overall_accuracy": metrics["overall_accuracy"],
                    "source_macro_accuracy": metrics["source_macro_accuracy"],
                    "value_tag_f1": metrics["value_tag_f1"],
                    "record_count": metrics["record_count"],
                    "missing_prediction_count": metrics["missing_prediction_count"],
                    "invalid_prediction_count": metrics["invalid_prediction_count"],
                })
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "cross_family_resolution_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    comparisons = []
    for set_id in ("A", "B"):
        low = {str(r["instance_id"]): r for r in read_jsonl(paths[(set_id, 768)])}
        high = {str(r["instance_id"]): r for r in read_jsonl(paths[(set_id, 3072)])}
        source_values: dict[str, list[float]] = defaultdict(list)
        for record in hidden[set_id]:
            iid = str(record["instance_id"])
            source_values[str(record["source_id"])].append(correct(record, high.get(iid)) - correct(record, low.get(iid)))
        source_diffs = {s: sum(v) / len(v) for s, v in source_values.items()}
        low_m = cells[f"internvl35_{set_id.lower()}_p0_768"]
        high_m = cells[f"internvl35_{set_id.lower()}_p0_3072"]
        ci_low, ci_high = bootstrap(source_diffs, args.bootstrap_reps, 1701)
        comparisons.append({
            "family": "InternVL3.5-8B",
            "set_id": set_id,
            "baseline_side": 768,
            "condition_side": 3072,
            "overall_baseline_accuracy": low_m["overall_accuracy"],
            "overall_condition_accuracy": high_m["overall_accuracy"],
            "overall_difference": high_m["overall_accuracy"] - low_m["overall_accuracy"],
            "value_f1_baseline": low_m["value_tag_f1"],
            "value_f1_condition": high_m["value_tag_f1"],
            "value_f1_difference": high_m["value_tag_f1"] - low_m["value_tag_f1"],
            "task_differences": {t: high_m["task_accuracy"][t] - low_m["task_accuracy"][t] for t in TASKS},
            "source_bootstrap_ci95_low": ci_low,
            "source_bootstrap_ci95_high": ci_high,
            "source_count": len(source_diffs),
            "bootstrap_reps": args.bootstrap_reps,
            "seed": 1701,
        })
    (outdir / "cross_family_resolution_bootstrap.json").write_text(
        json.dumps({"status": "pass", "family": "InternVL3.5-8B", "cells": cells, "comparisons": comparisons,
                    "bootstrap_method": "paired source-cluster bootstrap"}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "cells": len(cells), "table": str(csv_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
