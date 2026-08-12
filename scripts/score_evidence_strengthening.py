"""Score E2--E5 evidence-strengthening experiments from immutable JSONL.

Inference runs only against answer-isolated manifests.  This post-hoc scorer is
the sole component that reads the local hidden Set-B store.  It retains strict
and deterministic semantic metrics side by side and uses paired source-cluster
bootstrap confidence intervals for every declared condition comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from run_e1_evidence_audit import TASKS, comparison_rows, evaluate, read_rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["row_type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[round((len(values) - 1) * fraction)]


def runtime_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in predictions if str(row.get("status", "ok")) == "ok"]
    tokens = [float(row["output_token_count"]) for row in successful if row.get("output_token_count") is not None]
    latency = [float(row["latency_seconds"]) for row in successful if row.get("latency_seconds") is not None]
    peaks = [float(row["peak_allocated_bytes"]) for row in successful if row.get("peak_allocated_bytes") is not None]
    caps = [
        bool(row.get("output_reached_max_new_tokens", row.get("output_token_count") == row.get("max_new_tokens")))
        for row in successful
        if row.get("output_token_count") is not None and row.get("max_new_tokens") is not None
    ]
    return {
        "prediction_row_count": len(predictions),
        "ok_row_count": len(successful),
        "error_row_count": len(predictions) - len(successful),
        "output_token_recorded_count": len(tokens),
        "output_token_mean": statistics.mean(tokens) if tokens else None,
        "output_token_median": statistics.median(tokens) if tokens else None,
        "output_token_p95": percentile(tokens, 0.95),
        "token_cap_observed_count": sum(caps),
        "token_cap_observed_rate": sum(caps) / len(caps) if caps else None,
        "latency_recorded_count": len(latency),
        "latency_seconds_mean": statistics.mean(latency) if latency else None,
        "latency_seconds_median": statistics.median(latency) if latency else None,
        "latency_seconds_p95": percentile(latency, 0.95),
        "peak_allocated_bytes_max": max(peaks) if peaks else None,
        "peak_allocated_bytes_mean": statistics.mean(peaks) if peaks else None,
        "actual_input_pixel_count_values": sorted({int(row["actual_input_pixel_count"]) for row in successful if row.get("actual_input_pixel_count") is not None}),
        "input_pixel_count_values": sorted({int(row["input_pixel_count"]) for row in successful if row.get("input_pixel_count") is not None}),
        "dynamic_tile_count_values": sorted({int(row["dynamic_tile_count"]) for row in successful if row.get("dynamic_tile_count") is not None}),
        "input_image_count_values": sorted({int(row["input_image_count"]) for row in successful if row.get("input_image_count") is not None}),
        "max_new_tokens_values": sorted({int(row["max_new_tokens"]) for row in successful if row.get("max_new_tokens") is not None}),
    }


def records_for_tasks(records: list[dict[str, Any]], tasks: set[str] | None) -> list[dict[str, Any]]:
    return [row for row in records if tasks is None or str(row["task"]) in tasks]


def predictions_for_tasks(predictions: list[dict[str, Any]], tasks: set[str] | None) -> list[dict[str, Any]]:
    return [row for row in predictions if tasks is None or str(row.get("task")) in tasks]


def score_cell(
    *, label: str, records: list[dict[str, Any]], predictions: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    result = evaluate(records, predictions)
    return {
        "label": label,
        "metadata": metadata,
        "metrics": result["metrics"],
        "runtime": runtime_summary(predictions),
        "events": result["events"],
    }


def cell_table(cell: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = cell["metrics"]
    for task in TASKS:
        task_metrics = metrics["task"][task]
        if not task_metrics["record_count"]:
            continue
        rows.append(
            {
                "row_type": "cell",
                "label": cell["label"],
                **cell["metadata"],
                "task": task,
                "record_count": task_metrics["record_count"],
                "strict_accuracy": task_metrics["strict_accuracy"],
                "semantic_accuracy": task_metrics["semantic_accuracy"],
                "format_compliance_rate": task_metrics["format_compliance_rate"],
                "semantic_parse_rate": task_metrics["semantic_parse_rate"],
                "strict_to_semantic_gain_count": task_metrics["strict_to_semantic_gain_count"],
                "strict_value_tag_f1": metrics["strict_value_tags"]["f1"] if task == "value" else None,
                "semantic_value_tag_f1": metrics["semantic_value_tags"]["f1"] if task == "value" else None,
                "token_cap_observed_rate": cell["runtime"]["token_cap_observed_rate"],
                "output_token_mean": cell["runtime"]["output_token_mean"],
                "latency_seconds_mean": cell["runtime"]["latency_seconds_mean"],
                "peak_allocated_bytes_max": cell["runtime"]["peak_allocated_bytes_max"],
                "actual_input_pixel_count_values": json.dumps(cell["runtime"]["actual_input_pixel_count_values"]),
                "input_pixel_count_values": json.dumps(cell["runtime"]["input_pixel_count_values"]),
                "dynamic_tile_count_values": json.dumps(cell["runtime"]["dynamic_tile_count_values"]),
            }
        )
    return rows


def paired_transitions(label: str, baseline_events: list[dict[str, Any]], condition_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_id = {event["instance_id"]: event for event in baseline_events}
    condition_by_id = {event["instance_id"]: event for event in condition_events}
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for mode, metric in (("strict", "strict_correct"), ("semantic", "semantic_correct")):
            pairs = [
                (baseline_by_id[instance_id], condition_by_id[instance_id])
                for instance_id in sorted(set(baseline_by_id) & set(condition_by_id))
                if baseline_by_id[instance_id]["task"] == task
            ]
            if not pairs:
                continue
            counts = Counter(f"baseline_{int(left[metric])}_to_condition_{int(right[metric])}" for left, right in pairs)
            rows.append(
                {
                    "comparison": label,
                    "task": task,
                    "mode": mode,
                    "record_count": len(pairs),
                    "both_correct": counts["baseline_1_to_condition_1"],
                    "both_wrong": counts["baseline_0_to_condition_0"],
                    "baseline_correct_to_condition_wrong": counts["baseline_1_to_condition_0"],
                    "baseline_wrong_to_condition_correct": counts["baseline_0_to_condition_1"],
                }
            )
    return rows


def add_comparison(
    comparisons: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    label: str,
    baseline: dict[str, Any],
    condition: dict[str, Any],
    reps: int,
    seed: int,
) -> None:
    comparisons.extend(
        row
        for row in comparison_rows(label, baseline["events"], condition["events"], reps, seed)
        if int(row["source_count"]) > 0
    )
    transitions.extend(paired_transitions(label, baseline["events"], condition["events"]))


def e2(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    value_records = records_for_tasks(records, {"value"})
    base_dir = root / "outputs/final_replication"
    new_dir = root / "outputs/evidence_strengthening/qwen8_value_budget_v1"
    cells = {
        "qwen8_b_p0_value_192_768": score_cell(label="qwen8_b_p0_value_192_768", records=value_records, predictions=predictions_for_tasks(read_rows(base_dir / "qwen8_b_p0_768.jsonl"), {"value"}), metadata={"experiment": "E2", "condition": "correct_image", "side": 768, "max_new_tokens": 192}),
        "qwen8_b_p0_value_192_3072": score_cell(label="qwen8_b_p0_value_192_3072", records=value_records, predictions=predictions_for_tasks(read_rows(base_dir / "qwen8_b_p0_3072.jsonl"), {"value"}), metadata={"experiment": "E2", "condition": "correct_image", "side": 3072, "max_new_tokens": 192}),
        "qwen8_b_p0_value_512_768": score_cell(label="qwen8_b_p0_value_512_768", records=value_records, predictions=read_rows(new_dir / "qwen8_value_budget_v1_768.jsonl"), metadata={"experiment": "E2", "condition": "correct_image", "side": 768, "max_new_tokens": 512}),
        "qwen8_b_p0_value_512_3072": score_cell(label="qwen8_b_p0_value_512_3072", records=value_records, predictions=read_rows(new_dir / "qwen8_value_budget_v1_3072.jsonl"), metadata={"experiment": "E2", "condition": "correct_image", "side": 3072, "max_new_tokens": 512}),
    }
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    add_comparison(comparisons, transitions, "e2_512_minus_192_768", cells["qwen8_b_p0_value_192_768"], cells["qwen8_b_p0_value_512_768"], reps, 2101)
    add_comparison(comparisons, transitions, "e2_512_minus_192_3072", cells["qwen8_b_p0_value_192_3072"], cells["qwen8_b_p0_value_512_3072"], reps, 2102)
    add_comparison(comparisons, transitions, "e2_512_3072_minus_768", cells["qwen8_b_p0_value_512_768"], cells["qwen8_b_p0_value_512_3072"], reps, 2103)
    add_comparison(comparisons, transitions, "e2_192_3072_minus_768_reference", cells["qwen8_b_p0_value_192_768"], cells["qwen8_b_p0_value_192_3072"], reps, 2104)
    return "qwen8_value_budget_sensitivity_v1", {"experiment": "E2", "status": "pass", "records": {"task_filter": "value", "record_count": len(value_records)}, "cells": {key: {field: value for field, value in cell.items() if field != "events"} for key, cell in cells.items()}, "comparisons": comparisons, "paired_transitions": transitions, "bootstrap_reps": reps}


def e3(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    base_dir = root / "outputs/final_replication"
    new_dir = root / "outputs/evidence_strengthening/qwen8_image_shuffle_v1"
    cells: dict[str, dict[str, Any]] = {}
    for side in (768, 3072):
        cells[f"qwen8_b_p0_correct_{side}"] = score_cell(label=f"qwen8_b_p0_correct_{side}", records=records, predictions=read_rows(base_dir / f"qwen8_b_p0_{side}.jsonl"), metadata={"experiment": "E3", "condition": "correct_image", "side": side, "max_new_tokens": 192})
        cells[f"qwen8_b_p0_source_shuffled_{side}"] = score_cell(label=f"qwen8_b_p0_source_shuffled_{side}", records=records, predictions=read_rows(new_dir / f"qwen8_image_shuffle_v1_{side}.jsonl"), metadata={"experiment": "E3", "condition": "source_shuffled_no_fixed_point", "side": side, "max_new_tokens": 192})
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for side, seed in ((768, 2301), (3072, 2302)):
        add_comparison(comparisons, transitions, f"e3_shuffled_minus_correct_{side}", cells[f"qwen8_b_p0_correct_{side}"], cells[f"qwen8_b_p0_source_shuffled_{side}"], reps, seed)
    return "image_dependence_control_v1", {"experiment": "E3", "status": "pass", "records": {"task_filter": "all", "record_count": len(records)}, "cells": {key: {field: value for field, value in cell.items() if field != "events"} for key, cell in cells.items()}, "comparisons": comparisons, "paired_transitions": transitions, "bootstrap_reps": reps}


def e4(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    directory = root / "outputs/evidence_strengthening/internvl_tile_budget_v1"
    low = score_cell(label="internvl35_b_tile_low", records=records, predictions=read_rows(directory / "internvl35_b_tile_low.jsonl"), metadata={"experiment": "E4", "condition": "actual_tile_budget_low", "dynamic_preprocess_max_num": 1})
    high = score_cell(label="internvl35_b_tile_high", records=records, predictions=read_rows(directory / "internvl35_b_tile_high.jsonl"), metadata={"experiment": "E4", "condition": "actual_tile_budget_high", "dynamic_preprocess_max_num": 12})
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    add_comparison(comparisons, transitions, "e4_high_minus_low_actual_tile_budget", low, high, reps, 2401)
    cells = {"internvl35_b_tile_low": low, "internvl35_b_tile_high": high}
    return "internvl_tile_budget_v1", {"experiment": "E4", "status": "pass", "records": {"task_filter": "all", "record_count": len(records)}, "cells": {key: {field: value for field, value in cell.items() if field != "events"} for key, cell in cells.items()}, "comparisons": comparisons, "paired_transitions": transitions, "bootstrap_reps": reps}


def e5(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    base_dir = root / "outputs/final_replication"
    new_dir = root / "outputs/evidence_strengthening/qwen8_ontology_visible_v1"
    cells: dict[str, dict[str, Any]] = {}
    for side in (768, 3072):
        cells[f"qwen8_b_p0_raw_{side}"] = score_cell(label=f"qwen8_b_p0_raw_{side}", records=records, predictions=read_rows(base_dir / f"qwen8_b_p0_{side}.jsonl"), metadata={"experiment": "E5", "condition": "raw_image_only", "side": side, "max_new_tokens": 192})
        cells[f"qwen8_b_p0_ontology_visible_{side}"] = score_cell(label=f"qwen8_b_p0_ontology_visible_{side}", records=records, predictions=read_rows(new_dir / f"qwen8_ontology_visible_v1_{side}.jsonl"), metadata={"experiment": "E5", "condition": "public_training_symbol_prototype_legend", "side": side, "max_new_tokens": 192})
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for side, seed in ((768, 2501), (3072, 2502)):
        add_comparison(comparisons, transitions, f"e5_ontology_visible_minus_raw_{side}", cells[f"qwen8_b_p0_raw_{side}"], cells[f"qwen8_b_p0_ontology_visible_{side}"], reps, seed)
    return "ontology_visibility_effect_v1", {"experiment": "E5", "status": "pass", "records": {"task_filter": "all", "record_count": len(records)}, "cells": {key: {field: value for field, value in cell.items() if field != "events"} for key, cell in cells.items()}, "comparisons": comparisons, "paired_transitions": transitions, "bootstrap_reps": reps}


def e6(root: Path, records: list[dict[str, Any]], reps: int) -> tuple[str, dict[str, Any]]:
    """Score the pre-specified source-seed sensitivity without pooling seeds."""

    del records
    cells: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    seed_summaries: dict[str, dict[str, Any]] = {}
    for seed, bootstrap_seed in ((29, 2601), (31, 2602)):
        seed_records = read_rows(root / f"data/answer_store/source_seed{seed}_resolution_v1_hidden.jsonl")
        directory = root / f"outputs/evidence_strengthening/qwen8_source_seed{seed}_resolution_v1"
        low_key = f"qwen8_seed{seed}_p0_768"
        high_key = f"qwen8_seed{seed}_p0_3072"
        cells[low_key] = score_cell(
            label=low_key,
            records=seed_records,
            predictions=read_rows(directory / f"qwen8_source_seed{seed}_resolution_v1_768.jsonl"),
            metadata={"experiment": "E6", "source_split_seed": seed, "side": 768, "max_new_tokens": 192},
        )
        cells[high_key] = score_cell(
            label=high_key,
            records=seed_records,
            predictions=read_rows(directory / f"qwen8_source_seed{seed}_resolution_v1_3072.jsonl"),
            metadata={"experiment": "E6", "source_split_seed": seed, "side": 3072, "max_new_tokens": 192},
        )
        add_comparison(
            comparisons,
            transitions,
            f"e6_seed{seed}_3072_minus_768",
            cells[low_key],
            cells[high_key],
            reps,
            bootstrap_seed,
        )
        seed_summaries[str(seed)] = {"record_count": len(seed_records), "source_count": len({str(row["source_id"]) for row in seed_records})}
    return "source_seed_resolution_sensitivity_v1", {
        "experiment": "E6",
        "status": "pass",
        "source_split_sensitivity": seed_summaries,
        "cells": {key: {field: value for field, value in cell.items() if field != "events"} for key, cell in cells.items()},
        "comparisons": comparisons,
        "paired_transitions": transitions,
        "bootstrap_reps": reps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--experiment", required=True, choices=("e2", "e3", "e4", "e5", "e6"))
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    records = read_rows(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl")
    builders = {"e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}
    name, payload = builders[args.experiment](root, records, args.bootstrap_reps)
    outdir = root / args.output_dir
    write_json(outdir / f"{name}.json", payload)

    table_rows: list[dict[str, Any]] = []
    for cell in payload["cells"].values():
        table_rows.extend(cell_table({**cell, "events": []}))
    for row in payload["comparisons"]:
        table_rows.append({"row_type": "comparison", **row})
    for row in payload["paired_transitions"]:
        table_rows.append({"row_type": "paired_transition", **row})
    write_csv(outdir / f"{name}.csv", table_rows)
    print(json.dumps({"status": "pass", "experiment": payload["experiment"], "report": str(outdir / f"{name}.json"), "cell_count": len(payload["cells"]), "comparison_count": len(payload["comparisons"])}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
