"""Build the no-new-inference analyses requested by the RINENG v2 review.

The script uses immutable Qwen and OCR outputs plus scorer-only references. It
adds source-macro sensitivities, per-source workload, pooled uncertainty,
Set-B-excluded and strictly source-disjoint frozen-rule checks, OCR join
ablation, and deterministic error/complementarity summaries. It does not run
model inference or tune a prediction rule against reference answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from build_positive_narrative_hybrid_analysis_v5 import RULES, answer_set, fuse, render
from pidbench.pidqa_metrics import normalize_pidqa_answer
from run_e1_evidence_audit import evaluate, read_rows
from score_editorial_extension_experiments_v4 import geometry_joined_ocr_prediction


VERSION = "rineng-revision-analysis-v6"
DEFAULT_REPS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[round((len(ordered) - 1) * fraction)]


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def precision_from_counts(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def recall_from_counts(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn else 0.0


def per_source_metrics(truth: set[str], prediction: set[str]) -> dict[str, float | int]:
    """Return deterministic set metrics for one source.

    PIDQA value references in the analysed cells are non-empty. For a general
    empty-reference/empty-prediction record, precision, recall, and F1 are set
    to 1 because the set is exactly recovered. An empty prediction against a
    non-empty reference receives zero precision, recall, and F1.
    """

    tp = len(truth & prediction)
    fp = len(prediction - truth)
    fn = len(truth - prediction)
    if not truth and not prediction:
        precision = recall = f1 = 1.0
    else:
        precision = precision_from_counts(tp, fp)
        recall = recall_from_counts(tp, fn)
        f1 = f1_from_counts(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": int(truth == prediction),
        "truth_count": len(truth),
        "candidate_count": len(prediction),
        "false_candidate_count": fp,
        "recovered_tag_count": tp,
        "missed_tag_count": fn,
    }


def error_pattern(metrics: dict[str, float | int]) -> str:
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    if fp == 0 and fn == 0:
        return "exact_set"
    if int(metrics["candidate_count"]) == 0:
        return "empty_prediction"
    if tp == 0 and fp > 0:
        return "false_candidates_without_recovery"
    if tp > 0 and fn > 0 and fp == 0:
        return "partial_recovery_no_false_candidate"
    if tp > 0 and fn == 0 and fp > 0:
        return "complete_recovery_with_false_candidates"
    if tp > 0 and fn > 0 and fp > 0:
        return "partial_recovery_with_false_candidates"
    if tp == 0 and fp == 0 and fn > 0:
        return "empty_prediction"
    return "other"


def iqr_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def bootstrap_mean_ci(values: dict[str, float], reps: int, seed: int) -> list[float]:
    sources = sorted(values)
    if not sources:
        return [0.0, 0.0]
    observed = np.asarray([values[source] for source in sources], dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(sources), size=(reps, len(sources)))
    distribution = observed[indexes].mean(axis=1)
    distribution.sort()
    return [
        float(distribution[round((reps - 1) * 0.025)]),
        float(distribution[round((reps - 1) * 0.975)]),
    ]


def bootstrap_pooled_ci(
    rows: dict[str, dict[str, Any]], metric: str, reps: int, seed: int
) -> list[float]:
    sources = sorted(rows)
    if not sources:
        return [0.0, 0.0]
    counts = np.asarray(
        [
            [rows[source]["tp"], rows[source]["fp"], rows[source]["fn"], rows[source]["exact"]]
            for source in sources
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(sources), size=(reps, len(sources)))
    totals = counts[indexes].sum(axis=1)
    tp, fp, fn, exact = (totals[:, index] for index in range(4))
    if metric == "precision":
        denominator = tp + fp
        distribution = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    elif metric == "recall":
        denominator = tp + fn
        distribution = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    elif metric == "f1":
        denominator = 2 * tp + fp + fn
        distribution = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    elif metric == "exact":
        distribution = exact / len(sources)
    else:
        raise KeyError(metric)
    distribution.sort()
    return [
        float(distribution[round((reps - 1) * 0.025)]),
        float(distribution[round((reps - 1) * 0.975)]),
    ]


def paired_bootstrap_difference(
    baseline: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
    estimator: str,
    metric: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    paired_sources = sorted(set(baseline) & set(condition))
    if set(paired_sources) != set(baseline) or set(paired_sources) != set(condition):
        raise ValueError("Paired comparison source membership mismatch")
    if estimator == "source_macro_nonempty_reference":
        sources = [
            source
            for source in paired_sources
            if int(baseline[source]["truth_count"]) > 0
            and int(condition[source]["truth_count"]) > 0
        ]
    else:
        sources = paired_sources

    def estimate(rows: list[dict[str, Any]]) -> float:
        if estimator in {"source_macro", "source_macro_nonempty_reference"}:
            return sum(float(row[metric]) for row in rows) / len(rows)
        if estimator != "micro_pooled":
            raise KeyError(estimator)
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        if metric == "precision":
            return precision_from_counts(tp, fp)
        if metric == "recall":
            return recall_from_counts(tp, fn)
        if metric == "f1":
            return f1_from_counts(tp, fp, fn)
        if metric == "exact":
            return sum(int(row["exact"]) for row in rows) / len(rows)
        raise KeyError(metric)

    baseline_rows = [baseline[source] for source in sources]
    condition_rows = [condition[source] for source in sources]
    baseline_point = estimate(baseline_rows)
    condition_point = estimate(condition_rows)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(sources), size=(reps, len(sources)))
    if estimator in {"source_macro", "source_macro_nonempty_reference"}:
        observed_difference = np.asarray(
            [float(condition[source][metric]) - float(baseline[source][metric]) for source in sources],
            dtype=float,
        )
        distribution = observed_difference[indexes].mean(axis=1)
    else:
        baseline_counts = np.asarray(
            [
                [baseline[source]["tp"], baseline[source]["fp"], baseline[source]["fn"], baseline[source]["exact"]]
                for source in sources
            ],
            dtype=float,
        )
        condition_counts = np.asarray(
            [
                [condition[source]["tp"], condition[source]["fp"], condition[source]["fn"], condition[source]["exact"]]
                for source in sources
            ],
            dtype=float,
        )
        baseline_totals = baseline_counts[indexes].sum(axis=1)
        condition_totals = condition_counts[indexes].sum(axis=1)

        def vector_metric(totals: np.ndarray) -> np.ndarray:
            tp, fp, fn, exact = (totals[:, index] for index in range(4))
            if metric == "precision":
                denominator = tp + fp
                return np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
            if metric == "recall":
                denominator = tp + fn
                return np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
            if metric == "f1":
                denominator = 2 * tp + fp + fn
                return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
            if metric == "exact":
                return exact / len(sources)
            raise KeyError(metric)

        distribution = vector_metric(condition_totals) - vector_metric(baseline_totals)
    distribution.sort()
    return {
        "estimator": estimator,
        "metric": metric,
        "source_count": len(sources),
        "baseline": baseline_point,
        "condition": condition_point,
        "difference_condition_minus_baseline": condition_point - baseline_point,
        "source_bootstrap_ci95": [
            float(distribution[round((reps - 1) * 0.025)]),
            float(distribution[round((reps - 1) * 0.975)]),
        ],
        "bootstrap_reps": reps,
        "bootstrap_interval": "percentile",
    }


def prediction_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {str(row["instance_id"]): answer_set(row) for row in rows}


def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["instance_id"]): row for row in records}
    if len(result) != len(records):
        raise ValueError("Duplicate record instance_id")
    return result


def build_fusion_predictions(
    qwen_rows: list[dict[str, Any]], ocr_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    qwen = {str(row["instance_id"]): row for row in qwen_rows}
    ocr = {str(row["instance_id"]): row for row in ocr_rows}
    if set(qwen) != set(ocr):
        raise ValueError("Qwen/OCR membership mismatch")
    predictions: dict[str, list[dict[str, Any]]] = {
        "qwen": list(qwen_rows),
        "paddleocr_geometry": list(ocr_rows),
    }
    for rule in RULES:
        fused: list[dict[str, Any]] = []
        for instance_id in sorted(qwen):
            tags = fuse(answer_set(qwen[instance_id]), answer_set(ocr[instance_id]), rule)
            fused.append(
                {
                    "instance_id": instance_id,
                    "source_id": str(qwen[instance_id]["source_id"]),
                    "task": "value",
                    "action": "ANSWER",
                    "answer": render(tags),
                    "raw": render(tags),
                    "status": "ok",
                    "fusion_rule": rule,
                    "test_answer_used": False,
                }
            )
        predictions[rule] = fused
    return predictions


def validate_prediction_rows(rows: list[dict[str, Any]], label: str) -> None:
    ids = [str(row.get("instance_id")) for row in rows]
    failures: list[str] = []
    if len(ids) != len(set(ids)):
        failures.append("duplicate_instance_id")
    if any(str(row.get("task")) != "value" for row in rows):
        failures.append("non_value_task")
    if any(str(row.get("status", "ok")) != "ok" for row in rows):
        failures.append("non_ok_status")
    if any(row.get("test_answer_used") is True for row in rows):
        failures.append("test_answer_used_true")
    if any({"reference_answer", "truth", "cypher"} & set(row) for row in rows):
        failures.append("reference_field_in_prediction")
    if failures:
        raise ValueError(f"Invalid {label}: {', '.join(failures)}")


def analyse_dataset(
    *,
    label: str,
    records: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    reps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    record_index = records_by_id(records)
    expected_ids = set(record_index)
    source_ids = {str(row["source_id"]) for row in records}
    if len(records) != len(source_ids):
        raise ValueError(f"{label}: expected one value record per source")
    if any(str(row.get("task")) != "value" for row in records):
        raise ValueError(f"{label}: non-value scorer record")
    empty_reference_count = sum(
        not set(normalize_pidqa_answer(row.get("answer"), "value") or ()) for row in records
    )
    per_source_rows: list[dict[str, Any]] = []
    per_method: dict[str, dict[str, dict[str, Any]]] = {}
    methods: dict[str, Any] = {}

    for method_index, (method, rows) in enumerate(predictions.items()):
        validate_prediction_rows(rows, f"{label}/{method}")
        by_id = {str(row["instance_id"]): row for row in rows}
        if set(by_id) != expected_ids:
            raise ValueError(f"{label}/{method}: prediction membership mismatch")
        scored = evaluate(records, rows)
        metrics = scored["metrics"]
        source_rows: dict[str, dict[str, Any]] = {}
        for instance_id in sorted(expected_ids):
            record = record_index[instance_id]
            prediction = by_id[instance_id]
            truth = set(normalize_pidqa_answer(record.get("answer"), "value") or ())
            predicted = answer_set(prediction)
            values = per_source_metrics(truth, predicted)
            source_id = str(record["source_id"])
            row = {
                "dataset": label,
                "method": method,
                "instance_id": instance_id,
                "source_id": source_id,
                **values,
                "error_pattern": error_pattern(values),
                "truth_tags": ";".join(sorted(truth)),
                "prediction_tags": ";".join(sorted(predicted)),
            }
            source_rows[source_id] = row
            per_source_rows.append(row)
        per_method[method] = source_rows

        pooled = metrics["strict_value_tags"]
        source_macro: dict[str, Any] = {}
        source_macro_nonempty: dict[str, Any] = {}
        nonempty_source_rows = {
            source: row for source, row in source_rows.items() if int(row["truth_count"]) > 0
        }
        for metric_index, metric in enumerate(("precision", "recall", "f1", "exact")):
            values = [float(row[metric]) for row in source_rows.values()]
            source_macro[metric] = {
                **iqr_summary(values),
                "source_bootstrap_ci95": bootstrap_mean_ci(
                    {source: float(row[metric]) for source, row in source_rows.items()},
                    reps,
                    seed + 100 * method_index + metric_index,
                ),
            }
            nonempty_values = [float(row[metric]) for row in nonempty_source_rows.values()]
            source_macro_nonempty[metric] = {
                **iqr_summary(nonempty_values),
                "source_bootstrap_ci95": bootstrap_mean_ci(
                    {
                        source: float(row[metric])
                        for source, row in nonempty_source_rows.items()
                    },
                    reps,
                    seed + 5000 + 100 * method_index + metric_index,
                ),
            }
        workload = {
            metric: iqr_summary([float(row[metric]) for row in source_rows.values()])
            for metric in (
                "truth_count",
                "candidate_count",
                "recovered_tag_count",
                "false_candidate_count",
                "missed_tag_count",
            )
        }
        workload.update(
            {
                "empty_prediction_source_count": sum(
                    int(row["candidate_count"]) == 0 for row in source_rows.values()
                ),
                "source_with_false_candidate_count": sum(
                    int(row["false_candidate_count"]) > 0 for row in source_rows.values()
                ),
                "source_with_any_recovery_count": sum(
                    int(row["recovered_tag_count"]) > 0 for row in source_rows.values()
                ),
                "exact_source_count": sum(int(row["exact"]) for row in source_rows.values()),
            }
        )
        methods[method] = {
            "record_count": len(records),
            "source_count": len(source_rows),
            "micro_pooled": {
                "tp": int(pooled["tp"]),
                "fp": int(pooled["fp"]),
                "fn": int(pooled["fn"]),
                "precision": float(pooled["precision"]),
                "recall": float(pooled["recall"]),
                "f1": float(pooled["f1"]),
                "exact": float(metrics["task"]["value"]["strict_accuracy"]),
                "source_bootstrap_ci95": {
                    metric: bootstrap_pooled_ci(
                        source_rows,
                        metric,
                        reps,
                        seed + 1000 + 100 * method_index + metric_index,
                    )
                    for metric_index, metric in enumerate(("precision", "recall", "f1", "exact"))
                },
            },
            "source_macro": source_macro,
            "source_macro_nonempty_reference": source_macro_nonempty,
            "workload_per_source": workload,
            "error_pattern_source_counts": dict(
                sorted(Counter(str(row["error_pattern"]) for row in source_rows.values()).items())
            ),
        }

    requested = (
        ("union_minus_qwen", "qwen", "set_union"),
        ("union_minus_ocr", "paddleocr_geometry", "set_union"),
        ("intersection_minus_qwen", "qwen", "set_intersection"),
        ("ocr_first_minus_ocr", "paddleocr_geometry", "ocr_if_nonempty_else_qwen"),
    )
    comparisons: list[dict[str, Any]] = []
    for comparison_index, (name, baseline, condition) in enumerate(requested):
        if baseline not in per_method or condition not in per_method:
            continue
        for estimator_index, estimator in enumerate(
            ("micro_pooled", "source_macro", "source_macro_nonempty_reference")
        ):
            for metric_index, metric in enumerate(("precision", "recall", "f1", "exact")):
                comparisons.append(
                    {
                        "comparison": name,
                        **paired_bootstrap_difference(
                            per_method[baseline],
                            per_method[condition],
                            estimator,
                            metric,
                            reps,
                            seed
                            + 10_000
                            + 1000 * comparison_index
                            + 100 * estimator_index
                            + metric_index,
                        ),
                    }
                )
    return (
        {
            "label": label,
            "status": "pass",
            "record_count": len(records),
            "source_count": len(source_ids),
            "empty_reference_count": empty_reference_count,
            "empty_set_convention": (
                "empty truth plus empty prediction receives per-source P/R/F1=1; "
                "empty prediction against non-empty truth receives zero"
            ),
            "methods": methods,
            "comparisons": comparisons,
        },
        per_source_rows,
        per_method,
    )


def filter_rows_by_sources(rows: list[dict[str, Any]], sources: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("source_id")) in sources]


def compact_tag(tag: str) -> str:
    return "".join(character for character in tag.lower() if character.isalnum())


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_character in enumerate(left, start=1):
        current = [index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def string_error_diagnostic(
    per_method: dict[str, dict[str, dict[str, Any]]], methods: Iterable[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for source_id, source in sorted(per_method[method].items()):
            truth = set(filter(None, str(source["truth_tags"]).split(";")))
            prediction = set(filter(None, str(source["prediction_tags"]).split(";")))
            missed = sorted(truth - prediction)
            false = sorted(prediction - truth)
            available = set(false)
            for truth_tag in missed:
                best: tuple[int, str, str] | None = None
                left = compact_tag(truth_tag)
                for false_tag in sorted(available):
                    right = compact_tag(false_tag)
                    if not left or not right:
                        continue
                    if left.startswith(right) or right.startswith(left):
                        category = "prefix_or_suffix_fragment"
                        distance = abs(len(left) - len(right))
                    else:
                        distance = edit_distance(left, right)
                        category = "single_character_edit" if distance == 1 else "other_string_difference"
                    candidate = (distance, category, false_tag)
                    if best is None or candidate < best:
                        best = candidate
                if best is None:
                    rows.append(
                        {
                            "method": method,
                            "source_id": source_id,
                            "missed_tag": truth_tag,
                            "false_candidate": "",
                            "diagnostic": "unpaired_miss",
                            "compact_edit_distance": "",
                        }
                    )
                    continue
                distance, category, false_tag = best
                if category == "other_string_difference" and distance > 2:
                    rows.append(
                        {
                            "method": method,
                            "source_id": source_id,
                            "missed_tag": truth_tag,
                            "false_candidate": "",
                            "diagnostic": "unpaired_miss",
                            "compact_edit_distance": "",
                        }
                    )
                    continue
                available.remove(false_tag)
                rows.append(
                    {
                        "method": method,
                        "source_id": source_id,
                        "missed_tag": truth_tag,
                        "false_candidate": false_tag,
                        "diagnostic": category,
                        "compact_edit_distance": distance,
                    }
                )
            for false_tag in sorted(available):
                rows.append(
                    {
                        "method": method,
                        "source_id": source_id,
                        "missed_tag": "",
                        "false_candidate": false_tag,
                        "diagnostic": "unpaired_false_candidate",
                        "compact_edit_distance": "",
                    }
                )
    return rows


def complementarity_summary(
    qwen_rows: dict[str, dict[str, Any]], ocr_rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(qwen_rows) != set(ocr_rows):
        raise ValueError("Complementarity source mismatch")
    true_tags = Counter()
    false_candidates = Counter()
    source_patterns = Counter()
    for source_id in sorted(qwen_rows):
        qwen = qwen_rows[source_id]
        ocr = ocr_rows[source_id]
        truth = set(filter(None, str(qwen["truth_tags"]).split(";")))
        qpred = set(filter(None, str(qwen["prediction_tags"]).split(";")))
        opred = set(filter(None, str(ocr["prediction_tags"]).split(";")))
        if truth != set(filter(None, str(ocr["truth_tags"]).split(";"))):
            raise ValueError("Complementarity truth mismatch")
        for tag in truth:
            qhit = tag in qpred
            ohit = tag in opred
            true_tags[
                "recovered_by_both"
                if qhit and ohit
                else "qwen_only"
                if qhit
                else "ocr_only"
                if ohit
                else "recovered_by_neither"
            ] += 1
        for tag in (qpred | opred) - truth:
            qfalse = tag in qpred
            ofalse = tag in opred
            false_candidates[
                "false_in_both"
                if qfalse and ofalse
                else "qwen_only_false"
                if qfalse
                else "ocr_only_false"
            ] += 1
        q_only_recovery = len((truth & qpred) - opred)
        o_only_recovery = len((truth & opred) - qpred)
        source_patterns[
            "both_methods_add_unique_true_tags"
            if q_only_recovery and o_only_recovery
            else "qwen_adds_unique_true_tags"
            if q_only_recovery
            else "ocr_adds_unique_true_tags"
            if o_only_recovery
            else "no_unique_true_tag_by_method"
        ] += 1
    return {
        "true_tag_recovery_counts": dict(sorted(true_tags.items())),
        "false_candidate_overlap_counts": dict(sorted(false_candidates.items())),
        "source_complementarity_counts": dict(sorted(source_patterns.items())),
    }


def deterministic_boundary_case_gallery(
    per_method: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Select one auditable Qwen success, partial, and failure case.

    Selection is the minimum SHA-256 key within each predeclared outcome
    stratum. It is deterministic but outcome-conditioned, so the returned
    cases are boundary illustrations rather than representative samples.
    """

    required_methods = (
        "qwen",
        "paddleocr_geometry",
        "set_union",
        "set_intersection",
    )
    missing = [method for method in required_methods if method not in per_method]
    if missing:
        raise KeyError("Boundary gallery method missing: " + ", ".join(missing))
    if any(set(per_method[method]) != set(per_method["qwen"]) for method in required_methods):
        raise ValueError("Boundary gallery source membership mismatch")

    strata = (
        ("success", "exact_set"),
        ("partial", "partial_recovery_with_false_candidates"),
        ("failure", "false_candidates_without_recovery"),
    )
    cases: list[dict[str, Any]] = []
    for label, pattern in strata:
        eligible = [
            row
            for row in per_method["qwen"].values()
            if str(row["error_pattern"]) == pattern
        ]
        if not eligible:
            raise ValueError(f"No eligible Qwen boundary case for {label}/{pattern}")

        def selection_key(row: dict[str, Any]) -> tuple[str, str]:
            source_id = str(row["source_id"])
            value = f"{VERSION}|boundary-gallery|{label}|{source_id}".encode("utf-8")
            return hashlib.sha256(value).hexdigest(), source_id

        selected = min(eligible, key=selection_key)
        source_id = str(selected["source_id"])
        method_rows: dict[str, Any] = {}
        for method in required_methods:
            row = per_method[method][source_id]
            method_rows[method] = {
                "prediction_tags": list(
                    filter(None, str(row["prediction_tags"]).split(";"))
                ),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "error_pattern": str(row["error_pattern"]),
            }
        cases.append(
            {
                "case": label,
                "qwen_outcome_stratum": pattern,
                "eligible_source_count": len(eligible),
                "selection_rule": (
                    "minimum SHA-256 of version|boundary-gallery|case|source_id "
                    "within the predeclared Qwen outcome stratum"
                ),
                "selection_sha256": selection_key(selected)[0],
                "source_id": source_id,
                "instance_id": str(selected["instance_id"]),
                "reference_tags": list(
                    filter(None, str(selected["truth_tags"]).split(";"))
                ),
                "methods": method_rows,
            }
        )
    return cases


def ocr_join_audit(
    records: list[dict[str, Any]],
    literal_rows: list[dict[str, Any]],
    joined_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    records_index = records_by_id(records)
    literal = prediction_sets(literal_rows)
    joined = prediction_sets(joined_rows)
    if set(literal) != set(joined) or set(literal) != set(records_index):
        raise ValueError("OCR join audit membership mismatch")
    counts = Counter()
    source_effects = Counter()
    geometry_joined_candidate_count = 0
    for instance_id in sorted(literal):
        truth = set(normalize_pidqa_answer(records_index[instance_id].get("answer"), "value") or ())
        before = literal[instance_id]
        after = joined[instance_id]
        added = after - before
        removed = before - after
        counts["added_true_tags"] += len(added & truth)
        counts["added_false_candidates"] += len(added - truth)
        counts["removed_true_tags"] += len(removed & truth)
        counts["removed_false_candidates"] += len(removed - truth)
        before_f1 = float(per_source_metrics(truth, before)["f1"])
        after_f1 = float(per_source_metrics(truth, after)["f1"])
        source_effects[
            "improved_source"
            if after_f1 > before_f1
            else "degraded_source"
            if after_f1 < before_f1
            else "unchanged_source"
        ] += 1
        joined_row = next(row for row in joined_rows if str(row["instance_id"]) == instance_id)
        geometry_joined_candidate_count += len(joined_row.get("geometry_joined_candidates", []))
    return {
        "tag_transition_counts": dict(sorted(counts.items())),
        "source_f1_direction_counts": dict(sorted(source_effects.items())),
        "geometry_join_operation_count": geometry_joined_candidate_count,
        "rule": "reference-free vertical prefix/suffix join v1",
    }


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["row_type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_csv_rows(datasets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_label, dataset in datasets.items():
        for method, values in dataset["methods"].items():
            micro = values["micro_pooled"]
            macro = values["source_macro"]
            macro_nonempty = values["source_macro_nonempty_reference"]
            workload = values["workload_per_source"]
            rows.append(
                {
                    "row_type": "method",
                    "dataset": dataset_label,
                    "method": method,
                    "source_count": dataset["source_count"],
                    "micro_tp": micro["tp"],
                    "micro_fp": micro["fp"],
                    "micro_fn": micro["fn"],
                    "micro_precision": micro["precision"],
                    "micro_recall": micro["recall"],
                    "micro_f1": micro["f1"],
                    "micro_exact": micro["exact"],
                    "micro_precision_ci_low": micro["source_bootstrap_ci95"]["precision"][0],
                    "micro_precision_ci_high": micro["source_bootstrap_ci95"]["precision"][1],
                    "micro_f1_ci_low": micro["source_bootstrap_ci95"]["f1"][0],
                    "micro_f1_ci_high": micro["source_bootstrap_ci95"]["f1"][1],
                    "source_macro_precision": macro["precision"]["mean"],
                    "source_macro_recall": macro["recall"]["mean"],
                    "source_macro_f1": macro["f1"]["mean"],
                    "source_macro_f1_ci_low": macro["f1"]["source_bootstrap_ci95"][0],
                    "source_macro_f1_ci_high": macro["f1"]["source_bootstrap_ci95"][1],
                    "source_macro_nonempty_reference_f1": macro_nonempty["f1"]["mean"],
                    "source_macro_nonempty_reference_f1_ci_low": macro_nonempty["f1"]["source_bootstrap_ci95"][0],
                    "source_macro_nonempty_reference_f1_ci_high": macro_nonempty["f1"]["source_bootstrap_ci95"][1],
                    "candidate_median": workload["candidate_count"]["median"],
                    "candidate_q1": workload["candidate_count"]["q1"],
                    "candidate_q3": workload["candidate_count"]["q3"],
                    "false_candidate_median": workload["false_candidate_count"]["median"],
                    "false_candidate_q1": workload["false_candidate_count"]["q1"],
                    "false_candidate_q3": workload["false_candidate_count"]["q3"],
                    "empty_prediction_sources": workload["empty_prediction_source_count"],
                }
            )
        for comparison in dataset["comparisons"]:
            if comparison["metric"] not in {"f1", "precision", "recall"}:
                continue
            rows.append(
                {
                    "row_type": "comparison",
                    "dataset": dataset_label,
                    "method": comparison["comparison"],
                    "source_count": comparison["source_count"],
                    "estimator": comparison["estimator"],
                    "metric": comparison["metric"],
                    "baseline": comparison["baseline"],
                    "condition": comparison["condition"],
                    "difference": comparison["difference_condition_minus_baseline"],
                    "difference_ci_low": comparison["source_bootstrap_ci95"][0],
                    "difference_ci_high": comparison["source_bootstrap_ci95"][1],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument(
        "--output", default="reports/generated/rineng_revision_analysis_v6.json"
    )
    parser.add_argument(
        "--summary-csv", default="reports/generated/rineng_revision_analysis_v6.csv"
    )
    parser.add_argument(
        "--per-source-csv", default="reports/generated/rineng_revision_per_source_v6.csv"
    )
    parser.add_argument(
        "--error-csv", default="reports/generated/rineng_revision_error_taxonomy_v6.csv"
    )
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    root = Path(args.root).resolve()

    records_path = root / "data/processed/pidqa_records.jsonl"
    all_records = read_rows(records_path)
    all_records_by_id = records_by_id(all_records)
    qwen_set_b_path = root / "outputs/final_replication/qwen8_b_p0_3072.jsonl"
    ocr_set_b_path = (
        root
        / "outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl"
    )
    qwen_set_b = [
        row for row in read_rows(qwen_set_b_path) if str(row.get("task")) == "value"
    ]
    ocr_set_b_literal = read_rows(ocr_set_b_path)
    ocr_set_b_joined = [geometry_joined_ocr_prediction(row) for row in ocr_set_b_literal]
    validate_prediction_rows(qwen_set_b, "set_b/qwen")
    validate_prediction_rows(ocr_set_b_literal, "set_b/ocr_literal")
    set_b_ids = {str(row["instance_id"]) for row in qwen_set_b}
    set_b_sources = {str(row["source_id"]) for row in qwen_set_b}
    set_b_records = [all_records_by_id[instance_id] for instance_id in sorted(set_b_ids)]
    if len(set_b_records) != 100 or len(set_b_sources) != 100:
        raise ValueError("Expected 100 Set B value records/sources")

    set_b_predictions = build_fusion_predictions(qwen_set_b, ocr_set_b_joined)
    set_b_predictions["paddleocr_literal"] = ocr_set_b_literal
    datasets: dict[str, dict[str, Any]] = {}
    all_per_source_rows: list[dict[str, Any]] = []
    per_method_by_dataset: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    dataset, per_source, per_method = analyse_dataset(
        label="set_b",
        records=set_b_records,
        predictions=set_b_predictions,
        reps=args.bootstrap_reps,
        seed=6100,
    )
    datasets["set_b"] = dataset
    all_per_source_rows.extend(per_source)
    per_method_by_dataset["set_b"] = per_method

    seed_payload: dict[int, dict[str, Any]] = {}
    seed_sources: dict[int, set[str]] = {}
    for seed in (29, 31):
        qwen_path = (
            root
            / f"outputs/evidence_strengthening/qwen8_source_seed{seed}_resolution_v1/qwen8_source_seed{seed}_resolution_v1_3072.jsonl"
        )
        ocr_path = root / f"outputs/positive_narrative/paddleocr_seed{seed}_v1.jsonl"
        qwen = [row for row in read_rows(qwen_path) if str(row.get("task")) == "value"]
        ocr_raw = read_rows(ocr_path)
        ocr_joined = [geometry_joined_ocr_prediction(row) for row in ocr_raw]
        validate_prediction_rows(qwen, f"seed{seed}/qwen")
        validate_prediction_rows(ocr_raw, f"seed{seed}/ocr")
        seed_sources[seed] = {str(row["source_id"]) for row in qwen}
        seed_payload[seed] = {
            "qwen": qwen,
            "ocr": ocr_joined,
            "qwen_path": qwen_path,
            "ocr_path": ocr_path,
        }
        if len(seed_sources[seed]) != 100:
            raise ValueError(f"Expected 100 sources in seed {seed}")

    after_set_b_overlap = (seed_sources[29] - set_b_sources) & (
        seed_sources[31] - set_b_sources
    )
    subsets = {
        "seed29_excluding_set_b": seed_sources[29] - set_b_sources,
        "seed31_excluding_set_b": seed_sources[31] - set_b_sources,
        "seed29_strictly_disjoint": seed_sources[29] - set_b_sources - seed_sources[31],
        "seed31_strictly_disjoint": seed_sources[31] - set_b_sources - seed_sources[29],
    }
    expected_counts = {
        "seed29_excluding_set_b": 83,
        "seed31_excluding_set_b": 83,
        "seed29_strictly_disjoint": 65,
        "seed31_strictly_disjoint": 65,
    }
    for offset, (label, sources) in enumerate(subsets.items()):
        if len(sources) != expected_counts[label]:
            raise ValueError(f"Unexpected {label} source count: {len(sources)}")
        seed = 29 if "seed29" in label else 31
        qwen = filter_rows_by_sources(seed_payload[seed]["qwen"], sources)
        ocr = filter_rows_by_sources(seed_payload[seed]["ocr"], sources)
        ids = {str(row["instance_id"]) for row in qwen}
        records = [all_records_by_id[instance_id] for instance_id in sorted(ids)]
        predictions = build_fusion_predictions(qwen, ocr)
        dataset, per_source, per_method = analyse_dataset(
            label=label,
            records=records,
            predictions=predictions,
            reps=args.bootstrap_reps,
            seed=6200 + 100 * offset,
        )
        datasets[label] = dataset
        all_per_source_rows.extend(per_source)
        per_method_by_dataset[label] = per_method

    complementarity = complementarity_summary(
        per_method_by_dataset["set_b"]["qwen"],
        per_method_by_dataset["set_b"]["paddleocr_geometry"],
    )
    boundary_cases = deterministic_boundary_case_gallery(
        per_method_by_dataset["set_b"]
    )
    join_audit = ocr_join_audit(set_b_records, ocr_set_b_literal, ocr_set_b_joined)
    string_errors = string_error_diagnostic(
        per_method_by_dataset["set_b"],
        ("qwen", "paddleocr_literal", "paddleocr_geometry"),
    )
    error_summary = {
        method: dict(
            sorted(
                Counter(
                    str(row["error_pattern"])
                    for row in per_method_by_dataset["set_b"][method].values()
                ).items()
            )
        )
        for method in (
            "qwen",
            "paddleocr_literal",
            "paddleocr_geometry",
            "set_union",
            "set_intersection",
            "ocr_if_nonempty_else_qwen",
        )
    }
    string_error_summary = {
        method: dict(
            sorted(
                Counter(
                    str(row["diagnostic"])
                    for row in string_errors
                    if row["method"] == method
                ).items()
            )
        )
        for method in ("qwen", "paddleocr_literal", "paddleocr_geometry")
    }

    sources = [
        qwen_set_b_path,
        ocr_set_b_path,
        records_path,
        seed_payload[29]["qwen_path"],
        seed_payload[29]["ocr_path"],
        seed_payload[31]["qwen_path"],
        seed_payload[31]["ocr_path"],
    ]
    payload = {
        "version": VERSION,
        "status": "pass",
        "analysis_role": (
            "no-new-inference RINENG revision analysis; scorer-only references are used "
            "for evaluation and never for prediction construction"
        ),
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "interval": "95% percentile source bootstrap",
            "unit": "source_id",
            "pairing": "paired source resampling for method differences",
            "rng": "numpy.random.default_rng (PCG64), fixed per-analysis seeds",
        },
        "estimator_definitions": {
            "micro_pooled": "sum TP/FP/FN over sources, then compute P/R/F1",
            "source_macro": "compute P/R/F1 per source, then average sources equally",
            "source_macro_nonempty_reference": (
                "compute P/R/F1 per source and average only sources with at least one reference tag; "
                "reported as an empty-reference sensitivity"
            ),
            "exact": "exact normalized set equality per source",
        },
        "membership": {
            "set_b_source_count": len(set_b_sources),
            "seed29_source_count": len(seed_sources[29]),
            "seed31_source_count": len(seed_sources[31]),
            "set_b_seed29_overlap": len(set_b_sources & seed_sources[29]),
            "set_b_seed31_overlap": len(set_b_sources & seed_sources[31]),
            "seed29_seed31_overlap": len(seed_sources[29] & seed_sources[31]),
            "three_way_overlap": len(set_b_sources & seed_sources[29] & seed_sources[31]),
            "seed29_seed31_overlap_after_set_b_exclusion": len(after_set_b_overlap),
            "subsets": {label: len(values) for label, values in subsets.items()},
        },
        "datasets": datasets,
        "set_b_complementarity": complementarity,
        "set_b_deterministic_boundary_cases": boundary_cases,
        "set_b_ocr_join_audit": join_audit,
        "set_b_error_pattern_counts": error_summary,
        "set_b_string_error_diagnostic_counts": string_error_summary,
        "sources": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sources
        ],
        "integrity_boundaries": [
            "The initial Set B fusion analysis remains post-hoc descriptive.",
            "Set-B-excluded 83-source checks retain 18 sources shared between seed 29 and seed 31.",
            "The two 65-source strict-disjoint subsets share no sources with Set B or each other.",
            "All analysed drawings remain in the synthetic PIDQA family; no external replication is implied.",
            "String-error categories are deterministic form diagnostics, not causal failure annotations.",
        ],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_write(root / args.summary_csv, summary_csv_rows(datasets))
    csv_write(root / args.per_source_csv, all_per_source_rows)
    csv_write(root / args.error_csv, string_errors)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "datasets": len(datasets),
                "set_b_sources": len(set_b_sources),
                "nonoverlap_counts": {label: len(values) for label, values in subsets.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
