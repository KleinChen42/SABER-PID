"""Independently validate the frozen RINENG V8 extension score reports.

This validator deliberately does not import either V8 scorer. It recomputes
membership, hashes, answer-isolation flags, strict metrics, paired effects,
and 10,000-replicate intervals from immutable predictions. Bootstrap seeds are
independent of the scorers; endpoints must agree within a declared Monte Carlo
tolerance rather than being byte-identical copies of the original calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pidbench.pidqa_metrics import normalize_pidqa_answer  # noqa: E402


TAG_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,8})\s*(?:[- ]\s*)?"
    r"(\d+(?:\s*[.\-/]\s*\d+)*)([A-Za-z]?)(?![A-Za-z0-9])"
)
EXCLUDED_PREFIXES = {"mm", "rev"}
POINT_TOLERANCE = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)


def f1(counts: tuple[int, int, int] | list[int]) -> float:
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def pooled(counts: dict[str, tuple[int, int, int]], sample: Iterable[str] | None = None) -> float:
    totals = [0, 0, 0]
    keys = list(sample) if sample is not None else sorted(counts)
    for key in keys:
        for index, value in enumerate(counts[key]):
            totals[index] += value
    return f1(totals)


def quantile(values: list[float], probability: float) -> float:
    values.sort()
    return values[round((len(values) - 1) * probability)]


def bootstrap_delta(
    baseline: dict[str, tuple[int, int, int]],
    treatment: dict[str, tuple[int, int, int]],
    *,
    reps: int,
    seed: int,
) -> list[float]:
    keys = sorted(set(baseline) & set(treatment))
    if set(keys) != set(baseline) or set(keys) != set(treatment):
        raise ValueError("Bootstrap count groups are not paired")
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [rng.choice(keys) for _ in keys]
        draws.append(pooled(treatment, sample) - pooled(baseline, sample))
    return [quantile(draws.copy(), 0.025), quantile(draws, 0.975)]


def bootstrap_mean_delta(
    baseline: dict[str, float],
    treatment: dict[str, float],
    *,
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    keys = sorted(set(baseline) & set(treatment))
    if set(keys) != set(baseline) or set(keys) != set(treatment):
        raise ValueError("Bootstrap accuracy groups are not paired")
    deltas = [treatment[key] - baseline[key] for key in keys]
    point = statistics.mean(deltas)
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choice(deltas) for _ in deltas) for _ in range(reps)]
    return point, [quantile(draws.copy(), 0.025), quantile(draws, 0.975)]


def bootstrap_did(
    clean_correct: dict[str, tuple[int, int, int]],
    clean_shuffled: dict[str, tuple[int, int, int]],
    degraded_correct: dict[str, tuple[int, int, int]],
    degraded_shuffled: dict[str, tuple[int, int, int]],
    *,
    reps: int,
    seed: int,
) -> tuple[float, list[float]]:
    keys = sorted(
        set(clean_correct)
        & set(clean_shuffled)
        & set(degraded_correct)
        & set(degraded_shuffled)
    )
    if not keys:
        raise ValueError("No paired sources for difference-in-differences")

    def estimate(sample: list[str]) -> float:
        return (
            pooled(degraded_correct, sample)
            - pooled(degraded_shuffled, sample)
            - pooled(clean_correct, sample)
            + pooled(clean_shuffled, sample)
        )

    point = estimate(keys)
    rng = random.Random(seed)
    draws = [estimate([rng.choice(keys) for _ in keys]) for _ in range(reps)]
    return point, [quantile(draws.copy(), 0.025), quantile(draws, 0.975)]


def prediction_text(row: dict[str, Any]) -> Any:
    return row["answer"] if "answer" in row else row.get("raw")


def answered(row: dict[str, Any]) -> bool:
    if "action" in row:
        return str(row.get("action")) == "ANSWER"
    return str(row.get("status", "ok")) == "ok"


def pidqa_cell(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    pred_ids = [str(row.get("instance_id")) for row in predictions]
    ref_ids = [str(row["instance_id"]) for row in references]
    if len(pred_ids) != len(set(pred_ids)) or set(pred_ids) != set(ref_ids):
        raise ValueError("PIDQA prediction membership mismatch or duplicates")
    pred_by_id = {str(row["instance_id"]): row for row in predictions}
    source_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    source_correct: dict[str, list[int]] = defaultdict(list)
    task_correct: dict[str, list[int]] = defaultdict(list)
    correct_values: list[int] = []
    for reference in references:
        row = pred_by_id[str(reference["instance_id"])]
        task = str(reference["task"])
        truth = normalize_pidqa_answer(reference.get("answer"), task)
        predicted = normalize_pidqa_answer(prediction_text(row), task)
        is_correct = int(answered(row) and predicted == truth)
        source = str(reference["source_id"])
        source_correct[source].append(is_correct)
        task_correct[task].append(is_correct)
        correct_values.append(is_correct)
        if task == "value":
            truth_tags = set(truth or ())
            predicted_tags = set(predicted or ()) if answered(row) else set()
            counts = source_counts[source]
            counts[0] += len(truth_tags & predicted_tags)
            counts[1] += len(predicted_tags - truth_tags)
            counts[2] += len(truth_tags - predicted_tags)
    frozen_counts = {key: tuple(values) for key, values in sorted(source_counts.items())}
    totals = [sum(values[index] for values in frozen_counts.values()) for index in range(3)]
    tp, fp, fn = totals
    return {
        "counts": frozen_counts,
        "source_accuracy": {
            key: statistics.mean(values) for key, values in sorted(source_correct.items())
        },
        "metrics": {
            "strict_accuracy": statistics.mean(correct_values),
            "source_macro_accuracy": statistics.mean(
                statistics.mean(values) for values in source_correct.values()
            ),
            "task_accuracy": {
                key: statistics.mean(values) for key, values in task_correct.items()
            },
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": f1(totals),
        },
    }


def normalized_external_tags(value: Any, requested_prefix: str) -> set[str]:
    text = html.unescape(str(value or "")).replace("\r", " ").replace("\n", " ")
    tags: set[str] = set()
    for match in TAG_RE.finditer(text):
        prefix, numeric, suffix = match.groups()
        prefix = prefix.casefold()
        if prefix in EXCLUDED_PREFIXES or prefix != requested_prefix.casefold():
            continue
        parts = re.split(r"\s*[.\-/]\s*", numeric)
        tags.add(prefix + "-".join(parts) + suffix.casefold())
    return tags


def external_cell(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    pred_ids = [str(row.get("instance_id")) for row in predictions]
    ref_ids = [str(row["instance_id"]) for row in references]
    if len(pred_ids) != len(set(pred_ids)) or set(pred_ids) != set(ref_ids):
        raise ValueError("DEXPI prediction membership mismatch or duplicates")
    pred_by_id = {str(row["instance_id"]): row for row in predictions}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    exacts: list[int] = []
    group_exacts: dict[str, list[int]] = defaultdict(list)
    for reference in references:
        row = pred_by_id[str(reference["instance_id"])]
        group = str(reference["source_sheet"])
        truth = {str(tag).casefold() for tag in reference["answer"]}
        predicted = normalized_external_tags(
            prediction_text(row), str(reference["fields"]["Prefix"])
        )
        cell = counts[group]
        cell[0] += len(truth & predicted)
        cell[1] += len(predicted - truth)
        cell[2] += len(truth - predicted)
        exact = int(truth == predicted)
        exacts.append(exact)
        group_exacts[group].append(exact)
    frozen = {key: tuple(value) for key, value in sorted(counts.items())}
    totals = [sum(value[index] for value in frozen.values()) for index in range(3)]
    tp, fp, fn = totals
    return {
        "counts": frozen,
        "metrics": {
            "records": len(references),
            "logical_groups": len(frozen),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "exact_set_accuracy": statistics.mean(exacts),
            "logical_group_macro_exact_accuracy": statistics.mean(
                statistics.mean(value) for value in group_exacts.values()
            ),
        },
    }


def merge_maps(maps: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for values in maps:
        overlap = set(result) & set(values)
        if overlap:
            raise ValueError(f"Pooled source groups are not disjoint: {sorted(overlap)[:3]}")
        result.update(values)
    return result


def add_close(errors: list[str], actual: Any, expected: Any, label: str) -> float:
    difference = abs(float(actual) - float(expected))
    if difference > POINT_TOLERANCE:
        errors.append(f"{label}: point mismatch {actual} vs {expected}")
    return difference


def add_ci_close(
    errors: list[str], actual: list[float], expected: list[float], label: str, tolerance: float
) -> float:
    difference = max(abs(float(a) - float(b)) for a, b in zip(actual, expected))
    if difference > tolerance:
        errors.append(
            f"{label}: independent-bootstrap endpoint difference {difference:.6g} exceeds {tolerance}"
        )
    return difference


def validate_row_integrity(
    errors: list[str],
    rows: list[dict[str, Any]],
    expected_ids: list[str],
    plan_hash: str,
    label: str,
    expected_elements: int | None,
) -> None:
    ids = [str(row.get("instance_id")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        errors.append(f"{label}: membership or duplicate failure")
    if any(str(row.get("status")) != "ok" for row in rows):
        errors.append(f"{label}: non-ok prediction row")
    if any(row.get("test_answer_used") is True for row in rows):
        errors.append(f"{label}: hidden-answer flag is true")
    if any(str(row.get("plan_sha256")) != plan_hash for row in rows):
        errors.append(f"{label}: row plan hash mismatch")
    if expected_elements is not None and any(
        int(row.get("actual_input_pixel_count", row.get("input_pixel_count", -1)))
        != expected_elements
        for row in rows
    ):
        errors.append(f"{label}: input tensor-element budget mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--extension-score", default="reports/generated/rineng_v8_extension_score.json"
    )
    parser.add_argument(
        "--external-score", default="reports/generated/rineng_v8_dexpi_external_score.json"
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--ci-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--output", default="reports/generated/rineng_v8_independent_validation.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    extension = json.loads((root / args.extension_score).read_text(encoding="utf-8"))
    external = json.loads((root / args.external_score).read_text(encoding="utf-8"))
    quality_plan_path = root / "data/manifests/rineng_v8_quality_robustness_plan.json"
    internvl_plan_path = root / "data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json"
    external_plan_path = root / "data/manifests/rineng_v8_dexpi_external_plan.json"
    quality_plan = json.loads(quality_plan_path.read_text(encoding="utf-8"))
    internvl_plan = json.loads(internvl_plan_path.read_text(encoding="utf-8"))
    external_plan = json.loads(external_plan_path.read_text(encoding="utf-8"))
    references = read_jsonl(root / "data/processed/pidqa_records.jsonl")
    reference_by_id = {str(row["instance_id"]): row for row in references}
    errors: list[str] = []
    max_point_error = 0.0
    max_ci_error = 0.0

    if extension.get("status") != "pass" or external.get("status") != "pass":
        errors.append("One or both scorer reports are not pass")
    if extension.get("quality_plan", {}).get("sha256") != sha256(quality_plan_path):
        errors.append("Quality plan hash mismatch")
    if extension.get("internvl_plan", {}).get("sha256") != sha256(internvl_plan_path):
        errors.append("InternVL plan hash mismatch")
    if external.get("plan", {}).get("sha256") != sha256(external_plan_path):
        errors.append("DEXPI plan hash mismatch")

    quality_cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    quality_plan_hash = sha256(quality_plan_path)
    for spec in quality_plan["datasets"]:
        dataset_id = str(spec["dataset_id"])
        base = str(spec["base_dataset_id"])
        quality = str(spec["quality_condition"])
        manifest = read_jsonl(root / spec["correct_input"])
        expected_ids = [str(row["instance_id"]) for row in manifest]
        scorer_refs = [reference_by_id[key] for key in expected_ids]
        for condition in ("correct", "shuffled"):
            path = root / "outputs/rineng_v8/qwen3vl8b_quality" / (
                f"qwen3vl8b_{dataset_id}_p0_{condition}_3072.jsonl"
            )
            rows = read_jsonl(path)
            label = f"quality|qwen3vl8b|{base}|{quality}|{condition}"
            validate_row_integrity(
                errors, rows, expected_ids, quality_plan_hash, label, 35_979_264
            )
            cell = pidqa_cell(scorer_refs, rows)
            quality_cells[(base, quality, condition)] = cell
            reported_cell = extension["cells"][label]
            if reported_cell.get("sha256") != sha256(path):
                errors.append(f"{label}: raw output hash mismatch")
            tags = reported_cell["metrics"]["strict_value_tags"]
            for name in ("tp", "fp", "fn", "precision", "recall", "f1"):
                max_point_error = max(
                    max_point_error,
                    add_close(errors, tags[name], cell["metrics"][name], f"{label}|{name}"),
                )

    base_datasets = sorted({key[0] for key in quality_cells})
    pooled_label = "pooled_three_source_disjoint_subsets"

    def quality_map(dataset: str, quality: str, condition: str, field: str) -> dict[str, Any]:
        datasets = base_datasets if dataset == pooled_label else [dataset]
        return merge_maps(quality_cells[(item, quality, condition)][field] for item in datasets)

    for row in extension["quality_comparisons"]:
        dataset = str(row["dataset"])
        quality = str(row["quality"])
        contrast = str(row["contrast"])
        label = f"quality-comparison|{dataset}|{quality}|{contrast}"
        if contrast == "correct_minus_shuffled":
            correct = quality_map(dataset, quality, "correct", "counts")
            shuffled = quality_map(dataset, quality, "shuffled", "counts")
            point = pooled(correct) - pooled(shuffled)
            interval = bootstrap_delta(
                shuffled,
                correct,
                reps=args.bootstrap_reps,
                seed=stable_seed(label),
            )
            max_point_error = max(
                max_point_error,
                add_close(errors, row["value_f1_difference"], point, label),
            )
            max_ci_error = max(
                max_ci_error,
                add_ci_close(
                    errors,
                    row["value_f1_source_bootstrap_ci95"],
                    interval,
                    label,
                    args.ci_tolerance,
                ),
            )
        else:
            clean_correct = quality_map(dataset, "clean", "correct", "counts")
            clean_shuffled = quality_map(dataset, "clean", "shuffled", "counts")
            degraded_correct = quality_map(dataset, quality, "correct", "counts")
            degraded_shuffled = quality_map(dataset, quality, "shuffled", "counts")
            point, interval = bootstrap_did(
                clean_correct,
                clean_shuffled,
                degraded_correct,
                degraded_shuffled,
                reps=args.bootstrap_reps,
                seed=stable_seed(label),
            )
            max_point_error = max(
                max_point_error,
                add_close(
                    errors,
                    row["value_f1_difference_in_differences"],
                    point,
                    label,
                ),
            )
            max_ci_error = max(
                max_ci_error,
                add_ci_close(
                    errors, row["source_bootstrap_ci95"], interval, label, args.ci_tolerance
                ),
            )

    internvl_cells: dict[tuple[str, str], dict[str, Any]] = {}
    internvl_plan_hash = sha256(internvl_plan_path)
    internvl_elements = int(internvl_plan["frozen_inference"]["total_input_tensor_elements"])
    internvl_label = str(internvl_plan["models"][0]["model_label"])
    for spec in internvl_plan["datasets"]:
        dataset = str(spec["dataset_id"])
        expected_ids = [
            str(row["instance_id"]) for row in read_jsonl(root / spec["correct_input"])
        ]
        scorer_refs = [reference_by_id[key] for key in expected_ids]
        for condition in internvl_plan["conditions"]:
            path = root / "outputs/rineng_v8/internvl35_8b_budget54" / (
                f"{internvl_label}_{dataset}_p0_{condition}_letterbox54.jsonl"
            )
            rows = read_jsonl(path)
            label = f"internvl_budget54|{dataset}|{condition}"
            validate_row_integrity(
                errors,
                rows,
                expected_ids,
                internvl_plan_hash,
                label,
                None if condition == "text_only" else internvl_elements,
            )
            cell = pidqa_cell(scorer_refs, rows)
            internvl_cells[(dataset, condition)] = cell
            reported_cell = extension["cells"][label]
            if reported_cell.get("sha256") != sha256(path):
                errors.append(f"{label}: raw output hash mismatch")
            for name in ("tp", "fp", "fn", "precision", "recall", "f1"):
                max_point_error = max(
                    max_point_error,
                    add_close(
                        errors,
                        reported_cell["metrics"]["strict_value_tags"][name],
                        cell["metrics"][name],
                        f"{label}|{name}",
                    ),
                )

    internvl_datasets = [str(spec["dataset_id"]) for spec in internvl_plan["datasets"]]

    def internvl_counts(dataset: str, condition: str) -> dict[str, tuple[int, int, int]]:
        datasets = internvl_datasets if dataset == pooled_label else [dataset]
        return merge_maps(internvl_cells[(item, condition)]["counts"] for item in datasets)

    v7_cache: dict[str, dict[str, tuple[int, int, int]]] = {}
    for row in extension["internvl_comparisons"]:
        dataset = str(row["dataset"])
        contrast = str(row["contrast"])
        label = f"internvl-comparison|{dataset}|{contrast}"
        treatment = internvl_counts(dataset, "correct")
        if contrast.startswith("correct_minus_"):
            baseline = internvl_counts(dataset, contrast.removeprefix("correct_minus_"))
        else:
            datasets = internvl_datasets if dataset == pooled_label else [dataset]
            baseline_parts = []
            for item in datasets:
                if item not in v7_cache:
                    spec = next(value for value in internvl_plan["datasets"] if value["dataset_id"] == item)
                    expected_ids = [
                        str(value["instance_id"])
                        for value in read_jsonl(root / spec["correct_input"])
                    ]
                    refs = [reference_by_id[key] for key in expected_ids]
                    path = root / "outputs/rineng_overnight_v7/internvl35_8b" / (
                        f"internvl35_8b_{item}_p0_correct_tiles12.jsonl"
                    )
                    v7_cache[item] = pidqa_cell(refs, read_jsonl(path))["counts"]
                baseline_parts.append(v7_cache[item])
            baseline = merge_maps(baseline_parts)
        point = pooled(treatment) - pooled(baseline)
        interval = bootstrap_delta(
            baseline, treatment, reps=args.bootstrap_reps, seed=stable_seed(label)
        )
        max_point_error = max(
            max_point_error,
            add_close(errors, row["value_f1_difference"], point, label),
        )
        max_ci_error = max(
            max_ci_error,
            add_ci_close(
                errors,
                row["value_f1_source_bootstrap_ci95"],
                interval,
                label,
                args.ci_tolerance,
            ),
        )

    hidden_path = root / external_plan["answer_isolation"]["hidden_reference"]
    external_refs = read_jsonl(hidden_path)
    external_plan_hash = sha256(external_plan_path)
    external_paths = {
        "correct": root / "outputs/rineng_v8/dexpi_external_qwen/qwen3vl8b_dexpi_external_v8_p0_correct_3072.jsonl",
        "shuffled": root / "outputs/rineng_v8/dexpi_external_qwen/qwen3vl8b_dexpi_external_v8_p0_shuffled_3072.jsonl",
        "text_only": root / "outputs/rineng_v8/dexpi_external_qwen/qwen3vl8b_dexpi_external_v8_p0_text_only_3072.jsonl",
        "paddleocr_full_image": root / "outputs/rineng_v8/dexpi_external_ocr.jsonl",
    }
    expected_external_ids = [str(row["instance_id"]) for row in external_refs]
    external_cells: dict[str, dict[str, Any]] = {}
    for condition, path in external_paths.items():
        rows = read_jsonl(path)
        validate_row_integrity(
            errors,
            rows,
            expected_external_ids,
            external_plan_hash,
            f"dexpi|{condition}",
            None,
        )
        cell = external_cell(external_refs, rows)
        external_cells[condition] = cell
        if external["integrity"]["cells"][condition]["sha256"] != sha256(path):
            errors.append(f"dexpi|{condition}: raw output hash mismatch")
        for name, expected in cell["metrics"].items():
            max_point_error = max(
                max_point_error,
                add_close(errors, external["metrics"][condition][name], expected, f"dexpi|{condition}|{name}"),
            )

    for key, row in external["comparisons"].items():
        left, baseline_name = key.split("_minus_", 1)
        condition_alias = {"ocr": "paddleocr_full_image"}
        treatment = external_cells[condition_alias.get(left, left)]["counts"]
        baseline = external_cells[condition_alias.get(baseline_name, baseline_name)]["counts"]
        point = pooled(treatment) - pooled(baseline)
        interval = bootstrap_delta(
            baseline,
            treatment,
            reps=args.bootstrap_reps,
            seed=stable_seed(f"dexpi|{key}"),
        )
        max_point_error = max(
            max_point_error,
            add_close(errors, row["difference"], point, f"dexpi|{key}"),
        )
        max_ci_error = max(
            max_ci_error,
            add_ci_close(
                errors, row["ci95"], interval, f"dexpi|{key}", args.ci_tolerance
            ),
        )

    report = {
        "version": "rineng-v8-independent-validation",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
        "independence": {
            "imports_v8_scorers": False,
            "bootstrap_seed_relation": "independent SHA-256-derived seeds",
            "bootstrap_reps": args.bootstrap_reps,
            "ci_endpoint_tolerance": args.ci_tolerance,
        },
        "scope": {
            "quality_cells": len(quality_cells),
            "quality_comparisons": len(extension["quality_comparisons"]),
            "internvl_cells": len(internvl_cells),
            "internvl_comparisons": len(extension["internvl_comparisons"]),
            "dexpi_cells": len(external_cells),
            "dexpi_comparisons": len(external["comparisons"]),
        },
        "numeric_agreement": {
            "max_point_absolute_error": max_point_error,
            "max_independent_ci_endpoint_absolute_difference": max_ci_error,
        },
        "inputs": {
            args.extension_score: sha256(root / args.extension_score),
            args.external_score: sha256(root / args.external_score),
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "error_count": len(errors),
                "max_point_error": max_point_error,
                "max_ci_endpoint_difference": max_ci_error,
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
