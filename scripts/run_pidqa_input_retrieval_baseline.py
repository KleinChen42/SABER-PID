"""Evaluate input-only retrieval paths under random and source-aware splits.

The cache keys use only the visible question text and an image fingerprint.  No
source_id, hidden Cypher, or test answer is used to build predictions.  This is
an operational companion to the earlier source_id-based exposure diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from pidbench.io import read_jsonl, write_json
from pidbench.pidqa_metrics import normalize_pidqa_answer
from pidbench.question_keys import question_semantic_signature_for_record
from pidbench.splits import make_random_split, make_source_split


METHODS = ("L0_global_task_prior", "L1_text_exact", "L2_image_text_exact", "L3_image_semantic", "L4_phash_semantic", "L5_image_semantic_with_prior")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    """Return a small average hash robust to ordinary re-encoding/resizing."""

    with Image.open(path) as loaded:
        image = loaded.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
    return f"{int(bits, 2):0{len(bits) // 4}x}"


def image_map(image_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for path in sorted(image_root.rglob(pattern)):
            paths.setdefault(path.stem, path)
    return paths


def majority_answer(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    first_text: dict[tuple[str, str], str] = {}
    for row in rows:
        task = str(row["task"])
        canonical = repr(normalize_pidqa_answer(row.get("answer"), task))
        grouped[task][canonical] += 1
        first_text.setdefault((task, canonical), str(row.get("answer", "")))
    result: dict[str, str] = {}
    for task, counts in grouped.items():
        canonical, _ = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        result[task] = first_text[(task, canonical)]
    return result


def build_cache(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any]) -> tuple[dict[Any, str], set[Any]]:
    values: dict[Any, set[str]] = defaultdict(set)
    first_text: dict[Any, str] = {}
    for row in rows:
        key = key_fn(row)
        task = str(row["task"])
        canonical = repr(normalize_pidqa_answer(row.get("answer"), task))
        values[key].add(canonical)
        first_text.setdefault(key, str(row.get("answer", "")))
    unambiguous = {key: first_text[key] for key, options in values.items() if len(options) == 1}
    ambiguous = {key for key, options in values.items() if len(options) > 1}
    return unambiguous, ambiguous


def score_prediction(record: dict[str, Any], action: str, answer: Any) -> int:
    return int(
        action == "ANSWER"
        and normalize_pidqa_answer(answer, str(record["task"]))
        == normalize_pidqa_answer(record.get("answer"), str(record["task"]))
    )


def evaluate_method(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    method: str,
    exact_by_source_sheet: dict[str, str],
    phash_by_source_sheet: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prior = majority_answer(train_rows)
    text_cache, text_ambiguous = build_cache(train_rows, lambda row: str(row["question"]))
    image_text_cache, image_text_ambiguous = build_cache(
        train_rows,
        lambda row: (exact_by_source_sheet.get(str(row["source_sheet"])), str(row["question"])),
    )
    image_semantic_cache, image_semantic_ambiguous = build_cache(
        train_rows,
        lambda row: (exact_by_source_sheet.get(str(row["source_sheet"])), question_semantic_signature_for_record(row)),
    )
    phash_semantic_cache, phash_semantic_ambiguous = build_cache(
        train_rows,
        lambda row: (phash_by_source_sheet.get(str(row["source_sheet"])), question_semantic_signature_for_record(row)),
    )
    cache_by_method = {
        "L1_text_exact": (text_cache, text_ambiguous, lambda row: str(row["question"])),
        "L2_image_text_exact": (
            image_text_cache,
            image_text_ambiguous,
            lambda row: (exact_by_source_sheet.get(str(row["source_sheet"])), str(row["question"])),
        ),
        "L3_image_semantic": (
            image_semantic_cache,
            image_semantic_ambiguous,
            lambda row: (exact_by_source_sheet.get(str(row["source_sheet"])), question_semantic_signature_for_record(row)),
        ),
        "L4_phash_semantic": (
            phash_semantic_cache,
            phash_semantic_ambiguous,
            lambda row: (phash_by_source_sheet.get(str(row["source_sheet"])), question_semantic_signature_for_record(row)),
        ),
    }
    predictions: list[dict[str, Any]] = []
    answered = 0
    correct = 0
    by_task: dict[str, list[int]] = defaultdict(list)
    by_source: dict[str, list[int]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    for row in test_rows:
        task = str(row["task"])
        if method == "L0_global_task_prior":
            action, status, answer = "ANSWER", "task_prior", prior.get(task)
        elif method == "L5_image_semantic_with_prior":
            cache, ambiguous, key_fn = cache_by_method["L3_image_semantic"]
            key = key_fn(row)
            if key in cache:
                action, status, answer = "ANSWER", "image_semantic_hit", cache[key]
            else:
                action, status, answer = "ANSWER", "task_prior_fallback", prior.get(task)
        else:
            cache, ambiguous, key_fn = cache_by_method[method]
            key = key_fn(row)
            if key in cache:
                action, status, answer = "ANSWER", "cache_hit", cache[key]
            elif key in ambiguous:
                action, status, answer = "ABSTAIN", "ambiguous_key", None
            else:
                action, status, answer = "ABSTAIN", "cache_miss", None
        is_answered = int(action == "ANSWER")
        is_correct = score_prediction(row, action, answer)
        answered += is_answered
        correct += is_correct
        by_task[task].append(is_correct)
        by_source[str(row["source_id"])].append(is_correct)
        prediction = {
            "instance_id": str(row["instance_id"]),
            "source_id": str(row["source_id"]),
            "task": task,
            "action": action,
            "answer": answer,
            "status": status,
            "method": method,
        }
        predictions.append(prediction)
        if status in {"cache_hit", "image_semantic_hit"} and len(examples) < 10:
            examples.append({
                "instance_id": str(row["instance_id"]),
                "source_sheet": str(row["source_sheet"]),
                "task": task,
                "question": str(row["question"]),
                "status": status,
                "answer": answer,
            })

    total = len(test_rows)
    source_accuracy = [sum(values) / len(values) for values in by_source.values() if values]
    task_accuracy = {
        task: sum(values) / len(values) if values else 0.0
        for task, values in sorted(by_task.items())
    }
    cache_ambiguous = {
        "L1_text_exact": len(text_ambiguous),
        "L2_image_text_exact": len(image_text_ambiguous),
        "L3_image_semantic": len(image_semantic_ambiguous),
        "L4_phash_semantic": len(phash_semantic_ambiguous),
    }
    summary = {
        "method": method,
        "record_count": total,
        "prediction_count": len(predictions),
        "answered_count": answered,
        "correct_count": correct,
        "coverage": answered / total if total else 0.0,
        "answered_accuracy": correct / answered if answered else 0.0,
        "overall_accuracy": correct / total if total else 0.0,
        "source_macro_accuracy": sum(source_accuracy) / len(source_accuracy) if source_accuracy else 0.0,
        "task_accuracy": task_accuracy,
        "ambiguous_cache_keys": cache_ambiguous.get(method, 0),
        "examples": examples,
        "test_answers_used_to_build_predictions": False,
    }
    return summary, predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--seeds", default="3,17,29,43,71")
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--predictions-dir", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    images = image_map(Path(args.image_root))
    sheets = sorted({str(row["source_sheet"]) for row in records}, key=lambda value: int(value))
    missing_sheets = [sheet for sheet in sheets if sheet not in images]
    if missing_sheets:
        raise FileNotFoundError(
            f"Missing {len(missing_sheets)} image sheets; first examples: {missing_sheets[:10]}"
        )
    exact_by_sheet = {sheet: file_sha256(images[sheet]) for sheet in sheets}
    phash_by_sheet = {sheet: perceptual_hash(images[sheet]) for sheet in sheets}
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = Path(args.predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    prediction_paths: list[str] = []
    splitters = (("random", make_random_split), ("source", make_source_split))
    for split_name, splitter in splitters:
        for seed in seeds:
            assignments = splitter(records, seed)
            split_by_id = {str(row["instance_id"]): str(row["split"]) for row in assignments}
            train_rows = [row for row in records if split_by_id[str(row["instance_id"])] == "train"]
            test_rows = [row for row in records if split_by_id[str(row["instance_id"])] == "test"]
            for method in METHODS:
                summary, predictions = evaluate_method(
                    train_rows, test_rows, method, exact_by_sheet, phash_by_sheet
                )
                result = {
                    "split": split_name,
                    "seed": seed,
                    "train_records": len(train_rows),
                    "test_records": len(test_rows),
                    "train_source_count": len({str(row["source_id"]) for row in train_rows}),
                    "test_source_count": len({str(row["source_id"]) for row in test_rows}),
                    "image_source_count": len(images),
                    **{key: value for key, value in summary.items() if key != "examples"},
                    "examples": summary["examples"],
                }
                rows.append(result)
                prediction_path = output_dir / f"{split_name}_seed{seed}_{method}.jsonl"
                with prediction_path.open("w", encoding="utf-8") as handle:
                    for prediction in predictions:
                        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                prediction_paths.append(prediction_path.as_posix())
    payload = {
        "method": "input_only_image_question_retrieval",
        "identity": "image SHA-256 and 32x32 average perceptual hash",
        "semantic_key": "released-template question text parser",
        "source_id_used_as_cache_key": False,
        "test_answers_used_to_build_predictions": False,
        "seeds": seeds,
        "image_source_count": len(images),
        "rows": rows,
        "prediction_paths": prediction_paths,
    }
    write_json(args.json, payload)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "split", "seed", "method", "train_records", "test_records", "train_source_count",
        "test_source_count", "image_source_count", "coverage", "answered_accuracy",
        "overall_accuracy", "source_macro_accuracy", "correct_count", "answered_count",
        "ambiguous_cache_keys", "task_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, "") for key in csv_fields}
            flat["task_accuracy"] = json.dumps(row.get("task_accuracy", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(flat)
    print(json.dumps({"status": "complete", "row_count": len(rows), "image_source_count": len(images)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
