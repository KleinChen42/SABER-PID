"""Build the F0 answer-blind set and F1 task-level audit artifacts.

The set-B selector is intentionally answer-blind: it ranks candidate rows only
by stable hashes of public identifiers.  Existing balanced set A is not
modified.  The audit also recomputes task-level source-cluster effects from the
already scored mainline shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from pidbench.io import read_jsonl, write_json, write_jsonl
from pidbench.pidqa_metrics import normalize_pidqa_answer


TASKS = ("connectivity", "count", "spatial_count", "value")


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * probability)]


def image_hash(path: Path) -> tuple[str, str, tuple[int, int]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        small = rgb.resize((32, 32)).convert("L")
        pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)
    ahash = "".join("1" if value >= mean else "0" for value in pixels)
    return digest, ahash, (width, height)


def build_image_audit(root: Path, output_csv: Path, output_json: Path) -> dict[str, Any]:
    image_root = root / "data" / "raw" / "pidqa_images"
    rows: list[dict[str, Any]] = []
    for split_dir in (image_root / "train", image_root / "val"):
        for image_path in sorted(split_dir.glob("*.jpg")):
            source_sheet = image_path.stem
            source_id = f"pidqa-sheet-{int(source_sheet):03d}"
            sha256, ahash, size = image_hash(image_path)
            rows.append(
                {
                    "source_id": source_id,
                    "source_sheet": source_sheet,
                    "image_path": str(image_path.relative_to(root)).replace("\\", "/"),
                    "image_split": split_dir.name,
                    "sha256": sha256,
                    "ahash32": ahash,
                    "width": size[0],
                    "height": size[1],
                }
            )
    rows.sort(key=lambda row: row["source_id"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_sha: dict[str, list[str]] = defaultdict(list)
    by_ahash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row["source_id"])
        by_ahash[row["ahash32"]].append(row["source_id"])
    exact_clusters = [sorted(items) for items in by_sha.values() if len(items) > 1]
    ahash_clusters = [sorted(items) for items in by_ahash.values() if len(items) > 1]

    # Source assignments are deterministic from the project split function.
    records = list(read_jsonl(root / "data" / "processed" / "pidqa_records.jsonl"))
    source_ids = sorted({str(row["source_id"]) for row in records})
    split_summary: dict[str, Any] = {}
    for seed in (3, 17, 29, 43, 71):
        keys = list(source_ids)
        rng = random.Random(seed)
        rng.shuffle(keys)
        train_end = round(len(keys) * 0.60)
        cal_end = train_end + round(len(keys) * 0.20)
        assignment = {
            key: "train" if i < train_end else "calibration" if i < cal_end else "test"
            for i, key in enumerate(keys)
        }
        split_summary[str(seed)] = {
            "train_sources": sum(value == "train" for value in assignment.values()),
            "calibration_sources": sum(value == "calibration" for value in assignment.values()),
            "test_sources": sum(value == "test" for value in assignment.values()),
            "exact_clusters_crossing_splits": [
                cluster
                for cluster in exact_clusters
                if len({assignment[source] for source in cluster}) > 1
            ],
            "ahash_clusters_crossing_splits": [
                cluster
                for cluster in ahash_clusters
                if len({assignment[source] for source in cluster}) > 1
            ],
        }

    payload = {
        "status": "pass",
        "image_count": len(rows),
        "exact_cluster_count": len(exact_clusters),
        "exact_duplicate_source_count": sum(len(cluster) for cluster in exact_clusters),
        "ahash_cluster_count": len(ahash_clusters),
        "ahash_duplicate_source_count": sum(len(cluster) for cluster in ahash_clusters),
        "exact_clusters": exact_clusters,
        "ahash_clusters": ahash_clusters,
        "split_summary": split_summary,
        "image_hash_definition": "SHA-256 bytes plus 32x32 grayscale average hash",
    }
    write_json(output_json, payload)
    return payload


def build_hashblind_set(root: Path, output_dir: Path) -> dict[str, Any]:
    records = list(read_jsonl(root / "data" / "processed" / "pidqa_records.jsonl"))
    assignments = {
        str(row["instance_id"]): str(row["split"])
        for row in read_jsonl(root / "data" / "manifests" / "pidqa_source_split_seed17.jsonl")
    }
    set_a = {
        str(row["instance_id"])
        for row in read_jsonl(root / "data" / "processed" / "main400_source_test_diverse_public.jsonl")
    }
    image_by_source = {
        str(row["source_id"]): str(row["image_path"])
        for row in read_jsonl(root / "data" / "processed" / "main400_source_test_diverse_with_images_public.jsonl")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        instance_id = str(row["instance_id"])
        source = str(row["source_id"])
        task = str(row["task"])
        if assignments.get(instance_id) == "test" and instance_id not in set_a:
            grouped[(source, task)].append(row)

    sources = sorted({source for source, _ in grouped})
    selected: list[dict[str, Any]] = []
    for source in sources:
        for task in TASKS:
            candidates = grouped[(source, task)]
            if not candidates:
                raise RuntimeError(f"No answer-blind candidate for {source}/{task}")
            selected.append(
                min(
                    candidates,
                    key=lambda row: stable_key("hashblind-set-b", 17, source, task, row["instance_id"]),
                )
            )
    selected.sort(key=lambda row: (str(row["source_id"]), str(row["task"]), str(row["instance_id"])))
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for row in selected:
        source = str(row["source_id"])
        if source not in image_by_source:
            raise RuntimeError(f"Missing bound image for {source}")
        public = {key: value for key, value in row.items() if key not in {"answer", "cypher"}}
        public["image_path"] = image_by_source[source]
        public_rows.append(public)
        hidden_rows.append(dict(row))

    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "main400_hashblind_set_b_public.jsonl"
    hidden_path = root / "data" / "answer_store" / "main400_hashblind_set_b_hidden.jsonl"
    write_jsonl(public_path, public_rows)
    write_jsonl(hidden_path, hidden_rows)
    summary = {
        "status": "pass",
        "set_id": "hashblind_set_B",
        "selection": "stable SHA-256 ranking over public identifiers only",
        "seed": 17,
        "record_count": len(selected),
        "source_count": len({str(row["source_id"]) for row in selected}),
        "task_counts": dict(sorted(Counter(str(row["task"]) for row in selected).items())),
        "overlap_with_set_A": len({str(row["instance_id"]) for row in selected} & set_a),
        "public_path": str(public_path.relative_to(root)).replace("\\", "/"),
        "hidden_path": str(hidden_path.relative_to(root)).replace("\\", "/"),
    }
    write_json(output_dir / "main400_hashblind_set_b_summary.json", summary)
    return summary


def parse_tags(value: Any) -> set[str]:
    normalized = normalize_pidqa_answer(value, "value")
    return set(normalized or ())


def correctness(record: dict[str, Any], prediction: dict[str, Any]) -> int:
    return int(
        str(prediction.get("action", "INVALID")) == "ANSWER"
        and normalize_pidqa_answer(prediction.get("answer"), str(record["task"]))
        == normalize_pidqa_answer(record.get("answer"), str(record["task"]))
    )


def bootstrap_task_effects(
    records: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
    reps: int = 10000,
    seed: int = 17,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_id"])].append(record)
    sources = sorted(by_source)
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_records = [record for record in records if str(record["task"]) == task]
        base_values = {str(record["instance_id"]): correctness(record, baseline[str(record["instance_id"])]) for record in task_records}
        cond_values = {str(record["instance_id"]): correctness(record, condition[str(record["instance_id"])]) for record in task_records}
        base_acc = sum(base_values.values()) / len(task_records)
        cond_acc = sum(cond_values.values()) / len(task_records)
        rng = random.Random(seed)
        diffs: list[float] = []
        task_by_source = {
            source: [record for record in rows_for_source if str(record["task"]) == task]
            for source, rows_for_source in by_source.items()
        }
        for _ in range(reps):
            sampled = [rng.choice(sources) for _ in sources]
            sampled_records = [record for source in sampled for record in task_by_source[source]]
            if not sampled_records:
                continue
            diffs.append(
                sum(
                    cond_values[str(record["instance_id"])] - base_values[str(record["instance_id"])]
                    for record in sampled_records
                )
                / len(sampled_records)
            )
        rows.append(
            {
                "task": task,
                "record_count": len(task_records),
                "source_count": len(sources),
                "baseline_accuracy": base_acc,
                "condition_accuracy": cond_acc,
                "difference_condition_minus_baseline": cond_acc - base_acc,
                "bootstrap_ci95_low": quantile(diffs, 0.025),
                "bootstrap_ci95_high": quantile(diffs, 0.975),
                "bootstrap_reps": reps,
                "seed": seed,
            }
        )
    return rows


def f1_audit(root: Path, output_dir: Path) -> dict[str, Any]:
    records = list(read_jsonl(root / "data" / "answer_store" / "main400_source_test_diverse_hidden.jsonl"))
    labels = {
        "qwen8_768": root / "outputs/main/qwen3vl8b_source400_clean_768.jsonl",
        "qwen8_1536": root / "outputs/main/qwen3vl8b_source400_clean_1536.jsonl",
        "qwen8_2304": root / "outputs/main/qwen3vl8b_source400_clean_2304.jsonl",
        "qwen8_3072": root / "outputs/main/qwen3vl8b_source400_clean_3072.jsonl",
        "qwen32_1536": root / "outputs/main/qwen3vl32b_source400_clean_1536.jsonl",
        "qwen32_3072": root / "outputs/main/qwen3vl32b_source400_clean_3072.jsonl",
    }
    predictions = {label: {str(row["instance_id"]): row for row in read_jsonl(path)} for label, path in labels.items()}
    task_rows: list[dict[str, Any]] = []
    for label, by_id in predictions.items():
        for task in TASKS:
            task_records = [record for record in records if str(record["task"]) == task]
            correct = [correctness(record, by_id[str(record["instance_id"])]) for record in task_records]
            task_rows.append(
                {
                    "label": label,
                    "task": task,
                    "record_count": len(task_records),
                    "accuracy": sum(correct) / len(correct),
                }
            )

    effects: dict[str, Any] = {}
    for label, condition_label in (("qwen8", "qwen8_3072"), ("qwen32", "qwen32_3072")):
        baseline_label = f"{label}_768" if label == "qwen8" else "qwen32_1536"
        effects[f"{condition_label}_minus_{baseline_label}"] = bootstrap_task_effects(
            records, predictions[baseline_label], predictions[condition_label]
        )

    tag_rows: list[dict[str, Any]] = []
    value_records = [record for record in records if str(record["task"]) == "value"]
    for label, by_id in predictions.items():
        tp = fp = fn = exact = 0
        for record in value_records:
            truth = parse_tags(record.get("answer"))
            pred = parse_tags(by_id[str(record["instance_id"])].get("answer"))
            tp += len(truth & pred)
            fp += len(pred - truth)
            fn += len(truth - pred)
            exact += int(truth == pred and str(by_id[str(record["instance_id"])].get("action", "")) == "ANSWER")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        tag_rows.append(
            {
                "label": label,
                "record_count": len(value_records),
                "exact_set_accuracy": exact / len(value_records),
                "tag_tp": tp,
                "tag_fp": fp,
                "tag_fn": fn,
                "tag_precision": precision,
                "tag_recall": recall,
                "tag_f1": f1,
            }
        )

    flip_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_records = [record for record in records if str(record["task"]) == task]
        categories = Counter()
        for record in task_records:
            instance_id = str(record["instance_id"])
            base = correctness(record, predictions["qwen8_768"][instance_id])
            high = correctness(record, predictions["qwen8_3072"][instance_id])
            categories[f"{base}->{high}"] += 1
        for category, count in sorted(categories.items()):
            flip_rows.append({"task": task, "transition": category, "count": count})

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolution_task_effects_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    with (output_dir / "value_tag_resolution_analysis_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tag_rows[0]))
        writer.writeheader()
        writer.writerows(tag_rows)
    with (output_dir / "resolution_flip_matrix_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flip_rows[0]))
        writer.writeheader()
        writer.writerows(flip_rows)
    payload = {
        "status": "pass",
        "records": len(records),
        "source_count": len({str(row["source_id"]) for row in records}),
        "task_rows": task_rows,
        "tag_rows": tag_rows,
        "flip_rows": flip_rows,
        "bootstrap_effects": effects,
        "prediction_labels": sorted(predictions),
    }
    write_json(output_dir / "f1_task_effects_v2.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    generated = root / "reports" / "generated"
    image_payload = build_image_audit(
        root,
        generated / "pidqa_image_identity_audit_v2.csv",
        generated / "pidqa_cross_source_duplicate_audit_v2.json",
    )
    set_payload = build_hashblind_set(root, root / "data" / "processed")
    f1_payload = f1_audit(root, generated)
    summary = {
        "status": "pass",
        "f0": {
            "image_count": image_payload["image_count"],
            "exact_cluster_count": image_payload["exact_cluster_count"],
            "ahash_cluster_count": image_payload["ahash_cluster_count"],
            "set_b": set_payload,
        },
        "f1": {
            "records": f1_payload["records"],
            "task_rows": len(f1_payload["task_rows"]),
            "tag_rows": len(f1_payload["tag_rows"]),
            "bootstrap_comparisons": len(f1_payload["bootstrap_effects"]),
        },
    }
    write_json(generated / "f0_f1_audit_summary_v2.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
