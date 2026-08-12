"""Validate the frozen v7 plan, images, models, disk, and assigned GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_rows(gpus: list[int]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, encoding="utf-8")
    rows = []
    for line in output.splitlines():
        index, name, used, total, utilization = [part.strip() for part in line.split(",")]
        if int(index) in gpus:
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization_percent": int(utilization),
                }
            )
    return sorted(rows, key=lambda row: row["index"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--plan", default="data/manifests/rineng_overnight_v7_public_plan.json"
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument(
        "--output", default="reports/generated/rineng_overnight_v7_preflight.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan_path = root / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if plan.get("status") != "frozen_before_inference":
        failures.append("plan_not_frozen")

    source_sets: dict[str, set[str]] = {}
    dataset_reports: list[dict[str, Any]] = []
    image_paths: set[Path] = set()
    expected_records_per_model = 0
    for dataset in plan.get("datasets", []):
        dataset_id = str(dataset["dataset_id"])
        correct_path = root / dataset["correct_input"]
        shuffled_path = root / dataset["shuffled_input"]
        local_failures: list[str] = []
        for label, path, expected_hash in (
            ("correct", correct_path, dataset["correct_sha256"]),
            ("shuffled", shuffled_path, dataset["shuffled_sha256"]),
        ):
            if not path.is_file():
                local_failures.append(f"{label}_missing")
                continue
            if sha256(path) != expected_hash:
                local_failures.append(f"{label}_hash_mismatch")
        if local_failures:
            failures.extend(f"{dataset_id}:{value}" for value in local_failures)
            continue
        correct = read_jsonl(correct_path)
        shuffled = read_jsonl(shuffled_path)
        for label, rows in (("correct", correct), ("shuffled", shuffled)):
            if any(
                any(
                    "answer" in str(key).lower() or "cypher" in str(key).lower()
                    for key in row
                )
                for row in rows
            ):
                local_failures.append(f"{label}_answer_field")
            ids = [str(row["instance_id"]) for row in rows]
            if len(ids) != len(set(ids)):
                local_failures.append(f"{label}_duplicate_id")
            for row in rows:
                image_paths.add(root / str(row["image_path"]))
        if [str(row["instance_id"]) for row in correct] != [
            str(row["instance_id"]) for row in shuffled
        ]:
            local_failures.append("membership_or_order_mismatch")
        if any(
            str(row["source_id"]) == str(row.get("image_source_id"))
            for row in shuffled
        ):
            local_failures.append("shuffle_fixed_point")
        sources = {str(row["source_id"]) for row in correct}
        source_sets[dataset_id] = sources
        tasks = Counter(str(row["task"]) for row in correct)
        if len(correct) != int(dataset["record_count"]):
            local_failures.append("record_count_mismatch")
        if len(sources) != int(dataset["source_count"]):
            local_failures.append("source_count_mismatch")
        if dict(sorted(tasks.items())) != dataset["task_counts"]:
            local_failures.append("task_count_mismatch")
        expected_records_per_model += len(correct) * len(plan["prompts"]) * len(
            plan["conditions"]
        )
        dataset_reports.append(
            {
                "dataset_id": dataset_id,
                "record_count": len(correct),
                "source_count": len(sources),
                "task_counts": dict(sorted(tasks.items())),
                "failure_reasons": local_failures,
            }
        )
        failures.extend(f"{dataset_id}:{value}" for value in local_failures)

    dataset_ids = sorted(source_sets)
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(dataset_ids):
        for right in dataset_ids[left_index + 1 :]:
            overlaps[f"{left}|{right}"] = len(source_sets[left] & source_sets[right])
    if any(overlaps.values()):
        failures.append("dataset_source_overlap")

    missing_images = sorted(
        path.relative_to(root).as_posix() for path in image_paths if not path.is_file()
    )
    if missing_images:
        failures.append("input_images_missing")

    model_reports = []
    for model in plan.get("models", []):
        model_path = root / model["model_path"]
        present = model_path.is_dir() and (model_path / "config.json").is_file()
        model_reports.append(
            {
                "model_label": model["model_label"],
                "path": model["model_path"],
                "present": present,
            }
        )
        if not present:
            failures.append(f"model_missing:{model['model_label']}")

    requested_gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    observed_gpus = gpu_rows(requested_gpus)
    if {row["index"] for row in observed_gpus} != set(requested_gpus):
        failures.append("assigned_gpu_missing")
    for row in observed_gpus:
        if row["memory_used_mib"] > 1024 or row["utilization_percent"] > 5:
            failures.append(f"assigned_gpu_busy:{row['index']}")

    disk = shutil.disk_usage(root)
    if disk.free < 5 * 1024**3:
        failures.append("less_than_5_gib_free")

    report = {
        "version": "rineng-overnight-v7-preflight",
        "status": "pass" if not failures else "fail",
        "failure_reasons": failures,
        "plan": args.plan,
        "plan_sha256": sha256(plan_path),
        "datasets": dataset_reports,
        "pairwise_source_overlap": overlaps,
        "unique_input_image_count": len(image_paths),
        "missing_input_images": missing_images,
        "models": model_reports,
        "assigned_gpus": observed_gpus,
        "disk_free_bytes": disk.free,
        "expected_records_per_model": expected_records_per_model,
        "expected_total_inference_rows": expected_records_per_model
        * len(plan.get("models", [])),
        "network_download_required": False,
        "scorer_reference_read_by_inference": False,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": output.relative_to(root).as_posix(),
                "missing_images": len(missing_images),
                "expected_rows": report["expected_total_inference_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

