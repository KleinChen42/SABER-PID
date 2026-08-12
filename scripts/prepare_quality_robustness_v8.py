"""Prepare the frozen 3072/512 quality-robustness matrix for RINENG v8.

The script projects the answer-isolated v7 manifests into four image-quality
conditions: clean, JPEG quality 70, Gaussian blur radius 1, and 0.75x
downsample/restore.  Correct and source-shuffled image bindings remain paired.
Blur and downsample outputs are lossless PNGs so those interventions are not
silently combined with an additional lossy JPEG intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, __version__ as pillow_version


QUALITY_SPECS: dict[str, dict[str, Any]] = {
    "clean": {"family": "identity", "severity": None, "extension": None},
    "jpeg_q70": {"family": "jpeg", "severity": 70, "extension": ".jpg"},
    "blur_r1": {"family": "gaussian_blur", "severity": 1.0, "extension": ".png"},
    "downsample_s075": {
        "family": "downsample_restore",
        "severity": 0.75,
        "extension": ".png",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_public(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label}: empty public manifest")
    for row in rows:
        forbidden = [key for key in row if "answer" in str(key).lower() or "cypher" in str(key).lower()]
        if forbidden:
            raise ValueError(f"{label}: forbidden public fields {forbidden}")


def validate_pair(correct: list[dict[str, Any]], shuffled: list[dict[str, Any]], label: str) -> None:
    validate_public(correct, f"{label}/correct")
    validate_public(shuffled, f"{label}/shuffled")
    if [row["instance_id"] for row in correct] != [row["instance_id"] for row in shuffled]:
        raise ValueError(f"{label}: correct/shuffled membership or order mismatch")
    if any(str(row["source_id"]) == str(row.get("image_source_id")) for row in shuffled):
        raise ValueError(f"{label}: shuffled fixed point")


def stable_target(root: Path, condition: str, original_path: str, extension: str) -> Path:
    key = hashlib.sha256(original_path.encode("utf-8")).hexdigest()[:20]
    return root / "data" / "raw" / "rineng_v8_quality" / condition / f"{key}{extension}"


def transform_image(source: Path, target: Path, spec: dict[str, Any]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as loaded:
        image = loaded.convert("RGB")
    original_size = tuple(int(value) for value in image.size)
    family = spec["family"]
    if family == "jpeg":
        image.save(
            target,
            format="JPEG",
            quality=int(spec["severity"]),
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        intermediate_size = None
    elif family == "gaussian_blur":
        image.filter(ImageFilter.GaussianBlur(radius=float(spec["severity"]))).save(
            target, format="PNG", compress_level=6, optimize=False
        )
        intermediate_size = None
    elif family == "downsample_restore":
        scale = float(spec["severity"])
        intermediate_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image.resize(intermediate_size, Image.Resampling.LANCZOS).resize(
            image.size, Image.Resampling.LANCZOS
        ).save(target, format="PNG", compress_level=6, optimize=False)
    else:
        raise ValueError(f"Unsupported transform family: {family}")
    with Image.open(target) as generated:
        generated_size = tuple(int(value) for value in generated.size)
    if generated_size != original_size:
        raise ValueError(f"Output dimensions changed for {source}: {generated_size} != {original_size}")
    return {
        "original_size": list(original_size),
        "intermediate_size": list(intermediate_size) if intermediate_size else None,
        "output_size": list(generated_size),
        "output_bytes": target.stat().st_size,
        "output_sha256": sha256(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--parent-plan", default="data/manifests/rineng_overnight_v7_public_plan.json"
    )
    parser.add_argument(
        "--output-plan", default="data/manifests/rineng_v8_quality_robustness_plan.json"
    )
    parser.add_argument(
        "--image-manifest", default="data/manifests/rineng_v8_quality_images.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    parent_path = root / args.parent_plan
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("status") != "frozen_before_inference":
        raise ValueError("Parent plan is not frozen")

    plan_datasets = []
    image_records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    membership: dict[str, list[str]] = {}
    for dataset in parent["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        correct_path = root / dataset["correct_input"]
        shuffled_path = root / dataset["shuffled_input"]
        if sha256(correct_path) != dataset["correct_sha256"] or sha256(shuffled_path) != dataset["shuffled_sha256"]:
            raise ValueError(f"{dataset_id}: parent manifest hash mismatch")
        correct = read_jsonl(correct_path)
        shuffled = read_jsonl(shuffled_path)
        validate_pair(correct, shuffled, dataset_id)
        membership[dataset_id] = sorted({str(row["source_id"]) for row in correct})

        for condition, spec in QUALITY_SPECS.items():
            if condition == "clean":
                conditioned_correct = correct
                conditioned_shuffled = shuffled
                out_correct_path = correct_path
                out_shuffled_path = shuffled_path
                all_paths = sorted({str(row["image_path"]) for row in correct + shuffled})
                for original in all_paths:
                    source = root / original
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    key = (condition, original)
                    if key not in image_records_by_key:
                        with Image.open(source) as loaded:
                            size = [int(value) for value in loaded.size]
                        image_records_by_key[key] = {
                            "condition": condition,
                            "family": spec["family"],
                            "severity": spec["severity"],
                            "original_path": original,
                            "output_path": original,
                            "original_sha256": sha256(source),
                            "output_sha256": sha256(source),
                            "original_size": size,
                            "output_size": size,
                            "output_bytes": source.stat().st_size,
                        }
            else:
                path_map: dict[str, str] = {}
                all_paths = sorted({str(row["image_path"]) for row in correct + shuffled})
                for original in all_paths:
                    source = root / original
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    target = stable_target(root, condition, original, str(spec["extension"]))
                    metadata = transform_image(source, target, spec)
                    relative_target = target.relative_to(root).as_posix()
                    path_map[original] = relative_target
                    key = (condition, original)
                    image_records_by_key[key] = {
                        "condition": condition,
                        "family": spec["family"],
                        "severity": spec["severity"],
                        "original_path": original,
                        "output_path": relative_target,
                        "original_sha256": sha256(source),
                        **metadata,
                    }

                def project(row: dict[str, Any]) -> dict[str, Any]:
                    original = str(row["image_path"])
                    return {
                        **row,
                        "original_image_path": original,
                        "image_path": path_map[original],
                        "image_condition": condition,
                        "degradation_family": spec["family"],
                        "degradation_severity": spec["severity"],
                    }

                conditioned_correct = [project(row) for row in correct]
                conditioned_shuffled = [project(row) for row in shuffled]
                output_root = root / "data" / "processed" / "rineng_v8_quality"
                out_correct_path = output_root / f"{dataset_id}_{condition}_correct_public.jsonl"
                out_shuffled_path = output_root / f"{dataset_id}_{condition}_shuffled_public.jsonl"
                write_jsonl(out_correct_path, conditioned_correct)
                write_jsonl(out_shuffled_path, conditioned_shuffled)
                validate_pair(conditioned_correct, conditioned_shuffled, f"{dataset_id}/{condition}")

            plan_datasets.append(
                {
                    "dataset_id": f"{dataset_id}__{condition}",
                    "base_dataset_id": dataset_id,
                    "quality_condition": condition,
                    "quality_family": spec["family"],
                    "quality_severity": spec["severity"],
                    "correct_input": out_correct_path.relative_to(root).as_posix(),
                    "correct_sha256": sha256(out_correct_path),
                    "shuffled_input": out_shuffled_path.relative_to(root).as_posix(),
                    "shuffled_sha256": sha256(out_shuffled_path),
                    "record_count": len(conditioned_correct),
                    "source_count": len(membership[dataset_id]),
                }
            )

    image_records = sorted(
        image_records_by_key.values(), key=lambda row: (row["condition"], row["original_path"])
    )
    image_manifest = {
        "version": "rineng-v8-quality-images",
        "status": "pass",
        "generator": Path(__file__).name,
        "generator_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "pillow": pillow_version,
        "conditions": QUALITY_SPECS,
        "unique_condition_image_count": len(image_records),
        "records": image_records,
    }
    image_manifest_path = root / args.image_manifest
    image_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    image_manifest_path.write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plan = {
        "version": "rineng-v8-quality-robustness",
        "date": "2026-08-12",
        "status": "frozen_before_inference",
        "analysis_hierarchy": {
            "primary": "Within model/dataset/condition, correct-minus-shuffled strict value-tag F1 at the qualified 3072-side and 512-token configuration",
            "secondary": [
                "quality-minus-clean paired source differences",
                "all-task strict accuracy boundary",
                "source-macro strict accuracy",
            ],
            "interpretation": "Image quality is paired with a shuffled control so gains cannot be attributed to prompt or answer priors alone.",
        },
        "parent_plan": {"path": args.parent_plan, "sha256": sha256(parent_path)},
        "image_manifest": {
            "path": args.image_manifest,
            "sha256": sha256(image_manifest_path),
            "record_count": len(image_records),
        },
        "conditions": ["correct", "shuffled"],
        "quality_conditions": QUALITY_SPECS,
        "datasets": plan_datasets,
        "frozen_inference": {
            "model_label": "qwen3vl8b",
            "model_path": "models/Qwen3-VL-8B-Instruct-modelscope",
            "prompt_id": "p0",
            "prompt_sha256": "184085c056465088fdadaa882c6591ec175ec0fc2e100113365c2b2f20780223",
            "max_image_side": 3072,
            "max_new_tokens": 512,
            "do_sample": False,
            "task_filter": "all four PIDQA tasks",
        },
        "membership": {
            "datasets": {key: len(value) for key, value in membership.items()},
            "all_pairwise_source_disjoint": all(
                not (set(left) & set(right))
                for index, left in enumerate(membership.values())
                for right in list(membership.values())[index + 1 :]
            ),
        },
        "answer_isolation": {
            "forbidden_prediction_input_field_fragments": ["answer", "cypher"],
            "scorer_reference_read_by_inference": False,
            "status": "pass",
        },
    }
    plan_path = root / args.output_plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "plan": str(plan_path),
                "dataset_cells": len(plan_datasets),
                "unique_condition_images": len(image_records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
