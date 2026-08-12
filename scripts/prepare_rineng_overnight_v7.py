"""Freeze answer-isolated manifests for the RINENG v7 overnight matrix.

The matrix extends the current PIDQA evidence without downloading data or
reading scorer-only references.  It keeps Set B and adds two 65-source subsets
that share no source with Set B or with one another.  Each correct manifest is
paired with a deterministic source derangement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_source_shuffled_control import build_shuffle, write_jsonl
from run_vlm_f2_matrix import PROMPTS, prompt_hash


VERSION = "rineng-overnight-v7"
PROMPT_IDS = ("p0", "p1")
CONDITIONS = ("correct", "shuffled", "text_only")
PUBLIC_FIELDS = {
    "dataset",
    "e6_source_split_seed",
    "fields",
    "image_condition",
    "image_control",
    "image_path",
    "image_source_id",
    "instance_id",
    "original_image_path",
    "question",
    "question_template_id",
    "source_id",
    "source_sheet",
    "task",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key in PUBLIC_FIELDS} for row in rows]


def validate_answer_isolated(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label}: empty manifest")
    if any(
        any("answer" in str(key).lower() or "cypher" in str(key).lower() for key in row)
        for row in rows
    ):
        raise ValueError(f"{label}: answer-bearing field detected")
    ids = [str(row["instance_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate instance_id")


def dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {str(row["source_id"]) for row in rows}
    return {
        "record_count": len(rows),
        "source_count": len(sources),
        "task_counts": dict(sorted(Counter(str(row["task"]) for row in rows).items())),
    }


def validate_four_task_source_block(
    rows: list[dict[str, Any]], expected_sources: int, label: str
) -> None:
    validate_answer_isolated(rows, label)
    summary = dataset_summary(rows)
    expected_tasks = {
        "connectivity": expected_sources,
        "count": expected_sources,
        "spatial_count": expected_sources,
        "value": expected_sources,
    }
    if summary["source_count"] != expected_sources:
        raise ValueError(f"{label}: unexpected source count {summary['source_count']}")
    if summary["record_count"] != 4 * expected_sources:
        raise ValueError(f"{label}: unexpected record count {summary['record_count']}")
    if summary["task_counts"] != expected_tasks:
        raise ValueError(f"{label}: unexpected task counts {summary['task_counts']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="data/manifests/rineng_overnight_v7_public_plan.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    source_paths = {
        "set_b": root / "data/processed/main400_hashblind_set_b_remote_public.jsonl",
        "set_b_shuffled": root
        / "data/processed/main400_hashblind_set_b_shuffled_v1_remote_public.jsonl",
        "seed29": root / "data/processed/source_seed29_resolution_v1_remote_public.jsonl",
        "seed31": root / "data/processed/source_seed31_resolution_v1_remote_public.jsonl",
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    set_b = sanitize(read_jsonl(source_paths["set_b"]))
    set_b_shuffled = sanitize(read_jsonl(source_paths["set_b_shuffled"]))
    seed29 = sanitize(read_jsonl(source_paths["seed29"]))
    seed31 = sanitize(read_jsonl(source_paths["seed31"]))
    validate_four_task_source_block(set_b, 100, "set_b")
    validate_four_task_source_block(seed29, 100, "seed29")
    validate_four_task_source_block(seed31, 100, "seed31")
    validate_four_task_source_block(set_b_shuffled, 100, "set_b_shuffled")
    if [str(row["instance_id"]) for row in set_b] != [
        str(row["instance_id"]) for row in set_b_shuffled
    ]:
        raise ValueError("Set B shuffled manifest changed instance order")
    if any(
        str(row["source_id"]) == str(row.get("image_source_id"))
        for row in set_b_shuffled
    ):
        raise ValueError("Set B shuffled manifest contains a fixed point")

    set_b_sources = {str(row["source_id"]) for row in set_b}
    seed29_sources = {str(row["source_id"]) for row in seed29}
    seed31_sources = {str(row["source_id"]) for row in seed31}
    strict29_sources = seed29_sources - set_b_sources - seed31_sources
    strict31_sources = seed31_sources - set_b_sources - seed29_sources
    if len(strict29_sources) != 65 or len(strict31_sources) != 65:
        raise ValueError(
            f"Unexpected strict source counts: {len(strict29_sources)}/{len(strict31_sources)}"
        )
    if set_b_sources & strict29_sources or set_b_sources & strict31_sources:
        raise ValueError("Strict subset overlaps Set B")
    if strict29_sources & strict31_sources:
        raise ValueError("Strict subsets overlap one another")

    generated_dir = root / "data/processed/rineng_overnight_v7"
    generated_dir.mkdir(parents=True, exist_ok=True)
    set_b_correct_path = generated_dir / "set_b100_correct_public.jsonl"
    set_b_shuffled_path = generated_dir / "set_b100_shuffled_public.jsonl"
    write_jsonl(set_b_correct_path, set_b)
    write_jsonl(set_b_shuffled_path, set_b_shuffled)
    strict29 = [row for row in seed29 if str(row["source_id"]) in strict29_sources]
    strict31 = [row for row in seed31 if str(row["source_id"]) in strict31_sources]
    validate_four_task_source_block(strict29, 65, "seed29_strict65")
    validate_four_task_source_block(strict31, 65, "seed31_strict65")

    strict29_correct_path = generated_dir / "seed29_strict65_correct_public.jsonl"
    strict31_correct_path = generated_dir / "seed31_strict65_correct_public.jsonl"
    strict29_shuffled_path = generated_dir / "seed29_strict65_shuffled_public.jsonl"
    strict31_shuffled_path = generated_dir / "seed31_strict65_shuffled_public.jsonl"
    write_jsonl(strict29_correct_path, strict29)
    write_jsonl(strict31_correct_path, strict31)
    strict29_shuffled, _, _ = build_shuffle(strict29, 2026081201)
    strict31_shuffled, _, _ = build_shuffle(strict31, 2026081202)
    write_jsonl(strict29_shuffled_path, strict29_shuffled)
    write_jsonl(strict31_shuffled_path, strict31_shuffled)
    validate_four_task_source_block(strict29_shuffled, 65, "seed29_strict65_shuffled")
    validate_four_task_source_block(strict31_shuffled, 65, "seed31_strict65_shuffled")
    if any(
        str(row["source_id"]) == str(row.get("image_source_id"))
        for row in strict29_shuffled + strict31_shuffled
    ):
        raise ValueError("Generated shuffled manifest contains a fixed point")

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    datasets = [
        {
            "dataset_id": "set_b100",
            "correct_input": relative(set_b_correct_path),
            "shuffled_input": relative(set_b_shuffled_path),
            **dataset_summary(set_b),
        },
        {
            "dataset_id": "seed29_strict65",
            "correct_input": relative(strict29_correct_path),
            "shuffled_input": relative(strict29_shuffled_path),
            **dataset_summary(strict29),
        },
        {
            "dataset_id": "seed31_strict65",
            "correct_input": relative(strict31_correct_path),
            "shuffled_input": relative(strict31_shuffled_path),
            **dataset_summary(strict31),
        },
    ]
    for dataset in datasets:
        dataset["correct_sha256"] = sha256(root / dataset["correct_input"])
        dataset["shuffled_sha256"] = sha256(root / dataset["shuffled_input"])

    plan = {
        "version": VERSION,
        "status": "frozen_before_inference",
        "date": "2026-08-12",
        "answer_isolation": {
            "status": "pass",
            "forbidden_prediction_input_field_fragments": ["answer", "cypher"],
            "allowlisted_field_projection": sorted(PUBLIC_FIELDS),
            "scorer_reference_read_by_inference": False,
        },
        "datasets": datasets,
        "membership": {
            "set_b_sources": len(set_b_sources),
            "seed29_strict_sources": len(strict29_sources),
            "seed31_strict_sources": len(strict31_sources),
            "all_three_pairwise_source_overlap": 0,
        },
        "prompts": [
            {
                "prompt_id": prompt_id,
                "template": PROMPTS[prompt_id],
                "sha256": prompt_hash(prompt_id),
                "selection": "pre-existing frozen F2 prompt; selected before v7 inference",
            }
            for prompt_id in PROMPT_IDS
        ],
        "conditions": list(CONDITIONS),
        "models": [
            {
                "model_label": "qwen3vl8b",
                "model_path": "models/Qwen3-VL-8B-Instruct-modelscope",
                "family": "Qwen3-VL",
                "recommended_gpu": 2,
            },
            {
                "model_label": "qwen3vl32b",
                "model_path": "models/Qwen3-VL-32B-Instruct-modelscope",
                "family": "Qwen3-VL",
                "recommended_gpu": 0,
            },
            {
                "model_label": "internvl35_8b",
                "model_path": "models/InternVL3_5-8B-modelscope",
                "family": "InternVL",
                "recommended_gpu": 1,
            },
        ],
        "frozen_inference": {
            "max_image_side_qwen": 3072,
            "dynamic_preprocess_max_num_internvl": 12,
            "max_new_tokens": 512,
            "do_sample": False,
            "task_filter": "all four PIDQA tasks",
        },
        "analysis_hierarchy": {
            "primary": (
                "Within each model/dataset/prompt, correct-minus-shuffled and "
                "correct-minus-text-only strict value-tag F1 with paired source bootstrap intervals"
            ),
            "secondary": [
                "task-wise strict accuracy for all four tasks",
                "source-macro strict accuracy",
                "p1-minus-p0 sensitivity without best-prompt selection",
                "model-family and model-scale directional comparison",
            ],
            "interpretation": (
                "The two strict subsets are independent of Set B and each other, but all drawings "
                "remain inside the synthetic PIDQA family; no external-family claim is permitted."
            ),
        },
        "excluded": [
            "intermediate image-budget sweep",
            "learned fusion",
            "answer-aware prompt selection",
            "new model or dataset download",
        ],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": relative(output),
                "datasets": {row["dataset_id"]: row["record_count"] for row in datasets},
                "pairwise_source_overlap": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
