"""Run the frozen, visual-budget-matched InternVL counterfactual matrix.

The image is resized without distortion into a white 9x6 tile canvas and split
into fifty-four 448x448 tiles.  This produces 32,514,048 input tensor elements,
within 10% of the 35,979,264 elements recorded for the qualified Qwen 3072-side
processor path while preserving the common 512-token output cap under
InternVL's declared context limit.  Matching tensor elements is an engineering
control, not an assertion that the two vision encoders are equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from run_internvl35_f3_matrix import MEAN, STD
from run_internvl35_f3_matrix_v4 import load_model
from run_internvl_tile_budget_v1 import decode_token_count
from run_vlm_f2_matrix import PROMPTS, prompt_hash


CONDITIONS = ("correct", "shuffled", "text_only")


def wait_for_mainline_shard_barrier() -> None:
    """Hold only the original mainline before it duplicates active shards."""

    proc_cmdline = Path(f"/proc/{os.getppid()}/cmdline")
    if not proc_cmdline.is_file():
        return
    parent_command = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    )
    if "launch_rineng_v8_h200.sh mainline_full" not in parent_command:
        return
    control = Path(
        os.environ.get(
            "RINENG_INTERNVL_MAINLINE_BARRIER_FILE",
            "/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/control/internvl_mainline_barrier.txt",
        )
    )
    announced = False
    while control.is_file() and control.read_text(encoding="utf-8").strip() == "WAIT":
        if not announced:
            print(
                json.dumps(
                    {
                        "status": "waiting_for_condition_disjoint_shards",
                        "control": str(control),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            announced = True
        time.sleep(30)
    if announced:
        print(json.dumps({"status": "shard_barrier_released"}, sort_keys=True), flush=True)


def is_fatal_accelerator_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    markers = (
        "invalid access of peer gpu memory",
        "hardware error",
        "cuda error",
        "device-side assert",
        "illegal memory access",
    )
    return any(marker in text for marker in markers)


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


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_pair(correct: list[dict[str, Any]], shuffled: list[dict[str, Any]], label: str) -> None:
    for condition, rows in (("correct", correct), ("shuffled", shuffled)):
        if not rows:
            raise ValueError(f"{label}/{condition}: empty manifest")
        for row in rows:
            if any("answer" in str(key).lower() or "cypher" in str(key).lower() for key in row):
                raise ValueError(f"{label}/{condition}: answer-bearing public row")
        ids = [str(row["instance_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label}/{condition}: duplicate instance_id")
    if [row["instance_id"] for row in correct] != [row["instance_id"] for row in shuffled]:
        raise ValueError(f"{label}: paired membership/order mismatch")
    if any(str(row["source_id"]) == str(row.get("image_source_id")) for row in shuffled):
        raise ValueError(f"{label}: shuffled fixed point")


def letterbox_grid_tensor(
    image_path: Path,
    *,
    columns: int,
    rows: int,
    tile_side: int,
    fill: tuple[int, int, int] = (255, 255, 255),
    add_thumbnail: bool = False,
):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    original_size = tuple(int(value) for value in image.size)
    canvas_width, canvas_height = columns * tile_side, rows * tile_side
    scale = min(canvas_width / image.width, canvas_height / image.height)
    resized_size = (
        max(1, min(canvas_width, round(image.width * scale))),
        max(1, min(canvas_height, round(image.height * scale))),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill)
    offset = ((canvas_width - resized.width) // 2, (canvas_height - resized.height) // 2)
    canvas.paste(resized, offset)
    mean = np.asarray(MEAN, dtype=np.float32)
    std = np.asarray(STD, dtype=np.float32)
    tensors = []
    for row_index in range(rows):
        for column_index in range(columns):
            left, top = column_index * tile_side, row_index * tile_side
            tile = canvas.crop((left, top, left + tile_side, top + tile_side))
            array = np.asarray(tile, dtype=np.float32) / 255.0
            array = (array - mean) / std
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    if add_thumbnail:
        thumbnail = image.resize((tile_side, tile_side), Image.Resampling.LANCZOS)
        array = np.asarray(thumbnail, dtype=np.float32) / 255.0
        array = (array - mean) / std
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    pixels = torch.stack(tensors).to(dtype=torch.bfloat16)
    metadata = {
        "original_image_size": list(original_size),
        "letterbox_resized_image_size": list(resized_size),
        "letterbox_canvas_size": [canvas_width, canvas_height],
        "letterbox_offset": list(offset),
        "grid_columns": columns,
        "grid_rows": rows,
        "tile_side": tile_side,
        "tile_count": len(tensors),
        "thumbnail_added": add_thumbnail,
        "input_tensor_elements": int(pixels.numel()),
    }
    return pixels, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", default="internvl35_8b_budget60")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--datasets", default="")
    parser.add_argument("--record-limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-fatal-restarts", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    plan_path = resolve(root, args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_inference":
        raise ValueError("Plan is not frozen_before_inference")
    frozen = plan["frozen_inference"]
    columns = int(frozen["grid_columns"])
    grid_rows = int(frozen["grid_rows"])
    tile_side = int(frozen["tile_side"])
    max_new_tokens = int(frozen["max_new_tokens"])
    expected_elements = int(frozen["total_input_tensor_elements"])
    expected_tile_count = int(frozen["tile_count"])
    thumbnail_added = bool(frozen["thumbnail_added"])
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    requested_datasets = {
        value.strip() for value in args.datasets.split(",") if value.strip()
    }
    if not conditions or any(value not in CONDITIONS for value in conditions):
        raise ValueError("Unknown condition")
    if args.record_limit < 0:
        raise ValueError("record-limit cannot be negative")

    parent = plan["parent_plan"]
    parent_path = resolve(root, parent["path"])
    if sha256(parent_path) != parent["sha256"]:
        raise ValueError("Parent plan hash mismatch")
    datasets = []
    for spec in plan["datasets"]:
        if requested_datasets and str(spec["dataset_id"]) not in requested_datasets:
            continue
        correct_path = resolve(root, spec["correct_input"])
        shuffled_path = resolve(root, spec["shuffled_input"])
        if sha256(correct_path) != spec["correct_sha256"] or sha256(shuffled_path) != spec["shuffled_sha256"]:
            raise ValueError(f"{spec['dataset_id']}: manifest hash mismatch")
        correct, shuffled = read_jsonl(correct_path), read_jsonl(shuffled_path)
        validate_pair(correct, shuffled, spec["dataset_id"])
        if args.record_limit:
            correct, shuffled = correct[: args.record_limit], shuffled[: args.record_limit]
        datasets.append((spec, correct, shuffled))

    model_path = resolve(root, args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wait_for_mainline_shard_barrier()
    model, tokenizer = load_model(str(model_path))
    device = next(model.parameters()).device
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    plan_hash = sha256(plan_path)
    failures: list[dict[str, Any]] = []
    cell_summaries = []
    fatal_accelerator_failure = False

    for spec, correct, shuffled in datasets:
        dataset_id = str(spec["dataset_id"])
        for condition in conditions:
            records = shuffled if condition == "shuffled" else correct
            scope_suffix = f"_smoke{args.record_limit}" if args.record_limit else ""
            output_path = output_dir / (
                f"{args.model_label}_{dataset_id}_p0_{condition}_letterbox{expected_tile_count}{scope_suffix}.jsonl"
            )
            prior = read_jsonl(output_path) if output_path.is_file() else []
            rows_by_id = {
                str(row["instance_id"]): row
                for row in prior
                if str(row.get("status")) == "ok"
            }
            if args.skip_existing and len(rows_by_id) == len(records):
                cell_summaries.append(
                    {"cell": output_path.stem, "status": "pass", "rows": len(rows_by_id), "errors": 0}
                )
                continue
            for number, record in enumerate(records, start=1):
                instance_id = str(record["instance_id"])
                if instance_id in rows_by_id:
                    continue
                raw = ""
                try:
                    import torch

                    pixels = None
                    image_metadata: dict[str, Any] = {}
                    if condition != "text_only":
                        image_path = resolve(root, str(record["image_path"]))
                        pixels, image_metadata = letterbox_grid_tensor(
                            image_path,
                            columns=columns,
                            rows=grid_rows,
                            tile_side=tile_side,
                            add_thumbnail=thumbnail_added,
                        )
                        if int(pixels.numel()) != expected_elements:
                            raise ValueError("Realized tensor-element budget differs from frozen plan")
                        pixels = pixels.to(device)
                    question = PROMPTS["p0"].format(question=str(record["question"]))
                    if condition != "text_only":
                        question = "<image>\n" + question
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        response = model.chat(tokenizer, pixels, question, generation_config)
                    latency = time.perf_counter() - started
                    if isinstance(response, tuple):
                        response = response[0]
                    raw = str(response).strip()
                    output_tokens = decode_token_count(tokenizer, raw)
                    result: dict[str, Any] = {
                        "instance_id": instance_id,
                        "source_id": record["source_id"],
                        "source_sheet": record.get("source_sheet"),
                        "image_source_id": None
                        if condition == "text_only"
                        else record.get("image_source_id", record["source_id"]),
                        "task": record["task"],
                        "model": str(model_path),
                        "model_label": args.model_label,
                        "run_id": args.run_id,
                        "dataset_id": dataset_id,
                        "condition_id": condition,
                        "prompt_id": "p0",
                        "prompt_sha256": prompt_hash("p0"),
                        "plan_sha256": plan_hash,
                        "max_new_tokens": max_new_tokens,
                        "tiling_strategy": "aspect_preserving_white_letterbox_grid_plus_thumbnail"
                        if thumbnail_added
                        else "aspect_preserving_white_letterbox_grid",
                        "input_image_path": None if condition == "text_only" else str(record["image_path"]),
                        "input_pixel_count": 0 if condition == "text_only" else image_metadata["input_tensor_elements"],
                        "content_condition": condition,
                        "action": "ANSWER",
                        "answer": raw,
                        "raw": raw,
                        "latency_seconds": latency,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "output_token_count": output_tokens,
                        "output_reached_max_new_tokens": output_tokens is not None and output_tokens >= max_new_tokens,
                        "device": str(device),
                        "test_answer_used": False,
                        "record_limit": args.record_limit,
                        "status": "ok",
                        **image_metadata,
                    }
                except Exception as exc:
                    fatal_accelerator_failure = is_fatal_accelerator_error(exc)
                    result = {
                        "instance_id": instance_id,
                        "source_id": record.get("source_id"),
                        "task": record.get("task"),
                        "model_label": args.model_label,
                        "run_id": args.run_id,
                        "dataset_id": dataset_id,
                        "condition_id": condition,
                        "prompt_id": "p0",
                        "prompt_sha256": prompt_hash("p0"),
                        "plan_sha256": plan_hash,
                        "action": "INVALID",
                        "answer": None,
                        "raw": raw,
                        "test_answer_used": False,
                        "record_limit": args.record_limit,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append({"cell": output_path.stem, **result})
                rows_by_id[instance_id] = result
                write_jsonl(output_path, [rows_by_id[key] for key in sorted(rows_by_id)])
                print(
                    json.dumps(
                        {
                            "cell": output_path.stem,
                            "progress": f"{number}/{len(records)}",
                            "status": result["status"],
                            "latency_seconds": result.get("latency_seconds"),
                            "peak_allocated_bytes": result.get("peak_allocated_bytes"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if fatal_accelerator_failure:
                    break
            ok_count = sum(row.get("status") == "ok" for row in rows_by_id.values())
            error_count = len(rows_by_id) - ok_count
            cell_summaries.append(
                {
                    "cell": output_path.stem,
                    "status": "pass" if ok_count == len(records) and error_count == 0 else "fail",
                    "rows": len(rows_by_id),
                    "ok": ok_count,
                    "errors": error_count,
                }
            )
            if fatal_accelerator_failure:
                break
        if fatal_accelerator_failure:
            break

    status = "pass" if not failures and all(row["status"] == "pass" for row in cell_summaries) else "fail"
    summary = {
        "version": "rineng-internvl-budget-matched-v8",
        "status": status,
        "model_label": args.model_label,
        "run_id": args.run_id,
        "plan": str(plan_path.relative_to(root).as_posix()),
        "plan_sha256": plan_hash,
        "record_limit": args.record_limit,
        "conditions": conditions,
        "requested_datasets": sorted(requested_datasets),
        "cells": cell_summaries,
        "failure_count": len(failures),
        "failures": failures,
    }
    summary_name = f"run_summary_smoke{args.record_limit}.json" if args.record_limit else "run_summary.json"
    (output_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "summary": str(output_dir / summary_name)}, sort_keys=True))
    if fatal_accelerator_failure:
        attempt = int(os.environ.get("RINENG_INTERNVL_FATAL_RESTART", "0"))
        if attempt < args.max_fatal_restarts:
            next_environment = dict(os.environ)
            next_environment["RINENG_INTERNVL_FATAL_RESTART"] = str(attempt + 1)
            print(
                json.dumps(
                    {
                        "status": "fatal_accelerator_restart",
                        "attempt": attempt + 1,
                        "maximum": args.max_fatal_restarts,
                        "resume_policy": "fresh process with skip-existing valid rows",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(10)
            os.execvpe(sys.executable, [sys.executable, *sys.argv], next_environment)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
