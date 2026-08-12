"""Run E4 with actual InternVL dynamic-tile budgets rather than side labels.

Unlike the legacy F3 runner, this script never pre-resizes a P&ID image by a
nominal maximum side.  ``low`` and ``high`` are defined only by the frozen
``max_num`` values supplied to InternVL's dynamic preprocessing.  Each output
row records the realised tile count and tensor pixel count, which are the
authoritative independent variables for the E4 comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from run_internvl35_f3_matrix import MEAN, STD, dynamic_preprocess
from run_internvl35_f3_matrix_v4 import load_model


PROMPT_PREFIX = (
    "<image>\nYou analyze a piping and instrumentation diagram (P&ID). "
    "Answer only from the visible diagram. Return only the final answer, "
    "without explanation or Markdown.\n\nQuestion: "
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_path(value: str, image_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else image_root / path


def image_tensor_original(image_path: Path, max_num: int):
    """Produce InternVL pixels from the original image and record actual tiles."""

    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    original_size = tuple(int(value) for value in image.size)
    tiles, ratio = dynamic_preprocess(image, image_size=448, max_num=max_num)
    tensors = []
    for tile in tiles:
        array = np.asarray(tile, dtype=np.float32) / 255.0
        array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors).to(dtype=torch.bfloat16), original_size, len(tiles), ratio


def decode_token_count(tokenizer: Any, raw: str) -> int | None:
    try:
        encoded = tokenizer(raw, add_special_tokens=False)
        ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        return len(ids) if ids is not None else None
    except Exception:
        return None


def run_cell(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    model: Any,
    tokenizer: Any,
    image_root: Path,
    run_id: str,
    label: str,
    max_num: int,
    max_new_tokens: int,
    set_id: str,
    skip_existing: bool,
) -> None:
    import torch

    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {str(row.get("instance_id")) for row in existing}
    if skip_existing and len(done) >= len(records):
        print(json.dumps({"cell": output_path.stem, "status": "already_complete", "rows": len(existing)}, sort_keys=True), flush=True)
        return
    rows = list(existing)
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    device = next(model.parameters()).device
    for number, record in enumerate(records, start=1):
        instance_id = str(record["instance_id"])
        if instance_id in done:
            continue
        raw = ""
        try:
            image_path = resolve_path(str(record["image_path"]), image_root)
            pixels, original_size, tile_count, tile_ratio = image_tensor_original(image_path, max_num)
            pixels = pixels.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            question = PROMPT_PREFIX + str(record["question"])
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
                "task": record["task"],
                "model": str(getattr(model, "name_or_path", "InternVL3.5-8B")),
                "model_revision": str(getattr(model, "name_or_path", "InternVL3.5-8B")),
                "run_id": run_id,
                "condition_id": f"tile_budget_{label}",
                "set_id": set_id,
                "prompt_id": "p0",
                "action": "ANSWER",
                "answer": raw,
                "raw": raw,
                "latency_seconds": latency,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_new_tokens": max_new_tokens,
                "output_token_count": output_tokens,
                "output_reached_max_new_tokens": output_tokens == max_new_tokens if output_tokens is not None else None,
                "dynamic_preprocess_max_num": max_num,
                "dynamic_tile_count": tile_count,
                "dynamic_tile_ratio": list(tile_ratio),
                "input_pixel_count": int(pixels.numel()),
                "original_image_size": list(original_size),
                "resized_image_size": list(original_size),
                "pre_resize_applied": False,
                "input_image_path": str(record["image_path"]),
                "device": str(device),
                "status": "ok",
            }
        except Exception as exc:
            result = {
                "instance_id": instance_id,
                "source_id": record.get("source_id"),
                "source_sheet": record.get("source_sheet"),
                "task": record.get("task"),
                "run_id": run_id,
                "condition_id": f"tile_budget_{label}",
                "set_id": set_id,
                "prompt_id": "p0",
                "action": "INVALID",
                "answer": None,
                "raw": raw,
                "max_new_tokens": max_new_tokens,
                "output_token_count": None,
                "output_reached_max_new_tokens": None,
                "dynamic_preprocess_max_num": max_num,
                "input_image_path": record.get("image_path"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(result)
        done.add(instance_id)
        write_jsonl(output_path, rows)
        print(json.dumps({"cell": output_path.stem, "progress": f"{number}/{len(records)}", "instance_id": instance_id, "status": result["status"], "latency_seconds": result.get("latency_seconds")}, sort_keys=True), flush=True)
    print(json.dumps({"cell": output_path.stem, "status": "complete", "rows": len(rows)}, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--set-id", default="B")
    parser.add_argument("--low-max-num", type=int, default=1)
    parser.add_argument("--high-max-num", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.low_max_num < 1 or args.high_max_num <= args.low_max_num:
        raise ValueError("Require 1 <= --low-max-num < --high-max-num.")
    records = read_jsonl(Path(args.input))
    if not records:
        raise ValueError("Input manifest is empty.")
    if any("answer" in row or "cypher" in row for row in records):
        raise ValueError("Input must be answer-isolated public records.")
    if len({str(row["instance_id"]) for row in records}) != len(records):
        raise ValueError("Input manifest contains duplicate instance IDs.")

    model, tokenizer = load_model(args.model)
    output_dir = Path(args.output_dir)
    common = {
        "records": records,
        "model": model,
        "tokenizer": tokenizer,
        "image_root": Path(args.image_root),
        "run_id": args.run_id,
        "max_new_tokens": args.max_new_tokens,
        "set_id": args.set_id,
        "skip_existing": args.skip_existing,
    }
    run_cell(output_path=output_dir / "internvl35_b_tile_low.jsonl", label="low", max_num=args.low_max_num, **common)
    run_cell(output_path=output_dir / "internvl35_b_tile_high.jsonl", label="high", max_num=args.high_max_num, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
