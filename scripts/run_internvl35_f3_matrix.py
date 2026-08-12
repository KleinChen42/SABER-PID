"""Run the frozen InternVL3.5-8B P0 resolution cross-family matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(image, image_size: int = 448, max_num: int = 12):
    target_ratios = sorted(
        {(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num},
        key=lambda value: value[0] * value[1],
    )
    width, height = image.size
    ratio = find_closest_aspect_ratio(width / height, target_ratios, width, height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for index in range(blocks):
        left = (index % (target_width // image_size)) * image_size
        top = (index // (target_width // image_size)) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles, ratio


def image_tensor(image_path: Path, max_side: int):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    original = tuple(int(value) for value in image.size)
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    resized = tuple(int(value) for value in image.size)
    tiles, ratio = dynamic_preprocess(image)
    tensors = []
    for tile in tiles:
        array = np.asarray(tile, dtype=np.float32) / 255.0
        array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors).to(dtype=torch.bfloat16), original, resized, len(tiles), ratio


def load_model(model_path: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-a", required=True)
    parser.add_argument("--input-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--sets", default="A,B")
    parser.add_argument("--sides", default="768,3072")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    import torch

    records_by_set = {"A": read_jsonl(Path(args.input_a)), "B": read_jsonl(Path(args.input_b))}
    sets = [value.strip().upper() for value in args.sets.split(",") if value.strip()]
    sides = [int(value.strip()) for value in args.sides.split(",") if value.strip()]
    model, tokenizer = load_model(args.model)
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    for set_id in sets:
        for side in sides:
            records = records_by_set[set_id]
            output_path = output_dir / f"internvl35_{set_id.lower()}_p0_{side}.jsonl"
            rows = read_jsonl(output_path) if output_path.exists() else []
            done = {str(row.get("instance_id")) for row in rows}
            cell = output_path.stem
            for number, record in enumerate(records, start=1):
                instance_id = str(record["instance_id"])
                if args.skip_existing and instance_id in done:
                    continue
                image_path = Path(str(record["image_path"]))
                if not image_path.is_absolute():
                    image_path = image_root / image_path
                raw = ""
                try:
                    pixels, original_size, resized_size, tile_count, tile_ratio = image_tensor(image_path, side)
                    pixels = pixels.to(next(model.parameters()).device)
                    question = "<image>\nYou analyze a piping and instrumentation diagram (P&ID). Answer only from the visible diagram. Return only the final answer, without explanation or Markdown.\n\nQuestion: " + str(record["question"])
                    started = time.perf_counter()
                    with torch.inference_mode():
                        raw = model.chat(tokenizer, pixels, question, generation_config)
                    latency = time.perf_counter() - started
                    if isinstance(raw, tuple):
                        raw = raw[0]
                    raw = str(raw).strip()
                    result = {
                        "instance_id": instance_id,
                        "source_id": record["source_id"],
                        "source_sheet": record["source_sheet"],
                        "task": record["task"],
                        "model": args.model,
                        "model_revision": str(Path(args.model).resolve()),
                        "run_id": args.run_id,
                        "set_id": set_id,
                        "prompt_id": "p0",
                        "max_image_side": side,
                        "action": "ANSWER",
                        "answer": raw,
                        "raw": raw,
                        "latency_seconds": latency,
                        "dynamic_tile_count": tile_count,
                        "dynamic_tile_ratio": list(tile_ratio),
                        "input_pixel_count": int(pixels.numel()),
                        "original_image_size": list(original_size),
                        "resized_image_size": list(resized_size),
                        "status": "ok",
                    }
                except Exception as exc:
                    result = {
                        "instance_id": instance_id,
                        "source_id": record.get("source_id"),
                        "source_sheet": record.get("source_sheet"),
                        "task": record.get("task"),
                        "model": args.model,
                        "model_revision": str(Path(args.model).resolve()),
                        "run_id": args.run_id,
                        "set_id": set_id,
                        "prompt_id": "p0",
                        "max_image_side": side,
                        "action": "INVALID",
                        "answer": None,
                        "raw": raw,
                        "latency_seconds": None,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(result); done.add(instance_id); write_jsonl(output_path, rows)
                print(json.dumps({"cell": cell, "progress": f"{number}/{len(records)}", "status": result["status"], "latency_seconds": result.get("latency_seconds")}, sort_keys=True), flush=True)
            print(json.dumps({"cell": cell, "status": "complete", "rows": len(rows)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
