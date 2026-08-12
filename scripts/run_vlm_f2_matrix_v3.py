"""Optimized F2 launcher with BOM tolerance, image caching and action labels."""

from __future__ import annotations

import json
import time
from pathlib import Path

import run_vlm_f2_matrix as base


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


_IMAGE_CACHE = {}


def generate_one(model, processor, image_path, question, prompt_id, max_new_tokens, max_image_side):
    import torch
    from PIL import Image

    key = (str(image_path), int(max_image_side))
    image = _IMAGE_CACHE.get(key)
    if image is None:
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > max_image_side:
            image.thumbnail((max_image_side, max_image_side))
        _IMAGE_CACHE[key] = image.copy()
        image.close()
        image = _IMAGE_CACHE[key]
    original_size = tuple(int(v) for v in Image.open(image_path).size)
    resized_size = tuple(int(v) for v in image.size)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": base.prompt_for(question, prompt_id)}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    seconds = time.perf_counter() - started
    prompt_length = inputs.input_ids.shape[1]
    continuation = generated[:, prompt_length:]
    raw = processor.batch_decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return raw, seconds, original_size, resized_size, int(continuation.shape[1])


def write_jsonl(path, rows):
    for row in rows:
        if "action" not in row:
            row["action"] = "ANSWER" if row.get("status") == "ok" else "INVALID"
    return base._original_write_jsonl(path, rows)


base.read_jsonl = read_jsonl
base.generate_one = generate_one
base._original_write_jsonl = base.write_jsonl
base.write_jsonl = write_jsonl


if __name__ == "__main__":
    raise SystemExit(base.main())
