"""Run the frozen F2 Qwen3-VL prompt/resolution matrix.

This runner keeps each cell as a separate JSONL artifact while recording the
prompt, set, resolution, run id, image dimensions, model path and a compact
runtime provenance block on every row.  It accepts answer-isolated public
JSONL only; hidden answers are never read by the inference process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


PROMPTS: dict[str, str] = {
    "p0": (
        "You analyze a piping and instrumentation diagram (P&ID). "
        "Answer only from the visible diagram. Return only the final answer, "
        "without explanation or Markdown.\n\nQuestion: {question}"
    ),
    "p1": (
        "Read the P&ID image and answer the question using only visible symbols "
        "and connections. Give one concise final answer; do not explain your "
        "reasoning and do not use Markdown.\n\nQuestion: {question}"
    ),
    "p2": (
        "Inspect this P&ID and output only the requested value. For a yes/no "
        "question output Yes or No; for a counting question output one integer; "
        "for a symbol-list question output the visible class values as a comma- "
        "separated list; for a spatial/connectivity question output the requested "
        "yes/no or integer. Use no explanation or Markdown and use no information "
        "outside the image.\n\nQuestion: {question}"
    ),
}


def prompt_for(question: str, prompt_id: str) -> str:
    try:
        template = PROMPTS[prompt_id]
    except KeyError as exc:
        raise ValueError(f"unknown prompt id: {prompt_id}") from exc
    return template.format(question=question)


def prompt_hash(prompt_id: str) -> str:
    return hashlib.sha256(PROMPTS[prompt_id].encode("utf-8")).hexdigest()


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, processor


def generate_one(
    model,
    processor,
    image_path: Path,
    question: str,
    prompt_id: str,
    max_new_tokens: int,
    max_image_side: int,
) -> tuple[str, float, tuple[int, int], tuple[int, int], int]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    original_size = tuple(int(v) for v in image.size)
    if max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side))
    resized_size = tuple(int(v) for v in image.size)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_for(question, prompt_id)},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[rendered], images=[image], return_tensors="pt", padding=True
    )
    inputs = inputs.to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    seconds = time.perf_counter() - started
    prompt_length = inputs.input_ids.shape[1]
    continuation = generated[:, prompt_length:]
    raw = processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return raw, seconds, original_size, resized_size, int(continuation.shape[1])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-a", required=True)
    parser.add_argument("--input-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sets", default="A,B")
    parser.add_argument("--prompts", default="p0,p1,p2")
    parser.add_argument("--sides", default="768,3072")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    records_by_set = {
        "A": read_jsonl(Path(args.input_a)),
        "B": read_jsonl(Path(args.input_b)),
    }
    set_ids = [value.strip().upper() for value in args.sets.split(",") if value.strip()]
    prompt_ids = [value.strip().lower() for value in args.prompts.split(",") if value.strip()]
    sides = [int(value.strip()) for value in args.sides.split(",") if value.strip()]
    for set_id in set_ids:
        if set_id not in records_by_set:
            raise ValueError(f"unknown set: {set_id}")
    for prompt_id in prompt_ids:
        if prompt_id not in PROMPTS:
            raise ValueError(f"unknown prompt: {prompt_id}")

    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    model, processor = load_model(args.model)
    model_revision = str(Path(args.model).resolve())
    device = str(getattr(model, "device", "unknown"))
    total_cells = len(set_ids) * len(prompt_ids) * len(sides)
    completed_cells = 0
    for set_id in set_ids:
        records = records_by_set[set_id]
        for prompt_id in prompt_ids:
            for side in sides:
                cell_name = f"qwen8_{set_id.lower()}_{prompt_id}_{side}"
                output_path = output_dir / f"{cell_name}.jsonl"
                existing = read_jsonl(output_path) if output_path.exists() else []
                existing_ids = {str(row.get("instance_id")) for row in existing}
                rows = list(existing)
                if args.skip_existing and len(existing_ids) >= len(records):
                    completed_cells += 1
                    print(json.dumps({"cell": cell_name, "status": "already_complete", "rows": len(existing)}, sort_keys=True), flush=True)
                    continue
                for number, record in enumerate(records, start=1):
                    instance_id = str(record["instance_id"])
                    if instance_id in existing_ids:
                        continue
                    raw = ""
                    try:
                        image_path = Path(str(record["image_path"]))
                        if not image_path.is_absolute():
                            image_path = image_root / image_path
                        raw, latency, original_size, resized_size, output_tokens = generate_one(
                            model=model,
                            processor=processor,
                            image_path=image_path,
                            question=str(record["question"]),
                            prompt_id=prompt_id,
                            max_new_tokens=args.max_new_tokens,
                            max_image_side=side,
                        )
                        result = {
                            "instance_id": instance_id,
                            "source_id": record["source_id"],
                            "source_sheet": record["source_sheet"],
                            "task": record["task"],
                            "model": args.model,
                            "model_revision": model_revision,
                            "run_id": args.run_id,
                            "set_id": set_id,
                            "prompt_id": prompt_id,
                            "prompt_sha256": prompt_hash(prompt_id),
                            "max_image_side": side,
                            "mode": "direct",
                            "answer": raw,
                            "raw": raw,
                            "latency_seconds": latency,
                            "output_token_count": output_tokens,
                            "original_image_size": list(original_size),
                            "resized_image_size": list(resized_size),
                            "device": device,
                            "status": "ok",
                        }
                    except Exception as exc:
                        result = {
                            "instance_id": instance_id,
                            "source_id": record.get("source_id"),
                            "source_sheet": record.get("source_sheet"),
                            "task": record.get("task"),
                            "model": args.model,
                            "model_revision": model_revision,
                            "run_id": args.run_id,
                            "set_id": set_id,
                            "prompt_id": prompt_id,
                            "prompt_sha256": prompt_hash(prompt_id),
                            "max_image_side": side,
                            "mode": "direct",
                            "answer": None,
                            "raw": raw,
                            "latency_seconds": None,
                            "output_token_count": None,
                            "device": device,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(result)
                    existing_ids.add(instance_id)
                    write_jsonl(output_path, rows)
                    print(json.dumps({"cell": cell_name, "progress": f"{number}/{len(records)}", "instance_id": instance_id, "status": result["status"], "latency_seconds": result.get("latency_seconds")}, sort_keys=True), flush=True)
                completed_cells += 1
                print(json.dumps({"cell": cell_name, "status": "complete", "rows": len(rows), "completed_cells": f"{completed_cells}/{total_cells}"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
