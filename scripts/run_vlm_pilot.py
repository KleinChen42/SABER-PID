"""Run a small image-question pilot with a Hugging Face vision-language model.

The script deliberately accepts answer-isolated JSONL only. It writes raw model
outputs and a best-effort answer field; hidden truth is scored elsewhere.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def prompt_for(question: str, mode: str) -> str:
    if mode == "structured":
        return (
            "You analyze a piping and instrumentation diagram (P&ID). "
            "Answer only from the visible diagram. Return exactly one JSON object, "
            "with no Markdown, using this schema: "
            '{"action":"ANSWER|ABSTAIN|INSUFFICIENT","answer":value,'
            '"entities":[],"edges":[],"evidence":[],"confidence":0.0}. '
            "Use ANSWER only when the answer is visible.\n\n"
            f"Question: {question}"
        )
    return (
        "You analyze a piping and instrumentation diagram (P&ID). "
        "Answer only from the visible diagram. Return only the final answer, "
        "without explanation or Markdown.\n\n"
        f"Question: {question}"
    )


def extract_prediction(raw: str, mode: str) -> tuple[str, Any]:
    if mode == "structured":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return "INVALID", None
        action = str(parsed.get("action", "ANSWER"))
        return action, parsed.get("answer")
    return "ANSWER", raw.strip()


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


def generate_one(model, processor, image_path: Path, question: str, mode: str, max_new_tokens: int, max_image_side: int) -> tuple[str, float]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_for(question, mode)},
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
    return raw, seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("direct", "structured"), default="structured")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-image-side", type=int, default=1536)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from pidbench.io import read_jsonl, write_jsonl

    records = list(read_jsonl(args.input))
    if args.max_samples:
        records = records[: args.max_samples]
    output = Path(args.output)
    completed: set[str] = set()
    existing: list[dict[str, Any]] = []
    if args.resume and output.exists():
        existing = list(read_jsonl(output))
        completed = {str(row["instance_id"]) for row in existing}

    model, processor = load_model(args.model)
    results = list(existing)
    for number, record in enumerate(records, start=1):
        instance_id = str(record["instance_id"])
        if instance_id in completed:
            continue
        image_path = Path(str(record["image_path"]))
        try:
            raw, latency_seconds = generate_one(
                model=model,
                processor=processor,
                image_path=image_path,
                question=str(record["question"]),
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                max_image_side=args.max_image_side,
            )
            action, answer = extract_prediction(raw, args.mode)
            result = {
                "instance_id": instance_id,
                "source_id": record["source_id"],
                "source_sheet": record["source_sheet"],
                "task": record["task"],
                "model": args.model,
                "mode": args.mode,
                "action": action,
                "answer": answer,
                "raw": raw,
                "latency_seconds": latency_seconds,
                "status": "ok",
            }
        except Exception as exc:  # Keep failures visible in the pilot report.
            result = {
                "instance_id": instance_id,
                "source_id": record["source_id"],
                "source_sheet": record["source_sheet"],
                "task": record["task"],
                "model": args.model,
                "mode": args.mode,
                "action": "INVALID",
                "answer": None,
                "raw": "",
                "latency_seconds": None,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        write_jsonl(output, results)
        print(
            json.dumps(
                {
                    "progress": f"{number}/{len(records)}",
                    "instance_id": instance_id,
                    "status": result["status"],
                    "latency_seconds": result["latency_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
