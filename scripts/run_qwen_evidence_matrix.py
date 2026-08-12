"""Run answer-isolated Qwen evidence-strengthening conditions.

This runner deliberately reuses the frozen F2 prompt and decoding semantics,
but gives E2/E3/E5 their own output namespace and richer per-row provenance.
It never reads an answer-bearing file: input records must be public manifests.

Supported controls:

* E2: ``--tasks value --max-new-tokens 512``;
* E3: a public manifest whose ``image_path`` has been source-shuffled;
* E5/E7: a single frozen legend image supplied through ``--legend-image``;
* E8: the same Qwen decoder and prompt with ``--text-only``.

Rows are appended atomically by rewrite after each completed instance so a
remote interruption can resume with ``--skip-existing``.  An output that
reaches the decoding ceiling is marked as *cap-reached*, not asserted to be a
semantically truncated answer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from run_vlm_f2_matrix import load_model, prompt_for, prompt_hash


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_csv(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def select_records(records: Iterable[dict[str, Any]], tasks: set[str]) -> list[dict[str, Any]]:
    """Return records in manifest order, optionally restricted to named tasks."""

    return [row for row in records if not tasks or str(row.get("task")) in tasks]


def resolve_path(value: str, image_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else image_root / path


def image_budget_metadata(inputs: Any) -> dict[str, Any]:
    """Expose processor output shape without guessing visual-token semantics."""

    pixels = inputs.get("pixel_values") if hasattr(inputs, "get") else None
    if pixels is None:
        return {
            "processor_pixel_values_shape": None,
            "actual_input_pixel_count": None,
            "actual_input_budget_status": "processor_pixel_values_not_exposed",
        }
    shape = [int(value) for value in pixels.shape]
    return {
        "processor_pixel_values_shape": shape,
        "actual_input_pixel_count": int(pixels.numel()),
        "actual_input_budget_status": "processor_pixel_values_recorded",
    }


def generate_one(
    *,
    model: Any,
    processor: Any,
    image_path: Path | None,
    question: str,
    prompt_id: str,
    max_new_tokens: int,
    max_image_side: int,
    legend_path: Path | None,
    text_only: bool,
) -> tuple[str, float, tuple[int, int] | None, tuple[int, int] | None, int, dict[str, Any]]:
    import torch
    from PIL import Image

    if text_only and legend_path is not None:
        raise ValueError("--text-only cannot be combined with --legend-image")
    if not text_only and image_path is None:
        raise ValueError("image_path is required unless --text-only is set")

    images: list[Any] = []
    content: list[dict[str, str]] = []
    original_size: tuple[int, int] | None = None
    resized_size: tuple[int, int] | None = None
    if not text_only:
        image = Image.open(image_path).convert("RGB")
        original_size = tuple(int(value) for value in image.size)
        if max(image.size) > max_image_side:
            image.thumbnail((max_image_side, max_image_side))
        resized_size = tuple(int(value) for value in image.size)
        images.append(image)
        content.append({"type": "image"})
    legend_original_size: tuple[int, int] | None = None
    if legend_path is not None:
        legend = Image.open(legend_path).convert("RGB")
        legend_original_size = tuple(int(value) for value in legend.size)
        if max(legend.size) > max_image_side:
            legend.thumbnail((max_image_side, max_image_side))
        images.append(legend)
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_for(question, prompt_id)})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor_kwargs: dict[str, Any] = {
        "text": [rendered],
        "return_tensors": "pt",
        "padding": True,
    }
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    inputs = inputs.to(model.device)
    budget = image_budget_metadata(inputs)
    budget["input_image_count"] = len(images)
    budget["legend_original_image_size"] = list(legend_original_size) if legend_original_size else None
    device_type = getattr(model.device, "type", str(model.device))
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    latency_seconds = time.perf_counter() - started
    if device_type == "cuda":
        budget["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(model.device))
    else:
        budget["peak_allocated_bytes"] = None
    prompt_length = inputs.input_ids.shape[1]
    continuation = generated[:, prompt_length:]
    raw = processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return (
        raw,
        latency_seconds,
        original_size,
        resized_size,
        int(continuation.shape[1]),
        budget,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Answer-isolated public JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--set-id", default="B")
    parser.add_argument("--prompt", default="p0")
    parser.add_argument("--tasks", default="", help="Comma-separated task filter.")
    parser.add_argument("--sides", default="768,3072")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--legend-image", default=None)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run the frozen Qwen prompt without any image content (E8 control).",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.prompt not in {"p0", "p1", "p2"}:
        raise ValueError(f"unknown prompt: {args.prompt!r}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    sides = [int(value) for value in parse_csv(args.sides)]
    if not sides or any(side <= 0 for side in sides):
        raise ValueError("--sides must contain positive integers")
    tasks = set(parse_csv(args.tasks))
    records = select_records(read_jsonl(Path(args.input)), tasks)
    if not records:
        raise ValueError("No records selected; check --tasks and --input.")
    duplicate_ids = len({str(row.get("instance_id")) for row in records}) != len(records)
    if duplicate_ids:
        raise ValueError("Input manifest has duplicate instance_id values.")

    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    legend_path = Path(args.legend_image) if args.legend_image else None
    if legend_path is not None and not legend_path.exists():
        raise FileNotFoundError(f"Legend image does not exist: {legend_path}")
    if args.text_only and legend_path is not None:
        raise ValueError("--text-only cannot be combined with --legend-image")
    model, processor = load_model(args.model)
    model_revision = str(Path(args.model).resolve())
    device = str(getattr(model, "device", "unknown"))

    for side in sides:
        output_path = output_dir / f"qwen8_{args.condition_id}_{side}.jsonl"
        existing = read_jsonl(output_path) if output_path.exists() else []
        existing_ids = {str(row.get("instance_id")) for row in existing}
        if args.skip_existing and len(existing_ids) >= len(records):
            print(json.dumps({"cell": output_path.stem, "status": "already_complete", "rows": len(existing)}, sort_keys=True), flush=True)
            continue
        rows = list(existing)
        for number, record in enumerate(records, start=1):
            instance_id = str(record["instance_id"])
            if instance_id in existing_ids:
                continue
            raw = ""
            try:
                image_path = (
                    None
                    if args.text_only
                    else resolve_path(str(record["image_path"]), image_root)
                )
                raw, latency, original_size, resized_size, output_tokens, budget = generate_one(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    question=str(record["question"]),
                    prompt_id=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    max_image_side=side,
                    legend_path=legend_path,
                    text_only=args.text_only,
                )
                result: dict[str, Any] = {
                    "instance_id": instance_id,
                    "source_id": record["source_id"],
                    "source_sheet": record.get("source_sheet"),
                    "image_source_id": record.get("image_source_id", record["source_id"]),
                    "task": record["task"],
                    "model": args.model,
                    "model_revision": model_revision,
                    "run_id": args.run_id,
                    "condition_id": args.condition_id,
                    "set_id": args.set_id,
                    "prompt_id": args.prompt,
                    "prompt_sha256": prompt_hash(args.prompt),
                    "max_image_side": side,
                    "max_new_tokens": args.max_new_tokens,
                    "answer": raw,
                    "raw": raw,
                    "latency_seconds": latency,
                    "output_token_count": output_tokens,
                    "output_reached_max_new_tokens": output_tokens >= args.max_new_tokens,
                    "original_image_size": list(original_size) if original_size else None,
                    "resized_image_size": list(resized_size) if resized_size else None,
                    "input_image_path": None if args.text_only else str(record["image_path"]),
                    "legend_image_path": str(legend_path) if legend_path else None,
                    "content_condition": (
                        "text_only"
                        if args.text_only
                        else "image_with_legend"
                        if legend_path is not None
                        else "image_only"
                    ),
                    "device": device,
                    "status": "ok",
                    **budget,
                }
            except Exception as exc:
                result = {
                    "instance_id": instance_id,
                    "source_id": record.get("source_id"),
                    "source_sheet": record.get("source_sheet"),
                    "image_source_id": record.get("image_source_id", record.get("source_id")),
                    "task": record.get("task"),
                    "model": args.model,
                    "model_revision": model_revision,
                    "run_id": args.run_id,
                    "condition_id": args.condition_id,
                    "set_id": args.set_id,
                    "prompt_id": args.prompt,
                    "prompt_sha256": prompt_hash(args.prompt),
                    "max_image_side": side,
                    "max_new_tokens": args.max_new_tokens,
                    "answer": None,
                    "raw": raw,
                    "latency_seconds": None,
                    "output_token_count": None,
                    "output_reached_max_new_tokens": None,
                    "input_image_path": None if args.text_only else record.get("image_path"),
                    "legend_image_path": str(legend_path) if legend_path else None,
                    "content_condition": "text_only" if args.text_only else "image_condition_error",
                    "device": device,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(result)
            existing_ids.add(instance_id)
            write_jsonl(output_path, rows)
            print(
                json.dumps(
                    {
                        "cell": output_path.stem,
                        "progress": f"{number}/{len(records)}",
                        "instance_id": instance_id,
                        "status": result["status"],
                        "latency_seconds": result.get("latency_seconds"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(json.dumps({"cell": output_path.stem, "status": "complete", "rows": len(rows)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
