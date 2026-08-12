"""Run a frozen InternVL correct/shuffled/text-only value ladder.

The runner reads only answer-isolated public manifests.  It holds model,
question order, prompt text, decoding, realised image-tile budget, and output
cap fixed while changing only the image-evidence condition.  Correct and
source-shuffled rows use InternVL's original-image dynamic preprocessing;
text-only rows pass no pixel tensor and omit the image placeholder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from run_internvl35_f3_matrix_v4 import load_model
from run_internvl_tile_budget_v1 import decode_token_count, image_tensor_original


PROMPT_BODY = (
    "You analyze a piping and instrumentation diagram (P&ID). "
    "Answer only from the visible diagram. Return only the final answer, "
    "without explanation or Markdown.\n\nQuestion: {question}"
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_BODY.encode("utf-8")).hexdigest()
CONDITIONS = ("correct", "shuffled", "text_only")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_path(value: str, image_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else image_root / path


def validate_manifest(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} manifest is empty")
    if any("answer" in row or "cypher" in row for row in rows):
        raise ValueError(f"{label} manifest is not answer-isolated")
    ids = [str(row["instance_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} manifest has duplicate instance IDs")


def select_value(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("task")) == "value"]
    if len(selected) != 100 or len({str(row["source_id"]) for row in selected}) != 100:
        raise ValueError("Expected exactly one value question for each of 100 sources")
    return selected


def run_condition(
    *,
    condition: str,
    records: list[dict[str, Any]],
    output_path: Path,
    model: Any,
    tokenizer: Any,
    image_root: Path,
    run_id: str,
    max_num: int,
    max_new_tokens: int,
    skip_existing: bool,
) -> None:
    import torch

    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {str(row.get("instance_id")) for row in existing}
    if skip_existing and len(done) >= len(records):
        print(json.dumps({"cell": output_path.stem, "status": "already_complete", "rows": len(existing)}, sort_keys=True), flush=True)
        return
    rows = list(existing)
    device = next(model.parameters()).device
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    for number, record in enumerate(records, start=1):
        instance_id = str(record["instance_id"])
        if instance_id in done:
            continue
        raw = ""
        try:
            pixels = None
            tile_count = 0
            tile_ratio = None
            original_size = None
            input_pixel_count = 0
            image_path = None
            if condition != "text_only":
                image_path = resolve_path(str(record["image_path"]), image_root)
                pixels, original_size, tile_count, tile_ratio = image_tensor_original(image_path, max_num)
                pixels = pixels.to(device)
                input_pixel_count = int(pixels.numel())
            question = PROMPT_BODY.format(question=str(record["question"]))
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
                "image_source_id": None if condition == "text_only" else record.get("image_source_id", record["source_id"]),
                "task": record["task"],
                "model": str(getattr(model, "name_or_path", "InternVL3.5-8B")),
                "model_revision": str(getattr(model, "name_or_path", "InternVL3.5-8B")),
                "run_id": run_id,
                "condition_id": condition,
                "set_id": "B",
                "prompt_id": "internvl_p0_matched",
                "prompt_sha256": PROMPT_SHA256,
                "tokenizer_fix_mistral_regex": True,
                "action": "ANSWER",
                "answer": raw,
                "raw": raw,
                "latency_seconds": latency,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_new_tokens": max_new_tokens,
                "output_token_count": output_tokens,
                "output_reached_max_new_tokens": output_tokens >= max_new_tokens if output_tokens is not None else None,
                "dynamic_preprocess_max_num": max_num if condition != "text_only" else 0,
                "dynamic_tile_count": tile_count,
                "dynamic_tile_ratio": list(tile_ratio) if tile_ratio is not None else None,
                "input_pixel_count": input_pixel_count,
                "original_image_size": list(original_size) if original_size else None,
                "input_image_path": str(record["image_path"]) if condition != "text_only" else None,
                "content_condition": condition,
                "device": str(device),
                "status": "ok",
            }
        except Exception as exc:
            result = {
                "instance_id": instance_id,
                "source_id": record.get("source_id"),
                "source_sheet": record.get("source_sheet"),
                "image_source_id": None if condition == "text_only" else record.get("image_source_id", record.get("source_id")),
                "task": record.get("task"),
                "run_id": run_id,
                "condition_id": condition,
                "set_id": "B",
                "prompt_id": "internvl_p0_matched",
                "prompt_sha256": PROMPT_SHA256,
                "tokenizer_fix_mistral_regex": True,
                "action": "INVALID",
                "answer": None,
                "raw": raw,
                "max_new_tokens": max_new_tokens,
                "output_token_count": None,
                "output_reached_max_new_tokens": None,
                "input_image_path": None if condition == "text_only" else record.get("image_path"),
                "content_condition": condition,
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
    parser.add_argument("--correct-input", required=True)
    parser.add_argument("--shuffled-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--max-num", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    requested = parse_csv(args.conditions)
    if not requested or any(value not in CONDITIONS for value in requested):
        raise ValueError(f"--conditions must be drawn from {CONDITIONS}")
    correct = select_value(read_jsonl(Path(args.correct_input)))
    shuffled = select_value(read_jsonl(Path(args.shuffled_input)))
    validate_manifest(correct, "correct")
    validate_manifest(shuffled, "shuffled")
    if [str(row["instance_id"]) for row in correct] != [str(row["instance_id"]) for row in shuffled]:
        raise ValueError("Correct and shuffled manifests do not have identical value-instance order")
    if any(str(row["source_id"]) == str(row.get("image_source_id")) for row in shuffled):
        raise ValueError("Shuffled manifest contains a fixed point")
    model, tokenizer = load_model(args.model)
    for condition in requested:
        records = shuffled if condition == "shuffled" else correct
        run_condition(
            condition=condition,
            records=records,
            output_path=Path(args.output_dir) / f"internvl35_8b_value_{condition}.jsonl",
            model=model,
            tokenizer=tokenizer,
            image_root=Path(args.image_root),
            run_id=args.run_id,
            max_num=args.max_num,
            max_new_tokens=args.max_new_tokens,
            skip_existing=args.skip_existing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
