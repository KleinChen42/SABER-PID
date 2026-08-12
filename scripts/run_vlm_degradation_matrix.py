"""Run Qwen3-VL-8B at 1536 px on one F5 degradation condition."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def read(path: Path):
    with path.open("r", encoding="utf-8-sig") as h: return [json.loads(line) for line in h if line.strip()]


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows: h.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--model", required=True); p.add_argument("--image-root", required=True); p.add_argument("--condition", required=True); p.add_argument("--run-id", required=True); p.add_argument("--max-image-side", type=int, default=1536); p.add_argument("--max-new-tokens", type=int, default=192); p.add_argument("--resume", action="store_true"); a = p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    records = read(Path(a.input)); output = Path(a.output); rows = read(output) if a.resume and output.exists() else []; done = {str(row.get("instance_id")) for row in rows}; cache = {}
    processor = AutoProcessor.from_pretrained(a.model); model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    for number, record in enumerate(records, start=1):
        iid = str(record["instance_id"])
        if iid in done: continue
        raw = ""
        try:
            path = Path(str(record["image_path"])); path = path if path.is_absolute() else Path(a.image_root) / path
            key = str(path)
            image = cache.get(key)
            if image is None:
                image = Image.open(path).convert("RGB"); image.thumbnail((a.max_image_side, a.max_image_side)); cache[key] = image.copy(); image.close(); image = cache[key]
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "You analyze a P&ID. Answer only from visible evidence. Return only the final answer, without explanation or Markdown.\n\nQuestion: " + str(record["question"])}]}]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True); inputs = processor(text=[rendered], images=[image], return_tensors="pt", padding=True).to(model.device); start = time.perf_counter()
            with torch.inference_mode(): generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            latency = time.perf_counter() - start; raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            row = {"instance_id": iid, "source_id": record["source_id"], "source_sheet": record["source_sheet"], "task": record["task"], "model": a.model, "model_revision": str(Path(a.model).resolve()), "run_id": a.run_id, "set_id": "B", "prompt_id": "p0", "condition": a.condition, "max_image_side": a.max_image_side, "action": "ANSWER", "answer": raw, "raw": raw, "latency_seconds": latency, "status": "ok"}
        except Exception as exc:
            row = {"instance_id": iid, "source_id": record.get("source_id"), "source_sheet": record.get("source_sheet"), "task": record.get("task"), "model": a.model, "run_id": a.run_id, "set_id": "B", "prompt_id": "p0", "condition": a.condition, "max_image_side": a.max_image_side, "action": "INVALID", "answer": None, "raw": raw, "latency_seconds": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row); done.add(iid); write(output, rows); print(json.dumps({"condition": a.condition, "progress": f"{number}/{len(records)}", "status": row["status"], "latency_seconds": row.get("latency_seconds")}, sort_keys=True), flush=True)
    print(json.dumps({"condition": a.condition, "status": "complete", "rows": len(rows)}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
