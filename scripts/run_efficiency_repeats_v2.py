"""Run one frozen efficiency condition with warm-up and three repeats."""
from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, time
from pathlib import Path

def read(path: Path):
    with path.open("r", encoding="utf-8-sig") as h: return [json.loads(x) for x in h if x.strip()]

def append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h: h.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def gpu_uuid() -> str | None:
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().splitlines()
        return text[0].strip() if text else None
    except Exception: return None

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--model", required=True); p.add_argument("--family", choices=("qwen", "internvl"), required=True); p.add_argument("--condition", required=True); p.add_argument("--max-image-side", type=int, required=True); p.add_argument("--run-id", required=True); p.add_argument("--image-root", required=True); p.add_argument("--repeats", type=int, default=3); p.add_argument("--warmup", type=int, default=20); p.add_argument("--resume", action="store_true"); a = p.parse_args()
    import torch
    from PIL import Image
    records = read(Path(a.input)); out = Path(a.output); done = set()
    if a.resume and out.exists():
        for row in read(out):
            if row.get("condition") == a.condition and row.get("family") == a.family and row.get("max_image_side") == a.max_image_side and row.get("phase") == "measure": done.add((int(row.get("repeat", -1)), str(row.get("instance_id"))))
    device = next(iter(torch.cuda.device_count() and [torch.device("cuda:0")] or [torch.device("cpu")]))
    processor = model = tokenizer = None
    if a.family == "qwen":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(a.model); model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    else:
        import run_internvl35_f3_matrix_v4 as f3
        model, tokenizer = f3.load_model(a.model)
        from run_internvl35_f3_matrix import image_tensor
    image_root = Path(a.image_root); cache: dict[str, Image.Image] = {}
    def path_for(rec):
        x = Path(str(rec["image_path"])); return x if x.is_absolute() else image_root / x
    def infer(rec):
        raw = ""; source_path = path_for(rec); original = resized = None; tiles = None; pixels_count = None; output_tokens = None
        cpu_start = time.perf_counter();
        try:
            if a.family == "qwen":
                key = str(source_path); image = cache.get(key)
                if image is None:
                    image = Image.open(source_path).convert("RGB"); original = tuple(image.size); image.thumbnail((a.max_image_side, a.max_image_side)); cache[key] = image.copy(); image.close(); image = cache[key]
                else: original = tuple(image.size)
                resized = tuple(image.size)
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "You analyze a P&ID. Answer only from visible evidence. Return only the final answer, without explanation or Markdown.\n\nQuestion: " + str(rec["question"])}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True); inputs = processor(text=[rendered], images=[image], return_tensors="pt", padding=True).to(model.device); pixels_count = int(inputs["pixel_values"].numel()) if "pixel_values" in inputs else None
                if torch.cuda.is_available(): torch.cuda.synchronize()
                gen_start = time.perf_counter()
                with torch.inference_mode(): generated = model.generate(**inputs, max_new_tokens=192, do_sample=False)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                gen_latency = time.perf_counter() - gen_start; output_tokens = max(0, int(generated.shape[1] - inputs.input_ids.shape[1])); raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip(); tiles = 1
            else:
                pixels, original, resized, tiles, ratio = image_tensor(source_path, a.max_image_side); pixels = pixels.to(next(model.parameters()).device); pixels_count = int(pixels.numel()); question = "<image>\nYou analyze a piping and instrumentation diagram (P&ID). Answer only from the visible diagram. Return only the final answer, without explanation or Markdown.\n\nQuestion: " + str(rec["question"])
                if torch.cuda.is_available(): torch.cuda.synchronize()
                gen_start = time.perf_counter()
                with torch.inference_mode(): raw = model.chat(tokenizer, pixels, question, {"max_new_tokens": 192, "do_sample": False})
                if torch.cuda.is_available(): torch.cuda.synchronize()
                gen_latency = time.perf_counter() - gen_start; raw = raw[0] if isinstance(raw, tuple) else raw; raw = str(raw).strip(); output_tokens = len(tokenizer.encode(raw, add_special_tokens=False))
            cpu_latency = time.perf_counter() - cpu_start
            allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0; reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
            return {"answer": raw, "status": "ok", "cpu_latency_seconds": cpu_latency, "generate_latency_seconds": gen_latency, "output_token_count": output_tokens, "original_image_size": list(original) if original else None, "resized_image_size": list(resized) if resized else None, "dynamic_tile_count": tiles, "input_pixel_count": pixels_count, "gpu_peak_memory_allocated_bytes": allocated, "gpu_peak_memory_reserved_bytes": reserved}
        except Exception as exc:
            return {"answer": None, "status": "error", "cpu_latency_seconds": time.perf_counter() - cpu_start, "generate_latency_seconds": None, "output_token_count": None, "original_image_size": list(original) if original else None, "resized_image_size": list(resized) if resized else None, "dynamic_tile_count": tiles, "input_pixel_count": pixels_count, "gpu_peak_memory_allocated_bytes": None, "gpu_peak_memory_reserved_bytes": None, "error": f"{type(exc).__name__}: {exc}"}
    # Warm-up on a deterministic prefix for every condition/model invocation.
    for i, rec in enumerate(records[: a.warmup]):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        result = infer(rec); append(out, {"run_id": a.run_id, "family": a.family, "model": a.model, "condition": a.condition, "max_image_side": a.max_image_side, "phase": "warmup", "repeat": 0, "warmup_index": i, "instance_id": str(rec["instance_id"]), "source_id": rec.get("source_id"), "task": rec.get("task"), "gpu_uuid": gpu_uuid(), **result})
    for repeat in range(1, a.repeats + 1):
        key = int(hashlib.sha256(a.condition.encode()).hexdigest()[:8], 16); ordered = list(records); random.Random(20260805 + repeat * 1009 + key).shuffle(ordered)
        for rec in ordered:
            iid = str(rec["instance_id"])
            if (repeat, iid) in done: continue
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            result = infer(rec); append(out, {"run_id": a.run_id, "family": a.family, "model": a.model, "condition": a.condition, "max_image_side": a.max_image_side, "phase": "measure", "repeat": repeat, "instance_id": iid, "source_id": rec.get("source_id"), "task": rec.get("task"), "gpu_uuid": gpu_uuid(), **result})
            print(json.dumps({"condition": a.condition, "repeat": repeat, "progress": f"{len([x for x in ordered if (repeat, str(x['instance_id'])) not in done])}/{len(ordered)}", "status": result.get("status")}, sort_keys=True), flush=True)
    print(json.dumps({"status": "complete", "condition": a.condition, "family": a.family, "repeats": a.repeats, "records": len(records)}, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
