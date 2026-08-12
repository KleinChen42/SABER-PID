"""Run a frozen, full-image PaddleOCR baseline on 100 Set-B value sources.

The runner receives only the answer-isolated public manifest.  It applies one
English OCR pipeline to the complete source drawing (no crops), concatenates
recognized strings, extracts candidates with the project's frozen legal-tag
grammar, and retains candidates beginning with the question's public prefix.
No reference answer or test-time tuning is available to this process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Iterable

from pidbench.semantic_answer_parser import parse_semantic_answer


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


def value_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("task")) == "value"]
    if len(selected) != 100 or len({str(row["source_id"]) for row in selected}) != 100:
        raise ValueError("Expected exactly one value record for each of 100 sources")
    if any("answer" in row or "cypher" in row for row in selected):
        raise ValueError("Input is not answer-isolated")
    return selected


def flatten_result(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    pages = result if isinstance(result, list) else []
    for page in pages:
        if page is None:
            continue
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box, recognition = item[0], item[1]
            if not isinstance(recognition, (list, tuple)) or not recognition:
                continue
            text = str(recognition[0]).strip()
            confidence = float(recognition[1]) if len(recognition) > 1 else None
            if text:
                lines.append({"text": text, "confidence": confidence, "box": box})
    return lines


def extract_prediction(lines: list[dict[str, Any]], prefix: str) -> tuple[str, list[str], str]:
    combined = " | ".join(str(row["text"]) for row in lines)
    parsed = parse_semantic_answer(combined, "value")
    candidates = [] if not parsed.parsed or parsed.value is None else list(parsed.value)
    prefix_lower = prefix.strip().lower()
    matched = sorted({str(tag).lower() for tag in candidates if str(tag).lower().startswith(prefix_lower)})
    return (", ".join(matched) if matched else "[]"), matched, combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--det-limit-side-len", type=int, default=3072)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.det_limit_side_len <= 0:
        raise ValueError("--det-limit-side-len must be positive")
    records = value_records(read_jsonl(Path(args.input)))
    output_path = Path(args.output)
    # A failed row is diagnostic, not a completed observation.  Retain only
    # successful rows when resuming so a corrected runner can repair a failed
    # launch without creating duplicate instance IDs.
    existing = [
        row
        for row in (read_jsonl(output_path) if output_path.exists() else [])
        if row.get("status") == "ok"
    ]
    done = {str(row.get("instance_id")) for row in existing}
    if args.skip_existing and len(done) >= len(records):
        print(json.dumps({"status": "already_complete", "rows": len(existing)}, sort_keys=True))
        return 0

    from paddleocr import PaddleOCR

    versions = {
        name: importlib.metadata.version(name)
        for name in ("paddleocr", "paddlepaddle")
    }
    engine = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=False,
        show_log=False,
        det_limit_type="max",
        det_limit_side_len=args.det_limit_side_len,
        ocr_version="PP-OCRv4",
    )
    rows = list(existing)
    for number, record in enumerate(records, start=1):
        instance_id = str(record["instance_id"])
        if instance_id in done:
            continue
        started = time.perf_counter()
        try:
            image_path = resolve_path(str(record["image_path"]), Path(args.image_root))
            result = engine.ocr(str(image_path), cls=False)
            latency = time.perf_counter() - started
            lines = flatten_result(result)
            prefix = str(record.get("fields", {}).get("Prefix", "")).strip()
            if not prefix:
                raise ValueError("Public value record lacks fields.Prefix")
            answer, candidates, combined = extract_prediction(lines, prefix)
            row: dict[str, Any] = {
                "instance_id": instance_id,
                "source_id": record["source_id"],
                "source_sheet": record.get("source_sheet"),
                "task": "value",
                "model": "PaddleOCR PP-OCRv4 English CPU",
                "model_revision": versions,
                "run_id": args.run_id,
                "condition_id": "full_image_no_crop",
                "set_id": "B",
                "action": "ANSWER",
                "answer": answer,
                "raw": answer,
                "candidate_prefix": prefix,
                "candidate_tags": candidates,
                "ocr_text": combined,
                "ocr_lines": lines,
                "ocr_line_count": len(lines),
                "latency_seconds": latency,
                "det_limit_side_len": args.det_limit_side_len,
                "det_limit_type": "max",
                "use_angle_cls": False,
                "language": "en",
                "input_image_path": str(record["image_path"]),
                "crop_policy": "complete source image; no crop",
                "test_answer_used": False,
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "instance_id": instance_id,
                "source_id": record.get("source_id"),
                "source_sheet": record.get("source_sheet"),
                "task": "value",
                "run_id": args.run_id,
                "condition_id": "full_image_no_crop",
                "action": "INVALID",
                "answer": None,
                "raw": "",
                "latency_seconds": time.perf_counter() - started,
                "input_image_path": record.get("image_path"),
                "crop_policy": "complete source image; no crop",
                "test_answer_used": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        done.add(instance_id)
        write_jsonl(output_path, rows)
        print(json.dumps({"progress": f"{number}/{len(records)}", "instance_id": instance_id, "status": row["status"], "latency_seconds": row.get("latency_seconds"), "ocr_line_count": row.get("ocr_line_count")}, sort_keys=True), flush=True)
    print(json.dumps({"status": "complete", "rows": len(rows), "versions": versions}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
