"""Run the frozen full-image PaddleOCR baseline on the DEXPI v8 public set.

This process receives only an answer-isolated public manifest.  Candidate
tags are parsed with the external benchmark's public separator-stable grammar
and filtered by the question's public prefix.  Hidden XML-derived references
are never opened by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Iterable

from prepare_dexpi_external_v8 import tag_prefix, tags_in_text
from run_paddleocr_value_baseline_v1 import flatten_result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--det-limit-side-len", type=int, default=3072)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.det_limit_side_len <= 0:
        raise ValueError("--det-limit-side-len must be positive")

    input_path = Path(args.input).resolve()
    plan_path = Path(args.plan).resolve()
    output_path = Path(args.output).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_inference":
        raise ValueError("Plan is not frozen_before_inference")
    dataset = plan["datasets"][0]
    if sha256(input_path) != str(dataset["correct_sha256"]):
        raise ValueError("Public OCR input hash does not match the frozen plan")
    records = read_jsonl(input_path)
    if not records or len(records) != int(dataset["record_count"]):
        raise ValueError("Unexpected public OCR membership")
    if any(
        any("answer" in str(key).casefold() or "cypher" in str(key).casefold() for key in row)
        for row in records
    ):
        raise ValueError("Input is not answer-isolated")

    prior = read_jsonl(output_path) if output_path.is_file() else []
    rows_by_id = {
        str(row["instance_id"]): row
        for row in prior
        if str(row.get("status")) == "ok"
    }
    if args.skip_existing and len(rows_by_id) == len(records):
        print(json.dumps({"status": "already_complete", "rows": len(rows_by_id)}, sort_keys=True))
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
    plan_hash = sha256(plan_path)
    input_hash = sha256(input_path)
    for number, record in enumerate(records, start=1):
        instance_id = str(record["instance_id"])
        if instance_id in rows_by_id:
            continue
        started = time.perf_counter()
        try:
            image_path = Path(str(record["image_path"]))
            result = engine.ocr(str(image_path), cls=False)
            lines = flatten_result(result)
            combined = " | ".join(str(row["text"]) for row in lines)
            prefix = str(record.get("fields", {}).get("Prefix", "")).strip().casefold()
            if not prefix:
                raise ValueError("Public record lacks fields.Prefix")
            candidates = sorted(
                tag for tag in tags_in_text(combined) if tag_prefix(tag) == prefix
            )
            raw = ", ".join(candidates) if candidates else "[]"
            row: dict[str, Any] = {
                "instance_id": instance_id,
                "source_id": record["source_id"],
                "source_sheet": record.get("source_sheet"),
                "task": "value",
                "model": "PaddleOCR PP-OCRv4 English CPU",
                "model_revision": versions,
                "run_id": args.run_id,
                "dataset_id": "dexpi_external_v8",
                "condition_id": "paddleocr_full_image",
                "action": "ANSWER",
                "answer": raw,
                "raw": raw,
                "candidate_prefix": prefix,
                "candidate_tags": candidates,
                "ocr_text": combined,
                "ocr_lines": lines,
                "ocr_line_count": len(lines),
                "latency_seconds": time.perf_counter() - started,
                "det_limit_side_len": args.det_limit_side_len,
                "det_limit_type": "max",
                "use_angle_cls": False,
                "language": "en",
                "input_image_path": str(image_path),
                "crop_policy": "complete source image; no crop",
                "plan_sha256": plan_hash,
                "public_input_sha256": input_hash,
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
                "dataset_id": "dexpi_external_v8",
                "condition_id": "paddleocr_full_image",
                "action": "INVALID",
                "answer": None,
                "raw": "",
                "latency_seconds": time.perf_counter() - started,
                "plan_sha256": plan_hash,
                "public_input_sha256": input_hash,
                "test_answer_used": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows_by_id[instance_id] = row
        write_jsonl(output_path, [rows_by_id[key] for key in sorted(rows_by_id)])
        print(
            json.dumps(
                {
                    "progress": f"{number}/{len(records)}",
                    "instance_id": instance_id,
                    "status": row["status"],
                    "latency_seconds": row.get("latency_seconds"),
                    "ocr_line_count": row.get("ocr_line_count"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    failures = sum(str(row.get("status")) != "ok" for row in rows_by_id.values())
    status = "pass" if len(rows_by_id) == len(records) and failures == 0 else "fail"
    print(
        json.dumps(
            {"status": status, "rows": len(rows_by_id), "failures": failures, "versions": versions},
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
