"""Run a resumable Qwen correct/shuffled/text-only prompt matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_qwen_evidence_matrix import generate_one
from run_vlm_f2_matrix import PROMPTS, load_model, prompt_hash


CONDITIONS = ("correct", "shuffled", "text_only")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_pair(
    correct: list[dict[str, Any]], shuffled: list[dict[str, Any]], label: str
) -> None:
    for name, rows in (("correct", correct), ("shuffled", shuffled)):
        if not rows or any(
            any("answer" in str(key).lower() or "cypher" in str(key).lower() for key in row)
            for row in rows
        ):
            raise ValueError(f"{label}/{name}: manifest is empty or answer-bearing")
        ids = [str(row["instance_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label}/{name}: duplicate instance_id")
    if [str(row["instance_id"]) for row in correct] != [
        str(row["instance_id"]) for row in shuffled
    ]:
        raise ValueError(f"{label}: correct/shuffled membership or order differs")
    if any(
        str(row["source_id"]) == str(row.get("image_source_id"))
        for row in shuffled
    ):
        raise ValueError(f"{label}: shuffled manifest contains a fixed point")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompts", default="p0,p1")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--datasets", default="")
    parser.add_argument("--record-limit", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan_path = resolve_path(root, args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_inference":
        raise ValueError("Plan is not frozen_before_inference")
    prompts = parse_csv(args.prompts)
    conditions = parse_csv(args.conditions)
    requested_datasets = set(parse_csv(args.datasets))
    if not prompts or any(prompt not in PROMPTS for prompt in prompts):
        raise ValueError("Unknown prompt requested")
    if not conditions or any(condition not in CONDITIONS for condition in conditions):
        raise ValueError("Unknown condition requested")
    if args.max_image_side <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Image side and output cap must be positive")
    if args.record_limit < 0:
        raise ValueError("record-limit cannot be negative")

    datasets: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for spec in plan["datasets"]:
        if requested_datasets and str(spec["dataset_id"]) not in requested_datasets:
            continue
        correct_path = resolve_path(root, str(spec["correct_input"]))
        shuffled_path = resolve_path(root, str(spec["shuffled_input"]))
        if sha256(correct_path) != spec["correct_sha256"]:
            raise ValueError(f"{spec['dataset_id']}: correct manifest hash mismatch")
        if sha256(shuffled_path) != spec["shuffled_sha256"]:
            raise ValueError(f"{spec['dataset_id']}: shuffled manifest hash mismatch")
        correct = read_jsonl(correct_path)
        shuffled = read_jsonl(shuffled_path)
        validate_pair(correct, shuffled, str(spec["dataset_id"]))
        if args.record_limit:
            correct = correct[: args.record_limit]
            shuffled = shuffled[: args.record_limit]
        datasets.append((spec, correct, shuffled))

    model_path = resolve_path(root, args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(str(model_path))
    device = str(getattr(model, "device", "unknown"))
    plan_hash = sha256(plan_path)
    failures: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []

    for spec, correct, shuffled in datasets:
        dataset_id = str(spec["dataset_id"])
        for prompt_id in prompts:
            for condition in conditions:
                records = shuffled if condition == "shuffled" else correct
                output_path = output_dir / (
                    f"{args.model_label}_{dataset_id}_{prompt_id}_{condition}_"
                    f"{args.max_image_side}"
                    f"{'_smoke' + str(args.record_limit) if args.record_limit else ''}.jsonl"
                )
                prior = read_jsonl(output_path) if output_path.is_file() else []
                rows_by_id = {
                    str(row["instance_id"]): row
                    for row in prior
                    if str(row.get("status")) == "ok"
                }
                if args.skip_existing and len(rows_by_id) == len(records):
                    print(
                        json.dumps(
                            {"cell": output_path.stem, "status": "already_complete", "rows": len(rows_by_id)},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    cell_summaries.append(
                        {"cell": output_path.stem, "status": "pass", "rows": len(rows_by_id), "errors": 0}
                    )
                    continue
                for number, record in enumerate(records, start=1):
                    instance_id = str(record["instance_id"])
                    if instance_id in rows_by_id:
                        continue
                    raw = ""
                    try:
                        image_path = None
                        if condition != "text_only":
                            image_path = resolve_path(root, str(record["image_path"]))
                        raw, latency, original_size, resized_size, output_tokens, budget = generate_one(
                            model=model,
                            processor=processor,
                            image_path=image_path,
                            question=str(record["question"]),
                            prompt_id=prompt_id,
                            max_new_tokens=args.max_new_tokens,
                            max_image_side=args.max_image_side,
                            legend_path=None,
                            text_only=condition == "text_only",
                        )
                        result: dict[str, Any] = {
                            "instance_id": instance_id,
                            "source_id": record["source_id"],
                            "source_sheet": record.get("source_sheet"),
                            "image_source_id": None
                            if condition == "text_only"
                            else record.get("image_source_id", record["source_id"]),
                            "task": record["task"],
                            "model": str(model_path),
                            "model_label": args.model_label,
                            "run_id": args.run_id,
                            "dataset_id": dataset_id,
                            "condition_id": condition,
                            "prompt_id": prompt_id,
                            "prompt_sha256": prompt_hash(prompt_id),
                            "plan_sha256": plan_hash,
                            "max_image_side": args.max_image_side,
                            "max_new_tokens": args.max_new_tokens,
                            "action": "ANSWER",
                            "answer": raw,
                            "raw": raw,
                            "latency_seconds": latency,
                            "output_token_count": output_tokens,
                            "output_reached_max_new_tokens": output_tokens >= args.max_new_tokens,
                            "original_image_size": list(original_size) if original_size else None,
                            "resized_image_size": list(resized_size) if resized_size else None,
                            "input_image_path": None if condition == "text_only" else str(record["image_path"]),
                            "content_condition": condition,
                            "device": device,
                            "test_answer_used": False,
                            "status": "ok",
                            **budget,
                        }
                    except Exception as exc:
                        result = {
                            "instance_id": instance_id,
                            "source_id": record.get("source_id"),
                            "task": record.get("task"),
                            "model_label": args.model_label,
                            "run_id": args.run_id,
                            "dataset_id": dataset_id,
                            "condition_id": condition,
                            "prompt_id": prompt_id,
                            "prompt_sha256": prompt_hash(prompt_id),
                            "plan_sha256": plan_hash,
                            "max_image_side": args.max_image_side,
                            "max_new_tokens": args.max_new_tokens,
                            "action": "INVALID",
                            "answer": None,
                            "raw": raw,
                            "test_answer_used": False,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        failures.append({"cell": output_path.stem, **result})
                    rows_by_id[instance_id] = result
                    write_jsonl(output_path, [rows_by_id[key] for key in sorted(rows_by_id)])
                    print(
                        json.dumps(
                            {
                                "cell": output_path.stem,
                                "progress": f"{number}/{len(records)}",
                                "status": result["status"],
                                "latency_seconds": result.get("latency_seconds"),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                ok_count = sum(str(row.get("status")) == "ok" for row in rows_by_id.values())
                error_count = len(rows_by_id) - ok_count
                cell_summaries.append(
                    {
                        "cell": output_path.stem,
                        "status": "pass" if ok_count == len(records) and error_count == 0 else "fail",
                        "rows": len(rows_by_id),
                        "ok": ok_count,
                        "errors": error_count,
                    }
                )

    status = "pass" if not failures and all(row["status"] == "pass" for row in cell_summaries) else "fail"
    summary = {
        "version": "rineng-qwen-counterfactual-prompt-matrix-v7",
        "status": status,
        "model_label": args.model_label,
        "run_id": args.run_id,
        "plan": str(plan_path.relative_to(root).as_posix()),
        "plan_sha256": plan_hash,
        "requested_datasets": sorted(requested_datasets),
        "record_limit": args.record_limit,
        "cells": cell_summaries,
        "failure_count": len(failures),
        "failures": failures[:50],
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "summary": str(summary_path)}, sort_keys=True), flush=True)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
