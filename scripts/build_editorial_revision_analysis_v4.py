"""Assemble deterministic v4 editorial evidence from frozen artifacts.

This script performs no model inference.  It selects the manuscript evidence
cases by pre-declared SHA-256 rules, assembles task-level counterfactual
effects, and exposes actual operating quantities for figures and tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from run_e1_evidence_audit import evaluate, read_rows


TASKS = ("connectivity", "count", "spatial_count", "value")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison(report: dict[str, Any], name: str, metric: str, task: str) -> dict[str, Any]:
    for row in report["comparisons"]:
        if row["comparison"] == name and row["metric"] == metric and row["task"] == task:
            return row
    raise KeyError(f"Missing comparison {name}/{metric}/{task}")


def reverse(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["baseline_mean"] = row["condition_mean"]
    result["condition_mean"] = row["baseline_mean"]
    result["difference_condition_minus_baseline"] = -float(row["difference_condition_minus_baseline"])
    result["source_bootstrap_ci95_low"] = -float(row["source_bootstrap_ci95_high"])
    result["source_bootstrap_ci95_high"] = -float(row["source_bootstrap_ci95_low"])
    return result


def value_tag_scores(event: dict[str, Any]) -> dict[str, Any]:
    truth = set(event["strict_truth_tags"])
    prediction = set(event["strict_prediction_tags"])
    tp = len(truth & prediction)
    precision = tp / len(prediction) if prediction else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else (1.0 if not prediction else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "parsed_tags": sorted(prediction),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set": int(truth == prediction),
    }


def cell_accuracy(report: dict[str, Any], cell: str, task: str) -> float:
    metrics = report["cells"][cell]["metrics"]
    return float(metrics["strict_accuracy"] if task == "overall" else metrics["task"][task]["strict_accuracy"])


def runtime_row(label: str, cell: dict[str, Any], *, input_basis: str, input_value: Any) -> dict[str, Any]:
    runtime = cell["runtime"]
    peak = runtime.get("peak_allocated_bytes_max")
    return {
        "label": label,
        "record_count": int(runtime["prediction_row_count"]),
        "input_basis": input_basis,
        "input_value": input_value,
        "max_new_tokens": cell.get("metadata", {}).get("max_new_tokens"),
        "output_token_mean": runtime.get("output_token_mean"),
        "output_token_p95": runtime.get("output_token_p95"),
        "token_cap_rate": runtime.get("token_cap_observed_rate"),
        "latency_seconds_mean": runtime.get("latency_seconds_mean"),
        "latency_seconds_p95": runtime.get("latency_seconds_p95"),
        "peak_allocated_gib": float(peak) / (1024**3) if peak is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", default="reports/generated/editorial_revision_evidence_v4.json")
    parser.add_argument("--csv", default="reports/generated/editorial_revision_task_effects_v4.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    generated = root / "reports/generated"

    hidden = read_rows(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl")
    public = read_rows(root / "data/processed/main400_hashblind_set_b_remote_public.jsonl")
    public_by_id = {str(row["instance_id"]): row for row in public}
    prediction_paths = {
        "correct_768": root / "outputs/final_replication/qwen8_b_p0_768.jsonl",
        "correct_3072": root / "outputs/final_replication/qwen8_b_p0_3072.jsonl",
        "shuffled_3072": root / "outputs/evidence_strengthening/qwen8_image_shuffle_v1/qwen8_image_shuffle_v1_3072.jsonl",
        "text_only": root / "outputs/evidence_strengthening/qwen8_text_only_v1/qwen8_text_only_v1_3072.jsonl",
    }
    events = {
        label: {str(row["instance_id"]): row for row in evaluate(hidden, read_rows(path))["events"]}
        for label, path in prediction_paths.items()
    }

    value_candidates: list[dict[str, Any]] = []
    structural_candidates: list[dict[str, Any]] = []
    for instance_id, high in events["correct_3072"].items():
        source_id = str(high["source_id"])
        rank = hashlib.sha256(f"{source_id}||{instance_id}".encode("utf-8")).hexdigest()
        if high["task"] == "value":
            condition_scores = {label: value_tag_scores(rows[instance_id]) for label, rows in events.items()}
            if condition_scores["correct_3072"]["f1"] == 1.0 and all(
                condition_scores[label]["f1"] == 0.0
                for label in ("correct_768", "shuffled_3072", "text_only")
            ):
                value_candidates.append({"rank_sha256": rank, "instance_id": instance_id, "source_id": source_id})
        if high["task"] == "spatial_count" and events["text_only"][instance_id]["strict_correct"] and not high["strict_correct"]:
            structural_candidates.append({"rank_sha256": rank, "instance_id": instance_id, "source_id": source_id})
    value_candidates.sort(key=lambda row: row["rank_sha256"])
    structural_candidates.sort(key=lambda row: row["rank_sha256"])
    if not value_candidates or not structural_candidates:
        raise RuntimeError("Strict deterministic evidence-case rule produced no eligible record")

    value_choice = value_candidates[0]
    value_id = value_choice["instance_id"]
    value_case = {
        **value_choice,
        "eligible_count": len(value_candidates),
        "selection_rule": "smallest SHA-256(source_id || '||' || instance_id) among records with correct-3072 strict tag F1=1 and correct-768/shuffled-3072/text-only strict tag F1=0",
        "question": public_by_id[value_id]["question"],
        "reference_tags": list(events["correct_3072"][value_id]["strict_truth"]),
        "image_path": "paper/assets/pidqa_sheet_282.jpg",
        "public_coordinate_status": "not_available; no crop or hand-selected box used",
        "conditions": {},
    }
    for label in ("correct_768", "correct_3072", "shuffled_3072", "text_only"):
        event = events[label][value_id]
        value_case["conditions"][label] = {
            "raw_output": event["raw_output"],
            **value_tag_scores(event),
        }

    structural_choice = structural_candidates[0]
    structural_id = structural_choice["instance_id"]
    structural_case = {
        **structural_choice,
        "eligible_count": len(structural_candidates),
        "selection_rule": "smallest SHA-256(source_id || '||' || instance_id) among spatial-count records that are strictly correct text-only and strictly wrong with the 3072 correct image",
        "question": public_by_id[structural_id]["question"],
        "reference": events["correct_3072"][structural_id]["strict_truth"],
        "text_only_output": events["text_only"][structural_id]["raw_output"],
        "correct_3072_output": events["correct_3072"][structural_id]["raw_output"],
        "image_path": "paper/assets/pidqa_sheet_184.jpg",
    }

    retrieval_report = read_json(generated / "pidqa_input_retrieval_seed_sweep.json")
    retrieval_points = []
    for seed in retrieval_report["seeds"]:
        pair: dict[str, Any] = {"seed": int(seed)}
        for split in ("random", "source"):
            rows = [
                row for row in retrieval_report["rows"]
                if row["method"] == "L5_image_semantic_with_prior" and row["split"] == split and int(row["seed"]) == int(seed)
            ]
            if len(rows) != 1:
                raise ValueError(f"Expected one L5 row for {split}/seed {seed}")
            pair[split] = float(rows[0]["overall_accuracy"])
            pair[f"{split}_train_records"] = int(rows[0]["train_records"])
            pair[f"{split}_test_records"] = int(rows[0]["test_records"])
        pair["gap"] = pair["random"] - pair["source"]
        retrieval_points.append(pair)

    prior = read_json(generated / "set_b_task_prior_v2.json")
    e2 = read_json(generated / "qwen8_value_budget_sensitivity_v1.json")
    e3 = read_json(generated / "image_dependence_control_v1.json")
    e4 = read_json(generated / "internvl_tile_budget_v1.json")
    e8 = read_json(generated / "text_only_image_grounding_control_v1.json")

    heatmap = []
    for task in (*TASKS, "overall"):
        prior_value = float(prior["metrics"]["strict_accuracy"] if task == "overall" else prior["metrics"]["task"][task]["strict_accuracy"])
        heatmap.append({
            "task": task,
            "task_prior": prior_value,
            "text_only": cell_accuracy(e8, "qwen8_b_p0_text_only", task),
            "shuffled": cell_accuracy(e8, "qwen8_b_p0_shuffled_3072", task),
            "correct": cell_accuracy(e8, "qwen8_b_p0_correct_3072", task),
        })

    task_effects: list[dict[str, Any]] = []
    for task in TASKS:
        metric = "strict_value_tag_f1" if task == "value" else "strict_correct"
        text = comparison(e8, "e8_correct_image_minus_text_only_3072", metric, task)
        shuffled = reverse(comparison(e3, "e3_shuffled_minus_correct_3072", metric, task))
        for contrast, row in (("correct_minus_text_only", text), ("correct_minus_shuffled", shuffled)):
            task_effects.append({
                "task": task,
                "metric": "strict tag F1" if task == "value" else "strict accuracy",
                "contrast": contrast,
                "baseline_mean": float(row["baseline_mean"]),
                "condition_mean": float(row["condition_mean"]),
                "difference": float(row["difference_condition_minus_baseline"]),
                "ci95_low": float(row["source_bootstrap_ci95_low"]),
                "ci95_high": float(row["source_bootstrap_ci95_high"]),
                "source_count": int(row["source_count"]),
            })

    operating_rows = [
        runtime_row("Qwen 768 correct, value, 512 cap", e2["cells"]["qwen8_b_p0_value_512_768"], input_basis="processor tensor elements", input_value=2211840),
        runtime_row("Qwen 3072 correct, value, 512 cap", e2["cells"]["qwen8_b_p0_value_512_3072"], input_basis="processor tensor elements", input_value=35979264),
        runtime_row("InternVL 1 tile, all tasks, 192 cap", e4["cells"]["internvl35_b_tile_low"], input_basis="tiles / tensor elements", input_value="1 / 602112"),
        runtime_row("InternVL 7 tiles, all tasks, 192 cap", e4["cells"]["internvl35_b_tile_high"], input_basis="tiles / tensor elements", input_value="7 / 4214784"),
    ]

    image_assets = [root / value_case["image_path"], root / structural_case["image_path"]]
    for path in image_assets:
        if not path.exists():
            raise FileNotFoundError(path)

    payload = {
        "status": "pass",
        "generator": "scripts/build_editorial_revision_analysis_v4.py",
        "inference_performed": False,
        "value_evidence_case": value_case,
        "structural_counterexample": structural_case,
        "retrieval_points": retrieval_points,
        "retrieval_gap_mean": sum(row["gap"] for row in retrieval_points) / len(retrieval_points),
        "heatmap_strict_accuracy": heatmap,
        "task_effects": task_effects,
        "operating_rows": operating_rows,
        "image_assets": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in image_assets
        ],
        "bootstrap": {"replicates": 10000, "unit": "source_id", "seeds": "comparison-specific fixed integers recorded in scoring scripts"},
        "scope": "Frozen Qwen results and existing corrected InternVL tile control only; optional new-model/OCR reports are scored separately.",
    }
    write_json(root / args.json, payload)

    csv_path = root / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_effects[0]))
        writer.writeheader()
        writer.writerows(task_effects)
    print(json.dumps({"status": "pass", "value_eligible": len(value_candidates), "structural_eligible": len(structural_candidates), "json": args.json, "csv": args.csv}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
