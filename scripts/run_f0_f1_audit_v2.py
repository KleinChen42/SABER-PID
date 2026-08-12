"""Fast continuation of F0/F1 using source-level bootstrap pre-aggregation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pidbench.io import read_jsonl, write_json


def load_base():
    path = Path(__file__).with_name("build_f0_f1_audit.py")
    spec = importlib.util.spec_from_file_location("f0_f1_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fast_bootstrap_task_effects(
    records: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
    reps: int = 10000,
    seed: int = 17,
) -> list[dict[str, Any]]:
    base = load_base()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_id"])].append(record)
    sources = sorted(by_source)
    rows: list[dict[str, Any]] = []
    for task in base.TASKS:
        task_records = [record for record in records if str(record["task"]) == task]
        base_values = {
            str(record["instance_id"]): base.correctness(record, baseline[str(record["instance_id"])])
            for record in task_records
        }
        condition_values = {
            str(record["instance_id"]): base.correctness(record, condition[str(record["instance_id"])])
            for record in task_records
        }
        base_acc = sum(base_values.values()) / len(task_records)
        condition_acc = sum(condition_values.values()) / len(task_records)
        source_differences: dict[str, float] = {}
        for source, source_records in by_source.items():
            task_source_records = [record for record in source_records if str(record["task"]) == task]
            source_differences[source] = sum(
                condition_values[str(record["instance_id"])] - base_values[str(record["instance_id"])]
                for record in task_source_records
            ) / len(task_source_records)
        rng = random.Random(seed)
        differences = [
            sum(source_differences[rng.choice(sources)] for _ in sources) / len(sources)
            for _ in range(reps)
        ]
        rows.append(
            {
                "task": task,
                "record_count": len(task_records),
                "source_count": len(sources),
                "baseline_accuracy": base_acc,
                "condition_accuracy": condition_acc,
                "difference_condition_minus_baseline": condition_acc - base_acc,
                "bootstrap_ci95_low": base.quantile(differences, 0.025),
                "bootstrap_ci95_high": base.quantile(differences, 0.975),
                "bootstrap_reps": reps,
                "seed": seed,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    base = load_base()
    base.bootstrap_task_effects = fast_bootstrap_task_effects
    generated = root / "reports" / "generated"
    # Set B is regenerated from public identifiers; the existing F0 image audit
    # is reused because it is already complete and content-addressed.
    set_payload = base.build_hashblind_set(root, root / "data" / "processed")
    f1_payload = base.f1_audit(root, generated)
    image_payload = json.loads((generated / "pidqa_cross_source_duplicate_audit_v2.json").read_text(encoding="utf-8"))
    summary = {
        "status": "pass",
        "f0": {
            "image_count": image_payload["image_count"],
            "exact_cluster_count": image_payload["exact_cluster_count"],
            "ahash_cluster_count": image_payload["ahash_cluster_count"],
            "set_b": set_payload,
        },
        "f1": {
            "records": f1_payload["records"],
            "task_rows": len(f1_payload["task_rows"]),
            "tag_rows": len(f1_payload["tag_rows"]),
            "bootstrap_comparisons": len(f1_payload["bootstrap_effects"]),
            "bootstrap_method": "source-level pre-aggregated paired bootstrap",
        },
    }
    write_json(generated / "f0_f1_audit_summary_v2.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
