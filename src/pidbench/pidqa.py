"""PIDQA normalization with source-sheet identity retained."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import write_jsonl


PIDQA_FILES: tuple[tuple[str, str], ...] = (
    ("Simple Counting/simple_counting.csv", "count"),
    ("Spatial Counting/spatial_counting.csv", "spatial_count"),
    ("Spatial Connections/spatial_connectivity.csv", "connectivity"),
    ("Value/value_based.csv", "value"),
)


def _record_id(relative_path: str, row_number: int, source: str, question: str) -> str:
    digest = hashlib.sha256(
        f"{relative_path}|{row_number}|{source}|{question}".encode("utf-8")
    ).hexdigest()[:16]
    return f"pidqa-{digest}"


def _source_id(raw_source: str) -> str:
    value = str(raw_source).strip()
    try:
        return f"pidqa-sheet-{int(value):03d}"
    except ValueError:
        return f"pidqa-sheet-{value}"


def load_pidqa(root: str | Path) -> list[dict[str, Any]]:
    """Read official PIDQA CSV files into a common JSON-serializable record format."""

    root_path = Path(root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative, task in PIDQA_FILES:
        csv_path = root_path / relative
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                question = (row.get("Question") or "").strip()
                answer = (row.get("GT") or "").strip()
                raw_source = (row.get("P&ID_number") or "").strip()
                # In the official Value CSV an empty result is encoded as an
                # empty field rather than an explicit empty list. It is a
                # valid answer to a set-valued query, not a missing label.
                empty_value_answer = task == "value" and not answer
                if empty_value_answer:
                    answer = "[]"
                if not question or not answer or not raw_source:
                    raise ValueError(
                        f"Required PIDQA field missing in {csv_path}:{row_number}"
                    )
                records.append(
                    {
                        "instance_id": _record_id(relative, row_number, raw_source, question),
                        "dataset": "pidqa",
                        "source_id": _source_id(raw_source),
                        "source_sheet": raw_source,
                        "task": task,
                        "question": question,
                        "answer": answer,
                        "answer_was_implicit_empty_list": empty_value_answer,
                        "cypher": (row.get("Cypher") or "").strip(),
                        "question_template_id": (row.get("Q_id") or "").strip(),
                        "fields": {
                            key: value
                            for key, value in row.items()
                            if key
                            not in {"P&ID_number", "Type", "Question", "GT", "Cypher", "Q_id"}
                            and value not in (None, "")
                        },
                    }
                )
    if missing:
        raise FileNotFoundError("Missing expected PIDQA files: " + "; ".join(missing))
    if not records:
        raise ValueError(f"No PIDQA records found under {root_path}")
    return records


def summarize_pidqa(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    by_task = Counter(str(row["task"]) for row in records)
    by_source = Counter(str(row["source_id"]) for row in records)
    templates = Counter(str(row["question_template_id"]) for row in records)
    return {
        "record_count": len(records),
        "source_count": len(by_source),
        "task_counts": dict(sorted(by_task.items())),
        "questions_per_source": {
            "min": min(by_source.values()),
            "max": max(by_source.values()),
        },
        "template_count": len([item for item in templates if item]),
    }


def normalize_pidqa(root: str | Path, output: str | Path) -> dict[str, Any]:
    records = load_pidqa(root)
    write_jsonl(output, records)
    return summarize_pidqa(records)
