"""Audits for same-drawing query exposure in a QA split.

This module deliberately models a *diagnostic cache*, not a vision-language
baseline.  It answers a held-out question only when the training partition
contains an unambiguous answer for the same source drawing and the same
released semantic query fields.  The audit therefore quantifies an evaluation
path a learned model could exploit when identical drawings occur on both sides
of a random QA split.  It must never be presented as an input-only VLM score.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def semantic_query_signature(record: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return the released task/field query signature used only for auditing."""

    fields = record.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"Record {record.get('instance_id')!r} has no semantic query fields")
    return (
        str(record["task"]),
        tuple(sorted((str(key), str(value)) for key, value in fields.items())),
    )


def build_same_source_cache_audit(
    records: Iterable[dict[str, Any]],
    assignments: Iterable[dict[str, Any]],
    *,
    train_split: str = "train",
    test_split: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build answer-isolated cache predictions and an exposure summary.

    Test answers are not read while building predictions.  ``source_id`` is an
    oracle proxy for exact drawing identity in PIDQA (one drawing per sheet),
    so the resulting score is an exposure diagnostic rather than a deployable
    method result.
    """

    rows = list(records)
    split_by_id = {
        str(row["instance_id"]): str(row["split"])
        for row in assignments
    }
    if len(split_by_id) != len(rows):
        missing = [str(row["instance_id"]) for row in rows if str(row["instance_id"]) not in split_by_id]
        if missing:
            raise ValueError(f"Split assignment missing {len(missing)} records; example: {missing[0]}")

    train_rows = [row for row in rows if split_by_id[str(row["instance_id"])] == train_split]
    test_rows = [row for row in rows if split_by_id[str(row["instance_id"])] == test_split]
    train_sources = {str(row["source_id"]) for row in train_rows}
    exact_question_keys = {
        (str(row["source_id"]), str(row["question"]))
        for row in train_rows
    }

    answer_values: dict[tuple[str, tuple[str, tuple[tuple[str, str], ...]]], set[str]] = defaultdict(set)
    answer_text: dict[tuple[str, tuple[str, tuple[tuple[str, str], ...]]], str] = {}
    for row in train_rows:
        cache_key = (str(row["source_id"]), semantic_query_signature(row))
        answer = str(row["answer"])
        answer_values[cache_key].add(answer)
        answer_text[cache_key] = answer

    unambiguous_cache = {
        cache_key: answer_text[cache_key]
        for cache_key, values in answer_values.items()
        if len(values) == 1
    }
    ambiguous_cache_keys = len(answer_values) - len(unambiguous_cache)

    totals = Counter()
    task_totals: dict[str, Counter[str]] = defaultdict(Counter)
    predictions: list[dict[str, Any]] = []
    for row in test_rows:
        task = str(row["task"])
        source_id = str(row["source_id"])
        cache_key = (source_id, semantic_query_signature(row))
        has_same_source = source_id in train_sources
        has_semantic_query = cache_key in answer_values
        has_exact_question = (source_id, str(row["question"])) in exact_question_keys
        cached_answer = unambiguous_cache.get(cache_key)

        totals["test_records"] += 1
        task_totals[task]["test_records"] += 1
        for name, condition in (
            ("same_source_records", has_same_source),
            ("semantic_query_overlap_records", has_semantic_query),
            ("exact_question_overlap_records", has_exact_question),
            ("unambiguous_cache_hits", cached_answer is not None),
        ):
            if condition:
                totals[name] += 1
                task_totals[task][name] += 1

        if cached_answer is not None:
            action, status, answer = "ANSWER", "cache_hit", cached_answer
        elif not has_same_source:
            action, status, answer = "ABSTAIN", "source_unseen", None
        elif has_semantic_query:
            action, status, answer = "ABSTAIN", "ambiguous_cache", None
        else:
            action, status, answer = "ABSTAIN", "query_unseen", None
        predictions.append(
            {
                "instance_id": str(row["instance_id"]),
                "source_id": source_id,
                "task": task,
                "action": action,
                "answer": answer,
                "status": status,
                "method": "same_source_semantic_cache_audit",
            }
        )

    def rate(counter: Counter[str], key: str) -> float:
        return counter[key] / counter["test_records"] if counter["test_records"] else 0.0

    summary = {
        "method": "same_source_semantic_cache_audit",
        "method_role": "diagnostic exposure audit, not an input-only VLM baseline",
        "identity_proxy": "source_id is an oracle proxy for exact drawing identity (one PIDQA sheet image per source)",
        "semantic_key": "task plus released query fields; used only to audit paraphrased semantic duplicates",
        "test_answers_used_to_build_predictions": False,
        "train_records": len(train_rows),
        "test_records": totals["test_records"],
        "train_source_count": len(train_sources),
        "ambiguous_train_cache_keys": ambiguous_cache_keys,
        "same_source_test_rate": rate(totals, "same_source_records"),
        "semantic_query_overlap_rate": rate(totals, "semantic_query_overlap_records"),
        "exact_question_overlap_rate": rate(totals, "exact_question_overlap_records"),
        "unambiguous_cache_hit_rate": rate(totals, "unambiguous_cache_hits"),
        "counts": dict(sorted(totals.items())),
        "by_task": {
            task: {
                "test_records": counter["test_records"],
                "same_source_test_rate": rate(counter, "same_source_records"),
                "semantic_query_overlap_rate": rate(counter, "semantic_query_overlap_records"),
                "exact_question_overlap_rate": rate(counter, "exact_question_overlap_records"),
                "unambiguous_cache_hit_rate": rate(counter, "unambiguous_cache_hits"),
                "counts": dict(sorted(counter.items())),
            }
            for task, counter in sorted(task_totals.items())
        },
    }
    return predictions, summary
