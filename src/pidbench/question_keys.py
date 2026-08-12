"""Input-text semantic keys for the four released PIDQA question families."""

from __future__ import annotations

import re
from typing import Any


_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?![A-Za-z])")
_UPPER_PREFIX = re.compile(r"\b[A-Z]{2,3}\b")


def question_semantic_signature(task: str, question: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Recover PIDQA's query arguments from visible question text alone.

    The parser is intentionally narrow and is only claimed for the released
    PIDQA templates.  It lets the exposure audit demonstrate that paraphrased
    duplicate queries do not require access to hidden graph answers or Cypher.
    """

    task = str(task)
    numbers = _NUMBER.findall(question)
    lowered = question.lower()
    if task == "count":
        if len(numbers) != 1:
            raise ValueError(f"Expected one class number in count question: {question!r}")
        fields: dict[str, str] = {"Symbol_Class": numbers[0]}
    elif task == "connectivity":
        if len(numbers) != 3:
            raise ValueError(f"Expected three class numbers in connectivity question: {question!r}")
        fields = dict(zip(("Symbol_XX", "Symbol_YY", "Symbol_ZZ"), numbers, strict=True))
    elif task == "spatial_count":
        if len(numbers) != 2:
            raise ValueError(f"Expected two class numbers in spatial-count question: {question!r}")
        # Two released templates describe the queried class after the connected
        # class ("given that they belong to ..." / trailing "with ...").
        # Preserve the released directed-query roles rather than sorting them.
        reversed_order = (
            "given that they belong to class" in lowered
            or re.search(r",\s*with\s+\d+\s*\??$", lowered) is not None
        )
        xx, yy = (numbers[1], numbers[0]) if reversed_order else (numbers[0], numbers[1])
        fields = {"Symbol_XX": xx, "Symbol_YY": yy}
    elif task == "value":
        if len(numbers) != 1:
            raise ValueError(f"Expected one class number in value question: {question!r}")
        prefixes = _UPPER_PREFIX.findall(question)
        if len(prefixes) != 1:
            raise ValueError(f"Expected one uppercase tag prefix in value question: {question!r}")
        fields = {"Prefix": prefixes[0], "Symbol_Class": numbers[0]}
    else:
        raise ValueError(f"Unsupported PIDQA task: {task!r}")
    return task, tuple(sorted(fields.items()))


def question_semantic_signature_for_record(record: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return question_semantic_signature(str(record["task"]), str(record["question"]))
