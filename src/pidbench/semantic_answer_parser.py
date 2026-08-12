"""Conservative semantic parsing for PIDQA answer-format sensitivity audits.

The existing :mod:`pidbench.pidqa_metrics` normaliser remains the strict
metric contract.  This module is deliberately separate: it preserves the
strict score while exposing whether a model placed a recoverable task value in
otherwise non-conforming output such as ``"Yes."`` or ``"There are 2"``.

It never edits raw predictions and does not use hidden answers to parse model
outputs.  The tag grammar is intentionally narrow and documented below so the
semantic metric is reproducible rather than a free-form judge.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from pidbench.pidqa_metrics import normalize_pidqa_answer


BOOL_FULL_RE = re.compile(r"^\s*(yes|no|true|false)\s*$", re.IGNORECASE)
BOOL_PREFIX_RE = re.compile(r"^\s*(yes|no|true|false)\b", re.IGNORECASE)
INTEGER_TOKEN_RE = re.compile(r"(?<![\w.-])([+-]?\d+)(?![\w.-])")

# A P&ID tag is an alphabetic prefix followed by either a numeric suffix, a
# hyphen-separated numeric hierarchy, or a space-separated numeric suffix.
# It accepts examples such as KL-58999, UV-00-001, SDL 299, and CV1.  Bare
# uppercase tags (e.g. STA) are handled separately.
TAG_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,8}(?:-\d+(?:-\d+)*|\s+\d+|\d+))(?![A-Za-z0-9])"
)
BARE_TAG_RE = re.compile(r"(?<![A-Za-z0-9-])([A-Z]{2,8})(?![A-Za-z0-9-])")
BARE_TAG_STOPWORDS = {
    "THE",
    "AND",
    "ARE",
    "ANSWER",
    "ANSWERS",
    "CLASS",
    "FOR",
    "FROM",
    "LIST",
    "OF",
    "SYMBOL",
    "SYMBOLS",
    "TAG",
    "TAGS",
    "THESE",
    "WITH",
}
TAG_LIST_RE = re.compile(
    r"^[A-Za-z]{1,8}(?:(?:-\d+(?:-\d+)*)|(?:\s+\d+)|\d+)?"
    r"(?:\s*,\s*[A-Za-z]{1,8}(?:(?:-\d+(?:-\d+)*)|(?:\s+\d+)|\d+)?)*$"
)


@dataclass(frozen=True)
class SemanticParse:
    """A deterministic parse result for one task-aware answer."""

    task: str
    value: Any
    parsed: bool
    format_compliant: bool
    parser_rule: str


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def _bool_value(token: str) -> bool:
    return token.lower() in {"yes", "true"}


def _extract_tag_set(text: str) -> tuple[str, ...]:
    """Extract tags using the frozen public-output grammar.

    The parser does not know which tags occur in the hidden truth set.  It is
    therefore safe to apply to predictions before scoring.
    """

    found: list[str] = []
    structured_spans: list[tuple[int, int]] = []
    for match in TAG_RE.finditer(text):
        token = " ".join(match.group(1).split()).lower()
        prefix = re.split(r"[-\s\d]", token, maxsplit=1)[0]
        if prefix and prefix not in {word.lower() for word in BARE_TAG_STOPWORDS}:
            found.append(token)
            structured_spans.append(match.span(1))
    for match in BARE_TAG_RE.finditer(text):
        token = match.group(1)
        start, end = match.span(1)
        # ``SDL`` in ``SDL 299`` is a substring of an already accepted,
        # more-specific structured tag.  Do not turn it into a second tag.
        if any(span_start <= start and end <= span_end for span_start, span_end in structured_spans):
            continue
        if token not in BARE_TAG_STOPWORDS:
            found.append(token.lower())
    return tuple(sorted(set(found)))


def _value_format_compliant(text: str) -> bool:
    stripped = text.strip()
    if stripped in {"", "[]"}:
        return True
    if TAG_LIST_RE.fullmatch(stripped):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return False
        if not isinstance(parsed, (list, tuple, set)):
            return False
        return all(isinstance(item, str) and bool(item.strip()) for item in parsed)
    return False


def parse_semantic_answer(value: Any, task: str) -> SemanticParse:
    """Parse a PIDQA prediction without mutating its raw output.

    ``format_compliant`` is intentionally stricter than ``parsed``.  A model
    can therefore receive semantic credit while remaining visibly noncompliant
    with the concise-answer contract.
    """

    if task not in {"connectivity", "count", "spatial_count", "value"}:
        raise ValueError(f"Unsupported PIDQA task: {task!r}")
    text = _text(value)
    if text is None:
        return SemanticParse(task, None, False, False, "missing")

    if task == "connectivity":
        full = BOOL_FULL_RE.fullmatch(text)
        if full:
            return SemanticParse(task, _bool_value(full.group(1)), True, True, "boolean_full")
        prefix = BOOL_PREFIX_RE.match(text)
        if prefix:
            return SemanticParse(task, _bool_value(prefix.group(1)), True, False, "boolean_prefix")
        return SemanticParse(task, None, False, False, "boolean_unparsed")

    if task in {"count", "spatial_count"}:
        if re.fullmatch(r"[+-]?\d+", text):
            return SemanticParse(task, int(text), True, True, "integer_full")
        tokens = INTEGER_TOKEN_RE.findall(text)
        if len(tokens) == 1:
            return SemanticParse(task, int(tokens[0]), True, False, "integer_single_token")
        return SemanticParse(task, None, False, False, "integer_unparsed_or_ambiguous")

    # value task
    extracted = _extract_tag_set(text)
    if extracted:
        return SemanticParse(task, extracted, True, _value_format_compliant(text), "tag_grammar")
    strict_value = normalize_pidqa_answer(value, "value")
    if strict_value == () and text.strip() in {"", "[]"}:
        return SemanticParse(task, (), True, True, "empty_set")
    if strict_value and all(re.fullmatch(r"[A-Za-z]{2,8}", item) for item in strict_value):
        return SemanticParse(task, tuple(sorted(strict_value)), True, _value_format_compliant(text), "bare_tag_list")
    return SemanticParse(task, None, False, False, "tag_unparsed")


def strict_value(value: Any, task: str) -> Any:
    """Return the unchanged legacy normalisation used for strict scoring."""

    return normalize_pidqa_answer(value, task)
