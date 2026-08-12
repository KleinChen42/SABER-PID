"""Strict-but-small structured-output parsing and model-visible graph checks."""

from __future__ import annotations

import json
from typing import Any


VALID_ACTIONS = {"ANSWER", "ABSTAIN", "INSUFFICIENT", "ESCALATE"}


def parse_structured_output(raw: str) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"json:{exc.msg}", "raw": raw}
    if not isinstance(result, dict):
        return {"ok": False, "error": "root_not_object", "raw": raw}
    action = result.get("action", "ANSWER")
    if action not in VALID_ACTIONS:
        return {"ok": False, "error": "invalid_action", "raw": raw}
    answer = result.get("answer")
    if action == "ANSWER" and answer is None:
        return {"ok": False, "error": "missing_answer", "raw": raw}
    return {"ok": True, "prediction": result, "raw": raw}


def check_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Check references internal to a prediction without looking at hidden truth."""

    violations: list[str] = []
    entities = prediction.get("entities") or []
    edges = prediction.get("edges") or []
    evidence = prediction.get("evidence") or []
    if not isinstance(entities, list) or not isinstance(edges, list) or not isinstance(evidence, list):
        return {"valid": False, "violations": ["non_list_graph_field"]}
    ids: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("local_id"):
            violations.append("missing_entity_id")
            continue
        ids.append(str(entity["local_id"]))
    id_set = set(ids)
    if len(ids) != len(id_set):
        violations.append("duplicate_entity_id")
    for edge in edges:
        if not isinstance(edge, dict):
            violations.append("invalid_edge")
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if not source or not target or source not in id_set or target not in id_set:
            violations.append("unknown_edge_endpoint")
        if source and source == target:
            violations.append("self_loop")
    for entity_id in evidence:
        if str(entity_id) not in id_set:
            violations.append("unknown_evidence")
    return {"valid": not violations, "violations": sorted(set(violations))}
