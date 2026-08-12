"""Audit materialized PID2Graph OPEN100 plans for a usable tag reference.

Node IDs are deliberately excluded: GraphML identifiers are internal graph
keys and are not evidence that a string is visible in the drawing.  A tag
reference is considered available only when a machine-readable data field is
explicitly named as text/tag/identifier metadata and contains a tag-like
value.  Node/edge class labels and geometry remain valid structural labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


TAG_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,8})\s*(?:[- ]\s*)?\d+(?:[.\-/]\d+)*(?![A-Za-z0-9])"
)
TAG_FIELD_HINTS = ("tag", "text", "identifier", "equipment", "instrument")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def graphml_schema(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    keys: dict[str, dict[str, str]] = {}
    for element in root.iter():
        if local_name(element.tag) != "key":
            continue
        key_id = str(element.attrib.get("id", ""))
        keys[key_id] = {
            "name": str(element.attrib.get("attr.name", key_id)),
            "for": str(element.attrib.get("for", "")),
            "type": str(element.attrib.get("attr.type", "")),
        }

    values: dict[str, list[str]] = defaultdict(list)
    node_count = 0
    edge_count = 0
    for element in root.iter():
        name = local_name(element.tag)
        if name == "node":
            node_count += 1
        elif name == "edge":
            edge_count += 1
        elif name == "data":
            key_id = str(element.attrib.get("key", ""))
            field = keys.get(key_id, {"name": key_id})["name"]
            value = " ".join((element.text or "").split())
            if value:
                values[field].append(value)

    tag_candidates: list[dict[str, str]] = []
    for field, field_values in values.items():
        if not any(hint in field.casefold() for hint in TAG_FIELD_HINTS):
            continue
        for value in field_values:
            for match in TAG_VALUE_RE.finditer(value):
                tag_candidates.append(
                    {"field": field, "value": value, "candidate": match.group(0)}
                )
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "keys": sorted(keys.values(), key=lambda row: (row["for"], row["name"])),
        "value_counts": {key: len(rows) for key, rows in sorted(values.items())},
        "value_examples": {
            key: sorted(set(rows))[:8] for key, rows in sorted(values.items())
        },
        "explicit_tag_reference_candidates": tag_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    catalog_path = Path(args.catalog).resolve()
    output_path = Path(args.output).resolve()
    pairs = []
    for graphml in sorted(root.rglob("*.graphml"), key=lambda path: path.as_posix()):
        image = graphml.with_suffix(".png")
        if not image.is_file():
            raise FileNotFoundError(f"Missing image pair for {graphml}")
        schema = graphml_schema(graphml)
        pairs.append(
            {
                "source_id": graphml.stem,
                "graphml": graphml.relative_to(root).as_posix(),
                "graphml_sha256": sha256(graphml),
                "image": image.relative_to(root).as_posix(),
                "image_sha256": sha256(image),
                **schema,
            }
        )
    if len(pairs) != 12:
        raise ValueError(f"Expected 12 complete OPEN100 plans, found {len(pairs)}")

    field_counts: Counter[str] = Counter()
    candidate_count = 0
    for row in pairs:
        field_counts.update(row["value_counts"])
        candidate_count += len(row["explicit_tag_reference_candidates"])
    report = {
        "version": "rineng-v8-pid2graph-open100-audit",
        "status": "pass",
        "official_source": {
            "record": "https://zenodo.org/records/14803338",
            "archive": "PID2Graph.zip",
            "official_size_bytes": 9303633645,
            "official_md5": "90f782220de97e7e249d2595c49ddc1c",
            "license": "CC BY-SA 4.0",
        },
        "transport": {
            "catalog": str(catalog_path),
            "catalog_sha256": sha256(catalog_path),
            "method": "HTTP byte-range sparse ZIP; 24 complete-plan members materialized",
            "complete_archive_downloaded": False,
        },
        "complete_plan_count": len(pairs),
        "aggregate_graphml_value_counts": dict(sorted(field_counts.items())),
        "explicit_tag_reference_candidate_count": candidate_count,
        "decision": {
            "structural_node_edge_class_reference": "available",
            "visible_text_tag_reference": "available" if candidate_count else "unavailable",
            "use_for_tag_retrieval_score": bool(candidate_count),
            "reason": (
                "Only explicitly named text/tag/identifier GraphML data fields may establish a tag reference; "
                "internal node IDs are never treated as visible drawing strings."
            ),
        },
        "plans": pairs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "plans": len(pairs),
                "explicit_tag_reference_candidates": candidate_count,
                "use_for_tag_retrieval_score": bool(candidate_count),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
