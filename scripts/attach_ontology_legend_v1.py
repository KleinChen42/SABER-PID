"""Attach the frozen E5 public visual legend to an answer-isolated manifest.

This small transport-safe step is used on H200 because its public image mirror
uses different relative paths from the local release layout.  It preserves all
question fields and only adds immutable control metadata; it rejects any input
row that exposes an answer or Cypher query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legend", required=True)
    args = parser.parse_args()
    legend = Path(args.legend)
    if not legend.exists():
        raise FileNotFoundError(f"Legend image does not exist: {legend}")
    records = read_jsonl(Path(args.input))
    if not records:
        raise ValueError("Input manifest is empty.")
    if any("answer" in row or "cypher" in row for row in records):
        raise ValueError("Input must be answer-isolated public records.")
    output: list[dict[str, Any]] = []
    for row in records:
        result = dict(row)
        result["ontology_control"] = "public_training_symbol_prototype_legend_v1"
        result["ontology_legend_path"] = legend.as_posix()
        result["ontology_legend_sha256"] = sha256(legend)
        output.append(result)
    write_jsonl(Path(args.output), output)
    print(json.dumps({"status": "pass", "record_count": len(output), "legend": legend.as_posix(), "legend_sha256": sha256(legend)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
