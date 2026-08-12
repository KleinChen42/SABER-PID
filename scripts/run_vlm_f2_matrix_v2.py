"""BOM-tolerant launcher for the frozen F2 matrix runner."""

from __future__ import annotations

import json
from pathlib import Path

import run_vlm_f2_matrix as base


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


base.read_jsonl = read_jsonl


if __name__ == "__main__":
    raise SystemExit(base.main())
