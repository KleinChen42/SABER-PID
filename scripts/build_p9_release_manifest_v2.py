"""Extend the P9 release manifest with the final closeout report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    previous = json.loads((root / "reports/generated/p9_release_manifest.json").read_text(encoding="utf-8"))
    paths = list(dict.fromkeys([row["path"] for row in previous["rows"]] + ["reports/P9_REPRODUCIBILITY_CLOSEOUT.md", "scripts/build_p9_release_manifest_v2.py"]))
    rows = []
    for relative in paths:
        path = root / relative
        exists = path.exists() and path.is_file()
        rows.append({"path": relative, "exists": exists, "size_bytes": path.stat().st_size if exists else 0, "sha256": digest(path) if exists else ""})
    payload = {"manifest_version": "p9-release-v2", "artifact_count": len(rows), "missing_count": sum(not row["exists"] for row in rows), "rows": rows, "claim_boundary": previous["claim_boundary"]}
    json_path, csv_path = Path(args.json), Path(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "exists", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest_version": payload["manifest_version"], "artifact_count": payload["artifact_count"], "missing_count": payload["missing_count"]}, indent=2))
    return 0 if payload["missing_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
