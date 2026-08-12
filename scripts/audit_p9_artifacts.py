"""Audit hashes, prediction shards, and release-boundary artifacts for P9."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("row is not an object")
            rows.append(value)
        except Exception as exc:
            errors.append(f"line {number}: {type(exc).__name__}: {exc}")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    public_path = root / "data/processed/main400_source_test_diverse_public.jsonl"
    hidden_path = root / "data/answer_store/main400_source_test_diverse_hidden.jsonl"
    public_rows, public_errors = read_jsonl(public_path)
    hidden_rows, hidden_errors = read_jsonl(hidden_path)
    public_ids = {str(row.get("instance_id")) for row in public_rows}
    hidden_ids = {str(row.get("instance_id")) for row in hidden_rows}
    check("public_parse", not public_errors, str(public_errors))
    check("hidden_parse", not hidden_errors, str(hidden_errors))
    check("public_count", len(public_rows) == 400, str(len(public_rows)))
    check("hidden_count", len(hidden_rows) == 400, str(len(hidden_rows)))
    check("source_count", len({str(row.get("source_id")) for row in public_rows}) == 100, str(len({str(row.get("source_id")) for row in public_rows})))
    check("public_hidden_id_set", public_ids == hidden_ids, f"public={len(public_ids)} hidden={len(hidden_ids)} intersection={len(public_ids & hidden_ids)}")
    check("answer_isolation", all("answer" not in row and "cypher" not in row for row in public_rows), "public answer/cypher fields absent")

    output_paths = sorted((root / "outputs/main").glob("qwen3vl*source400*.jsonl"))
    output_entries = []
    for path in output_paths:
        rows, errors = read_jsonl(path)
        ids = {str(row.get("instance_id")) for row in rows}
        statuses = Counter(str(row.get("status")) for row in rows)
        entry = {
            "path": str(path.relative_to(root)),
            "row_count": len(rows),
            "unique_ids": len(ids),
            "parse_errors": errors,
            "ids_match_hidden": ids == hidden_ids,
            "status_counts": dict(statuses),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        output_entries.append(entry)
        check(f"output_count:{path.name}", len(rows) == 400 and len(ids) == 400, str(entry))
        check(f"output_ids:{path.name}", ids == hidden_ids and not errors, str(entry))
        check(f"output_status:{path.name}", statuses == Counter({"ok": 400}), str(entry))
    check("output_shard_count", len(output_entries) == 9, str(len(output_entries)))

    required = [
        "pyproject.toml",
        "LICENSES.md",
        "17_FIXED_NEXT_EXPERIMENT_ROUTE.md",
        "reports/MAINLINE_FINAL_CLOSEOUT.md",
        "reports/MANUSCRIPT_RESULTS_DRAFT_V2.md",
        "reports/SUBMISSION_PACKAGE_INDEX_V2.md",
        "reports/generated/final_mainline_manifest.json",
        "reports/generated/p9_clean_smoke.json",
        "reports/generated/main_efficiency_frontier.json",
        "reports/generated/qwen3vl8b_source400_resolution_table.json",
        "reports/generated/qwen3vl32b_source400_resolution_table.json",
        "reports/generated/qwen3vl8b_source400_degradation_table.json",
    ]
    artifacts = []
    for relative in required + [entry["path"] for entry in output_entries]:
        path = root / relative
        exists = path.exists() and path.is_file()
        item = {"path": relative, "exists": exists, "size_bytes": path.stat().st_size if exists else 0, "sha256": sha256(path) if exists else ""}
        artifacts.append(item)
        check(f"artifact:{relative}", exists, relative)

    smoke = json.loads((root / "reports/generated/p9_clean_smoke.json").read_text(encoding="utf-8"))
    check("clean_smoke_pass", smoke.get("status") == "pass" and smoke.get("smoke_score", {}).get("overall_accuracy") == 1.0, str(smoke.get("status")))
    final_manifest = json.loads((root / "reports/generated/final_mainline_manifest.json").read_text(encoding="utf-8"))
    check("previous_manifest_complete", final_manifest.get("missing_count") == 0, str(final_manifest.get("missing_count")))
    license_texts = list((root / "data/raw/PIDQA").glob("LICENSE*"))
    check("license_evidence_present", bool(license_texts), str([str(path) for path in license_texts]))
    model_files = [path for path in (root / "models").rglob("*")] if (root / "models").exists() else []
    check("no_model_weights_in_repo", not model_files, str(len(model_files)))

    passed = all(bool(check_item["ok"]) for check_item in checks)
    payload = {
        "status": "pass" if passed else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "check_count": len(checks),
        "passed_checks": sum(bool(item["ok"]) for item in checks),
        "checks": checks,
        "output_entries": output_entries,
        "artifacts": artifacts,
        "release_boundary": {
            "code_and_manifests": "publishable after project-code license decision",
            "model_weights": "not present; do not redistribute",
            "raw_external_archives": "not present/verified; release by reference only",
            "energy": "not measured",
        },
    }
    json_path, csv_path = Path(args.json), Path(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "exists", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(artifacts)
    print(json.dumps({"status": payload["status"], "check_count": payload["check_count"], "passed_checks": payload["passed_checks"], "output_shards": len(output_entries)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
