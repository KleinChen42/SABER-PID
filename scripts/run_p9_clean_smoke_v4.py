"""Isolated package-path smoke with public/hidden records joined by instance ID."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    isolate = root / ".p9_source_isolated_v4"
    if isolate.exists():
        shutil.rmtree(isolate, ignore_errors=True)
    site, smoke = isolate / "site", isolate / "smoke"
    site.mkdir(parents=True)
    smoke.mkdir(parents=True)
    shutil.copytree(root / "src/pidbench", site / "pidbench")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(site)
    result: dict[str, object] = {"status": "failed", "isolation_root": str(isolate), "steps": []}
    try:
        for name, command in [
            ("isolated_compile", [sys.executable, "-m", "compileall", "-q", str(site / "pidbench")]),
            ("isolated_import", [sys.executable, "-c", "from pidbench.io import read_jsonl; from pidbench.cli import main; print('isolated_import_ok')"]),
        ]:
            step = run(command, root, env)
            result["steps"].append({"name": name, **step})
            if step["returncode"] != 0:
                raise RuntimeError(f"{name} failed")

        public_all = [json.loads(line) for line in (root / "data/processed/main400_source_test_diverse_public.jsonl").read_text(encoding="utf-8").splitlines()]
        hidden_all = {row["instance_id"]: row for row in (json.loads(line) for line in (root / "data/answer_store/main400_source_test_diverse_hidden.jsonl").read_text(encoding="utf-8").splitlines())}
        public = public_all[:4]
        hidden = [hidden_all[row["instance_id"]] for row in public]
        public_path, hidden_path, prediction_path = smoke / "public4.jsonl", smoke / "hidden4.jsonl", smoke / "predictions4.jsonl"
        score_path, table_path, table_csv_path = smoke / "score.json", smoke / "table.json", smoke / "table.csv"
        public_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in public) + "\n", encoding="utf-8")
        hidden_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in hidden) + "\n", encoding="utf-8")
        predictions = []
        for record in hidden:
            predictions.append({"instance_id": record["instance_id"], "source_id": record["source_id"], "source_sheet": record["source_sheet"], "task": record["task"], "model": "p9-isolated-smoke", "mode": "direct", "action": "ANSWER", "answer": record["answer"], "raw": str(record["answer"]), "latency_seconds": 0.0, "status": "ok"})
        prediction_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")
        score_step = run([sys.executable, "scripts/score_pidqa_subset_predictions.py", "--records", str(hidden_path), "--predictions", str(prediction_path), "--output", str(score_path), "--label", "p9_isolated_smoke"], root, env)
        result["steps"].append({"name": "isolated_scorer", **score_step})
        if score_step["returncode"] != 0:
            raise RuntimeError("isolated scorer failed")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        result["smoke_score"] = {key: score[key] for key in ("record_count", "prediction_count", "coverage", "overall_accuracy")}
        if result["smoke_score"] != {"record_count": 4, "prediction_count": 4, "coverage": 1.0, "overall_accuracy": 1.0}:
            raise RuntimeError(f"unexpected score: {result['smoke_score']}")
        table_step = run([sys.executable, "scripts/build_mainline_score_table.py", "--score", f"p9_isolated_smoke={score_path}", "--json", str(table_path), "--csv", str(table_csv_path)], root, env)
        result["steps"].append({"name": "isolated_table_rebuild", **table_step})
        if table_step["returncode"] != 0 or not table_path.exists() or not table_csv_path.exists():
            raise RuntimeError("isolated table rebuild failed")
        result["status"] = "pass"
        code = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
