"""Run a dependency-free clean-install smoke for the released package.

The smoke installs the local package into a temporary target directory with no
dependency resolution, then uses that installed package to read a four-record
answer-isolated sample and score it through the repository scorer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    temp_root = Path(tempfile.mkdtemp(prefix="pidbench_p9_"))
    site = temp_root / "site"
    smoke = temp_root / "smoke"
    smoke.mkdir(parents=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result: dict[str, object] = {"status": "failed", "temporary_root": str(temp_root), "steps": []}
    try:
        install = run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(site), "."],
            cwd=root,
            env=env,
        )
        result["steps"].append({"name": "local_package_install", **install})
        if install["returncode"] != 0:
            raise RuntimeError("local package installation failed")

        installed_env = dict(env)
        installed_env["PYTHONPATH"] = str(site)
        import_check = run(
            [sys.executable, "-c", "from pidbench.io import read_jsonl; from pidbench.cli import main; print('installed_import_ok')"],
            cwd=root,
            env=installed_env,
        )
        result["steps"].append({"name": "installed_import", **import_check})
        if import_check["returncode"] != 0:
            raise RuntimeError("installed package import failed")

        source_records = [
            json.loads(line)
            for line in (root / "data/answer_store/main400_source_test_diverse_hidden.jsonl").read_text(encoding="utf-8").splitlines()[:4]
        ]
        public_records = []
        for record in source_records:
            public_records.append({key: record[key] for key in ("instance_id", "source_id", "source_sheet", "task", "question", "image_path")})
        public_path = smoke / "public4.jsonl"
        hidden_path = smoke / "hidden4.jsonl"
        prediction_path = smoke / "predictions4.jsonl"
        score_path = smoke / "score4.json"
        table_path = smoke / "table.json"
        table_csv_path = smoke / "table.csv"
        public_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in public_records) + "\n", encoding="utf-8")
        hidden_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in source_records) + "\n", encoding="utf-8")
        predictions = []
        for record in source_records:
            predictions.append({
                "instance_id": record["instance_id"],
                "source_id": record["source_id"],
                "source_sheet": record["source_sheet"],
                "task": record["task"],
                "model": "p9-clean-smoke",
                "mode": "direct",
                "action": "ANSWER",
                "answer": record["answer"],
                "raw": str(record["answer"]),
                "latency_seconds": 0.0,
                "status": "ok",
            })
        prediction_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")

        score = run(
            [sys.executable, "scripts/score_pidqa_subset_predictions.py", "--records", str(hidden_path), "--predictions", str(prediction_path), "--output", str(score_path), "--label", "p9_clean_smoke"],
            cwd=root,
            env=installed_env,
        )
        result["steps"].append({"name": "installed_package_scorer", **score})
        if score["returncode"] != 0:
            raise RuntimeError("clean smoke scorer failed")
        score_payload = json.loads(score_path.read_text(encoding="utf-8"))
        result["smoke_score"] = {
            "record_count": score_payload["record_count"],
            "prediction_count": score_payload["prediction_count"],
            "coverage": score_payload["coverage"],
            "overall_accuracy": score_payload["overall_accuracy"],
        }
        if score_payload["record_count"] != 4 or score_payload["prediction_count"] != 4 or score_payload["coverage"] != 1.0 or score_payload["overall_accuracy"] != 1.0:
            raise RuntimeError(f"unexpected clean smoke score: {result['smoke_score']}")

        table = run(
            [sys.executable, "scripts/build_mainline_score_table.py", "--score", f"p9_clean_smoke={score_path}", "--json", str(table_path), "--csv", str(table_csv_path)],
            cwd=root,
            env=installed_env,
        )
        result["steps"].append({"name": "table_rebuild", **table})
        if table["returncode"] != 0 or not table_path.exists() or not table_csv_path.exists():
            raise RuntimeError("table rebuild failed")
        result["status"] = "pass"
        return_code = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        shutil.rmtree(temp_root, ignore_errors=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
