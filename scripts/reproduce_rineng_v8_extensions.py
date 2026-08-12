"""Inference-free reproduction of all RINENG V8 scores, intervals, figures, and tables."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def reproduction_commands(root: Path, python: str) -> list[list[str]]:
    return [
        [python, "scripts/build_cost_sensitive_operating_modes_v8.py", "--root", str(root)],
        [python, "scripts/score_rineng_v8_extensions.py", "--root", str(root), "--bootstrap-reps", "10000"],
        [python, "scripts/score_dexpi_external_v8.py", "--root", str(root), "--bootstrap-reps", "10000"],
        [python, "scripts/validate_rineng_v8_extensions.py", "--root", str(root), "--bootstrap-reps", "10000"],
        [python, "scripts/build_rineng_v8_extension_figures.py", "--root", str(root)],
        [python, "scripts/build_rineng_v8_tables.py", "--root", str(root)],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    environment = dict(os.environ)
    path_entries = [str(root / "src"), str(root / "scripts")]
    if environment.get("PYTHONPATH"):
        path_entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    completed: list[str] = []
    for command in reproduction_commands(root, sys.executable):
        subprocess.run(command, cwd=root, env=environment, check=True)
        completed.append(Path(command[1]).name)
    if not args.skip_tests:
        tests = [
            "tests/test_cost_sensitive_operating_modes_v8.py",
            "tests/test_rineng_v8_extension_preparation.py",
            "tests/test_prepare_dexpi_external_v8.py",
            "tests/test_score_rineng_v8_extensions.py",
            "tests/test_score_dexpi_external_v8.py",
            "tests/test_audit_pid2graph_open100_v8.py",
            "tests/test_build_rineng_v8_extension_figures.py",
            "tests/test_build_rineng_v8_tables.py",
            "tests/test_validate_rineng_v8_extensions.py",
        ]
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
            cwd=root,
            env=environment,
            check=True,
        )
        completed.append("v8_pytest")
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": "inference-free",
                "bootstrap_reps": 10_000,
                "completed": completed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
