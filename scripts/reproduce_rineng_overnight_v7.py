"""Revalidate recovered V7 outputs and rebuild all V7 paper artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    environment = dict(os.environ)
    path_entries = [str(root / "src"), str(root / "scripts")]
    if environment.get("PYTHONPATH"):
        path_entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    commands = [
        [sys.executable, "scripts/validate_rineng_overnight_v7.py", "--root", str(root)],
        [sys.executable, "scripts/build_rineng_overnight_v7_paper_artifacts.py", "--root", str(root)],
        [sys.executable, "scripts/build_rineng_overnight_v7_artifact_manifest.py", "--root", str(root)],
    ]
    completed: list[str] = []
    for command in commands:
        subprocess.run(command, cwd=root, env=environment, check=True)
        completed.append(Path(command[1]).name)
    print(json.dumps({"status": "pass", "completed": completed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
