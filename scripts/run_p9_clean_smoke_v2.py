"""Workspace-temp wrapper for the P9 clean smoke on restricted Windows hosts."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


def main() -> int:
    original = tempfile.mkdtemp
    tempfile.mkdtemp = lambda prefix="tmp": original(prefix=prefix, dir=r"C:\tmp")  # type: ignore[assignment]
    path = Path(__file__).with_name("run_p9_clean_smoke.py")
    spec = importlib.util.spec_from_file_location("p9_clean_smoke_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
