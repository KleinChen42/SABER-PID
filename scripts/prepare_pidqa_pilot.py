"""Attach locally available Dataset-P&ID images to an answer-isolated pilot file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pidbench.io import read_jsonl, write_json, write_jsonl


def resolve_images(image_root: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        for image in image_root.rglob(extension):
            candidates.setdefault(image.stem, image)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    image_map = resolve_images(Path(args.image_root))
    rows = list(read_jsonl(args.input))
    missing = sorted(
        {str(row["source_sheet"]) for row in rows} - set(image_map)
    )
    if missing:
        raise FileNotFoundError(
            "Images are missing for source sheets: " + ", ".join(missing)
        )
    enriched = []
    for row in rows:
        image = image_map[str(row["source_sheet"])].resolve()
        try:
            relative = image.relative_to(project_root)
        except ValueError:
            relative = image
        enriched.append({**row, "image_path": relative.as_posix()})
    write_jsonl(args.output, enriched)
    summary = {
        "record_count": len(enriched),
        "source_count": len({row["source_id"] for row in enriched}),
        "image_root": str(Path(args.image_root)),
        "answer_isolated": all("answer" not in row for row in enriched),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
