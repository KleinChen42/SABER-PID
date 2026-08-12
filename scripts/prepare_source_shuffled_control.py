"""Create the public, source-level image-shuffle control for E3.

The input must be an answer-isolated PIDQA manifest.  A deterministic Sattolo
cycle maps every question source to a *different* source image, preserving
questions, task labels, instance IDs, and the complete per-task distribution.
No answer, Cypher query, or model output is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sattolo_derangement(source_ids: list[str], seed: int) -> list[str]:
    """Return a seeded single-cycle permutation, hence no source is fixed."""

    if len(source_ids) < 2:
        raise ValueError("At least two distinct sources are required for a derangement.")
    shuffled = list(source_ids)
    rng = random.Random(seed)
    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = rng.randrange(index)
        shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    if any(left == right for left, right in zip(source_ids, shuffled)):
        raise AssertionError("Sattolo permutation unexpectedly contained a fixed point.")
    return shuffled


def build_shuffle(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    images_by_source: dict[str, set[str]] = defaultdict(set)
    for row in records:
        source_id = str(row["source_id"])
        image_path = str(row["image_path"])
        images_by_source[source_id].add(image_path)
    source_ids = sorted(images_by_source)
    if any(len(paths) != 1 for paths in images_by_source.values()):
        multi = sorted(source for source, paths in images_by_source.items() if len(paths) != 1)
        raise ValueError(f"Expected exactly one original image per source; got conflicts for {multi[:10]}")
    shuffled_sources = sattolo_derangement(source_ids, seed)
    mapping = dict(zip(source_ids, shuffled_sources))
    image_by_source = {source: next(iter(paths)) for source, paths in images_by_source.items()}
    output: list[dict[str, Any]] = []
    for row in records:
        source_id = str(row["source_id"])
        image_source_id = mapping[source_id]
        result = dict(row)
        result["original_image_path"] = str(row["image_path"])
        result["image_source_id"] = image_source_id
        result["image_path"] = image_by_source[image_source_id]
        result["image_control"] = "source_shuffled_no_fixed_point_v1"
        output.append(result)
    return output, mapping, image_by_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/main400_hashblind_set_b_public.jsonl")
    parser.add_argument("--manifest", default="data/manifests/set_b_source_shuffle_v1.json")
    parser.add_argument("--output", default="data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_jsonl(input_path)
    if not records:
        raise ValueError("Input manifest is empty.")
    if any("answer" in row or "cypher" in row for row in records):
        raise ValueError("Input is not answer-isolated; refusing to construct E3 control.")
    shuffled, mapping, images = build_shuffle(records, args.seed)
    if {str(row["instance_id"]) for row in shuffled} != {str(row["instance_id"]) for row in records}:
        raise AssertionError("Shuffle changed the instance-id set.")
    if any(str(row["source_id"]) == str(row["image_source_id"]) for row in shuffled):
        raise AssertionError("Shuffle retained an image from its own source.")
    input_tasks: dict[str, int] = defaultdict(int)
    output_tasks: dict[str, int] = defaultdict(int)
    for row in records:
        input_tasks[str(row["task"])] += 1
    for row in shuffled:
        output_tasks[str(row["task"])] += 1
    if dict(input_tasks) != dict(output_tasks):
        raise AssertionError("Shuffle changed the task distribution.")

    mapping_rows = [{"question_source_id": source, "image_source_id": mapping[source], "image_path": images[mapping[source]]} for source in sorted(mapping)]
    # Image roots differ between the local release layout and H200's flat
    # image mirror.  Hash only the source pairing so both environments can
    # prove they used the identical derangement while retaining their own
    # resolvable image paths in the full manifest.
    mapping_canonical_rows = [
        {"question_source_id": row["question_source_id"], "image_source_id": row["image_source_id"]}
        for row in mapping_rows
    ]
    mapping_canonical = json.dumps(mapping_canonical_rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    manifest = {
        "status": "pass",
        "control": "source_shuffled_no_fixed_point_v1",
        "algorithm": "Sattolo single-cycle derangement over sorted source_id values",
        "seed": args.seed,
        "input_path": str(input_path.as_posix()),
        "input_sha256": sha256_file(input_path),
        "record_count": len(records),
        "source_count": len(mapping),
        "task_counts": dict(sorted(input_tasks.items())),
        "mapping_sha256": hashlib.sha256(mapping_canonical).hexdigest(),
        "fixed_point_count": 0,
        "mapping": mapping_rows,
    }
    write_jsonl(Path(args.output), shuffled)
    write_json(Path(args.manifest), manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "record_count", "source_count", "mapping_sha256", "fixed_point_count")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
