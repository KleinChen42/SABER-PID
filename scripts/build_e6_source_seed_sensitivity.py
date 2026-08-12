"""Build answer-blind E6 source-split sensitivity manifests.

E6 is a pre-registered sensitivity unit, not a selection procedure.  For each
requested source-split seed, it uses the public source, task, and instance
identifiers to select one question from each task for every source in the
split's test partition.  The selected answers are written only to the local
answer store; remote manifests are answer-isolated and point to a dedicated
remote image directory that is populated separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pidbench.io import read_jsonl, write_json, write_jsonl
from pidbench.splits import make_source_split


TASKS = ("connectivity", "count", "spatial_count", "value")
PUBLIC_EXCLUDED = {"answer", "cypher"}
SELECTION_LABEL = "e6-source-seed-resolution-v1"


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_image_map(root: Path) -> dict[str, str]:
    image_root = root / "data" / "raw" / "pidqa_images"
    mapping: dict[str, str] = {}
    for split in ("train", "val"):
        for image in sorted((image_root / split).glob("*.jpg")):
            source_id = f"pidqa-sheet-{int(image.stem):03d}"
            if source_id in mapping:
                raise RuntimeError(f"Duplicate local image for {source_id}")
            mapping[source_id] = image.relative_to(root).as_posix()
    if len(mapping) != 500:
        raise RuntimeError(f"Expected 500 PIDQA source images, found {len(mapping)}")
    return mapping


def public_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in PUBLIC_EXCLUDED}


def select_seed(
    *, root: Path, raw_records: list[dict[str, Any]], public_records: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    assignments = make_source_split(public_records, seed)
    assignment_by_id = {str(row["instance_id"]): str(row["split"]) for row in assignments}
    test_sources = sorted(
        {str(row["source_id"]) for row in public_records if assignment_by_id[str(row["instance_id"])] == "test"}
    )
    if len(test_sources) != 100:
        raise RuntimeError(f"Seed {seed} expected 100 test sources, found {len(test_sources)}")

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in public_records:
        instance_id = str(row["instance_id"])
        if assignment_by_id[instance_id] == "test":
            candidates[(str(row["source_id"]), str(row["task"]))].append(row)

    selected_public: list[dict[str, Any]] = []
    for source_id in test_sources:
        for task in TASKS:
            options = candidates[(source_id, task)]
            if not options:
                raise RuntimeError(f"Seed {seed} has no public candidate for {source_id}/{task}")
            selected_public.append(
                min(
                    options,
                    key=lambda row: stable_key(SELECTION_LABEL, seed, source_id, task, row["instance_id"]),
                )
            )
    selected_public.sort(key=lambda row: (str(row["source_id"]), str(row["task"]), str(row["instance_id"])))
    selected_ids = {str(row["instance_id"]) for row in selected_public}
    raw_by_id = {str(row["instance_id"]): row for row in raw_records}
    if selected_ids - set(raw_by_id):
        raise RuntimeError("Selected records are missing from the local answer store source")

    images = source_image_map(root)
    image_provenance = [
        {
            "source_id": source_id,
            "local_image_path": images[source_id],
            "sha256": sha256_file(root / images[source_id]),
        }
        for source_id in test_sources
    ]
    local_public: list[dict[str, Any]] = []
    remote_public: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for public in selected_public:
        source_id = str(public["source_id"])
        if source_id not in images:
            raise RuntimeError(f"Missing local PIDQA image for {source_id}")
        source_sheet = str(public["source_sheet"])
        local_row = dict(public)
        local_row["image_path"] = images[source_id]
        local_row["e6_source_split_seed"] = seed
        local_public.append(local_row)

        remote_row = dict(local_row)
        remote_row["image_path"] = f"data/raw/e6_source_seed_v1/{source_sheet}.jpg"
        remote_public.append(remote_row)
        hidden.append(dict(raw_by_id[str(public["instance_id"])]))

    for row in (*local_public, *remote_public):
        if PUBLIC_EXCLUDED & set(row):
            raise RuntimeError("Public E6 manifest contains an answer-bearing field")
    if len({str(row["instance_id"]) for row in local_public}) != 400:
        raise RuntimeError(f"Seed {seed} did not produce 400 unique selections")
    if Counter(str(row["task"]) for row in local_public) != Counter({task: 100 for task in TASKS}):
        raise RuntimeError(f"Seed {seed} task balance is invalid")

    processed = root / "data" / "processed"
    answer_store = root / "data" / "answer_store"
    manifests = root / "data" / "manifests"
    local_path = processed / f"source_seed{seed}_resolution_v1_public.jsonl"
    remote_path = processed / f"source_seed{seed}_resolution_v1_remote_public.jsonl"
    hidden_path = answer_store / f"source_seed{seed}_resolution_v1_hidden.jsonl"
    manifest_path = manifests / f"source_seed{seed}_resolution_v1.json"
    write_jsonl(local_path, local_public)
    write_jsonl(remote_path, remote_public)
    write_jsonl(hidden_path, hidden)
    write_json(
        manifest_path,
        {
            "status": "pass",
            "experiment": "E6",
            "selection_label": SELECTION_LABEL,
            "source_split_seed": seed,
            "source_split": "full_PIDQA_500_source_three_way_60_20_20",
            "selection": "one candidate per test source/task by SHA-256 rank over public identifiers only",
            "answer_isolated_remote_input": True,
            "record_count": len(local_public),
            "source_count": len(test_sources),
            "task_counts": dict(sorted(Counter(str(row["task"]) for row in local_public).items())),
            "local_public_path": local_path.relative_to(root).as_posix(),
            "remote_public_path": remote_path.relative_to(root).as_posix(),
            "hidden_path": hidden_path.relative_to(root).as_posix(),
            "local_public_sha256": sha256_file(local_path),
            "remote_public_sha256": sha256_file(remote_path),
            "hidden_sha256": sha256_file(hidden_path),
            "remote_image_directory": "data/raw/e6_source_seed_v1",
            "required_image_source_ids": test_sources,
            "required_image_count": len(test_sources),
            "image_paths_local": sorted({str(row["image_path"]) for row in local_public}),
            "required_image_provenance": image_provenance,
            "public_excluded_fields": sorted(PUBLIC_EXCLUDED),
        },
    )
    return {
        "seed": seed,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "record_count": len(local_public),
        "source_count": len(test_sources),
        "remote_public_sha256": sha256_file(remote_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=[29, 31])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain one or more non-negative integer seeds")
    raw_records = list(read_jsonl(root / "data" / "processed" / "pidqa_records.jsonl"))
    public_records = [public_projection(row) for row in raw_records]
    if any(PUBLIC_EXCLUDED & set(row) for row in public_records):
        raise RuntimeError("Public selection projection is not answer-isolated")
    results = [select_seed(root=root, raw_records=raw_records, public_records=public_records, seed=seed) for seed in seeds]
    print(json.dumps({"status": "pass", "experiment": "E6", "seeds": results}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
