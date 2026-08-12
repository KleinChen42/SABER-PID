"""Build the frozen E5 visual class-key from public PIDQA training metadata.

Dataset-P&ID exposes integer class labels and symbol bounding boxes, but not
human-interpretable class names.  This script therefore makes no semantic-name
claim: it renders a fixed image showing ``Class 01`` through ``Class 32`` with
one deterministic public-training exemplar crop per integer class.

Only source-split training images and ``*_symbols.npy`` metadata are read.
The script refuses answer-bearing input manifests, never reads PIDQA answers or
Cypher, and records all selected source/bounding-box provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CLASS_IDS = tuple(range(1, 33))
SOURCE_URLS = {
    "pidqa": "https://github.com/mgupta70/PIDQA",
    "dataset_pid": "https://arxiv.org/abs/2109.03794",
}


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


def source_number(source_id: str) -> int:
    prefix = "pidqa-sheet-"
    if not source_id.startswith(prefix):
        raise ValueError(f"Unexpected PIDQA source_id: {source_id!r}")
    return int(source_id.removeprefix(prefix))


def source_image_path(image_root: Path, source_id: str) -> Path:
    number = source_number(source_id)
    for split in ("train", "val"):
        candidate = image_root / split / f"{number}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No public image found for {source_id} under {image_root}")


def training_sources(split_manifest: Path) -> set[str]:
    rows = read_jsonl(split_manifest)
    train = {str(row["source_id"]) for row in rows if str(row.get("split")) == "train"}
    if not train:
        raise ValueError(f"No train sources found in {split_manifest}")
    return train


def select_exemplars(dataset_root: Path, image_root: Path, train_sources: set[str]) -> dict[int, dict[str, Any]]:
    """Select one smallest `(source number, symbol index)` crop per class."""

    import numpy as np

    selected: dict[int, dict[str, Any]] = {}
    for source_id in sorted(train_sources, key=source_number):
        number = source_number(source_id)
        symbol_path = dataset_root / str(number) / f"{number}_symbols.npy"
        if not symbol_path.exists():
            continue
        image_path = source_image_path(image_root, source_id)
        symbols = np.load(symbol_path, allow_pickle=True)
        for index, row in enumerate(symbols):
            try:
                class_id = int(row[2])
                bbox = [int(value) for value in row[1]]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid symbol row in {symbol_path} at index {index}: {row!r}") from exc
            if class_id not in CLASS_IDS or class_id in selected:
                continue
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox for class {class_id} in {symbol_path}: {bbox!r}")
            selected[class_id] = {
                "class_id": class_id,
                "source_id": source_id,
                "source_number": number,
                "symbol_index": index,
                "bbox_xyxy": bbox,
                "symbol_annotation_path": symbol_path.as_posix(),
                "image_path": image_path.as_posix(),
            }
    missing = sorted(set(CLASS_IDS) - set(selected))
    if missing:
        raise ValueError(f"Public training annotations did not cover classes: {missing}")
    return selected


def crop_symbol(image, bbox: list[int], target_size: int):
    from PIL import Image

    x1, y1, x2, y2 = bbox
    x_lo, x_hi = sorted((x1, x2))
    y_lo, y_hi = sorted((y1, y2))
    width, height = max(1, x_hi - x_lo), max(1, y_hi - y_lo)
    # Keep a small, fixed context margin for line attachments while excluding
    # unrelated neighboring tags/text from the public source sheet.
    pad = max(8, int(max(width, height) * 0.12))
    left, top = max(0, x_lo - pad), max(0, y_lo - pad)
    right, bottom = min(image.width, x_hi + pad), min(image.height, y_hi + pad)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    crop.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (target_size, target_size), "white")
    tile.paste(crop, ((target_size - crop.width) // 2, (target_size - crop.height) // 2))
    return tile


def label_permutation(label_shift: int) -> dict[int, int]:
    """Return the displayed label for each exemplar class.

    A non-zero cyclic shift preserves the exact exemplar crops, grid geometry,
    typography, and second-image budget of the reference legend while breaking
    the numerical class-to-symbol mapping.  It is therefore a layout-matched
    control rather than a new ontology or a test-answer-bearing cue.
    """

    if not 0 <= label_shift < len(CLASS_IDS):
        raise ValueError(f"label_shift must be in [0, {len(CLASS_IDS) - 1}], got {label_shift}")
    return {
        class_id: ((class_id - 1 + label_shift) % len(CLASS_IDS)) + 1
        for class_id in CLASS_IDS
    }


def render_legend(
    exemplars: dict[int, dict[str, Any]], legend_path: Path, *, label_shift: int = 0
) -> dict[int, int]:
    from PIL import Image, ImageDraw, ImageFont

    cols, rows = 8, 4
    cell_width, cell_height = 256, 264
    label_height, crop_size = 32, 216
    canvas = Image.new("RGB", (cols * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    image_cache: dict[str, Any] = {}
    displayed_labels = label_permutation(label_shift)
    for class_id in CLASS_IDS:
        exemplar = exemplars[class_id]
        image_path = exemplar["image_path"]
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")
        row, col = divmod(class_id - 1, cols)
        x, y = col * cell_width, row * cell_height
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="black", width=1)
        draw.text(
            (x + 8, y + 8),
            f"Class {displayed_labels[class_id]:02d}",
            fill="black",
            font=font,
        )
        tile = crop_symbol(image_cache[image_path], exemplar["bbox_xyxy"], crop_size)
        canvas.paste(tile, (x + (cell_width - crop_size) // 2, y + label_height))
    legend_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(legend_path, format="PNG", optimize=False)
    return displayed_labels


def attach_legend(records: list[dict[str, Any]], legend_path: str) -> list[dict[str, Any]]:
    if any("answer" in row or "cypher" in row for row in records):
        raise ValueError("Input must be answer-isolated public records.")
    output = []
    for row in records:
        result = dict(row)
        result["ontology_control"] = "public_training_symbol_prototype_legend_v1"
        result["ontology_legend_path"] = legend_path
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/raw/Dataset-PID")
    parser.add_argument("--image-root", default="data/raw/pidqa_images")
    parser.add_argument("--split-manifest", default="data/manifests/pidqa_source_split_seed17.jsonl")
    parser.add_argument("--input", default="data/processed/main400_hashblind_set_b_public.jsonl")
    parser.add_argument("--legend", default="data/assets/pidqa_symbol_ontology_v1.png")
    parser.add_argument("--manifest", default="data/manifests/pidqa_symbol_ontology_v1.json")
    parser.add_argument("--output", default="data/processed/main400_set_b_ontology_visible_v1_public.jsonl")
    parser.add_argument(
        "--label-shift",
        type=int,
        default=0,
        help=(
            "Cyclic displayed-label shift for a layout-matched mapping control. "
            "Zero renders the reference legend."
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_root = Path(args.image_root)
    split_manifest = Path(args.split_manifest)
    legend_path = Path(args.legend)
    train = training_sources(split_manifest)
    exemplars = select_exemplars(dataset_root, image_root, train)
    displayed_labels = render_legend(exemplars, legend_path, label_shift=args.label_shift)
    records = read_jsonl(Path(args.input))
    output = attach_legend(records, legend_path.as_posix())
    write_jsonl(Path(args.output), output)

    source_hashes: dict[str, str] = {}
    annotation_hashes: dict[str, str] = {}
    for exemplar in exemplars.values():
        source_hashes.setdefault(exemplar["image_path"], sha256_file(Path(exemplar["image_path"])))
        annotation_hashes.setdefault(exemplar["symbol_annotation_path"], sha256_file(Path(exemplar["symbol_annotation_path"])))
    manifest = {
        "status": "pass",
        "ontology_control": (
            "public_training_symbol_prototype_legend_v1"
            if args.label_shift == 0
            else "public_training_symbol_prototype_legend_label_shift_control_v1"
        ),
        "access_date": "2026-08-10",
        "source_urls": SOURCE_URLS,
        "semantics": "Integer class IDs only; no human-readable symbol names are claimed or supplied.",
        "selection_rule": "First class occurrence by sorted source-number then symbol-array index, restricted to source_split_seed17 train sources.",
        "source_split_manifest": split_manifest.as_posix(),
        "source_split_manifest_sha256": sha256_file(split_manifest),
        "input_public_manifest": Path(args.input).as_posix(),
        "input_record_count": len(records),
        "legend_path": legend_path.as_posix(),
        "legend_sha256": sha256_file(legend_path),
        "legend_layout": {"rows": 4, "columns": 8, "cell_width": 256, "cell_height": 264, "crop_target_size": 216},
        "displayed_label_shift": args.label_shift,
        "exemplar_class_to_displayed_label": {
            str(class_id): displayed_labels[class_id] for class_id in CLASS_IDS
        },
        "class_count": len(exemplars),
        "exemplars": [exemplars[class_id] for class_id in CLASS_IDS],
        "source_image_sha256": source_hashes,
        "symbol_annotation_sha256": annotation_hashes,
        "no_hidden_answers_or_cypher_read": True,
    }
    write_json(Path(args.manifest), manifest)
    print(json.dumps({"status": "pass", "class_count": len(exemplars), "legend": legend_path.as_posix(), "legend_sha256": manifest["legend_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
