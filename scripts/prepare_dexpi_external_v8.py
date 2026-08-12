"""Freeze a public DEXPI external tag-reading benchmark for the v8 extension.

The source repository contains vendor-rendered P&ID images together with
same-stem DEXPI/Proteus XML exports.  This script accepts only exact
image--XML pairs, derives candidate tags from structured XML attributes, and
retains a tag only when the XML graphics also declare it as visible text.
Selection never uses a model prediction or a hidden benchmark answer.

The resulting public manifests contain questions and image paths but no
answers.  References are written separately for scorer-only use.  A
one-to-one source derangement supplies the shuffled-image control, and every
shuffled image comes from a different logical DEXPI test case.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


IMAGE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
TAG_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,8})\s*(?:[- ]\s*)?"
    r"(\d+(?:\s*[.\-/]\s*\d+)*)([A-Za-z]?)(?![A-Za-z0-9])"
)
FULL_TAG_RE = re.compile(
    r"^\s*([A-Za-z]{1,8})\s*(?:[- ]\s*)?"
    r"(\d+(?:\s*[.\-/]\s*\d+)*)([A-Za-z]?)\s*$"
)
LOGICAL_GROUP_RE = re.compile(r"([CEIP]\d{2})V\d{2}", re.IGNORECASE)
GENERIC_PREFIX = "tagnameprefixassignmentclass"
GENERIC_SEQUENCE = "tagnamesequencenumberassignmentclass"
GENERIC_SUFFIX = "tagnamesuffixassignmentclass"
EXCLUDED_PREFIXES = {"MM", "REV"}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_tag(value: str) -> str | None:
    """Return a separator-stable lowercase tag or ``None``.

    Numeric hierarchies use hyphens in the frozen representation, so printed
    forms such as ``PI 4712.01`` and ``PI-4712-01`` compare identically.
    """

    text = html.unescape(str(value)).replace("\r", " ").replace("\n", " ")
    match = FULL_TAG_RE.fullmatch(" ".join(text.split()))
    if not match:
        return None
    prefix, numeric, suffix = match.groups()
    if prefix.upper() in EXCLUDED_PREFIXES:
        return None
    numeric_parts = re.split(r"\s*[.\-/]\s*", numeric)
    return prefix.lower() + "-".join(numeric_parts) + suffix.lower()


def tag_prefix(tag: str) -> str:
    match = re.match(r"[a-z]+", tag)
    if not match:
        raise ValueError(f"Tag has no alphabetic prefix: {tag!r}")
    return match.group(0)


def tags_in_text(value: str) -> set[str]:
    result: set[str] = set()
    text = html.unescape(str(value)).replace("\r", " ").replace("\n", " ")
    for match in TAG_RE.finditer(text):
        tag = normalize_tag("".join(match.groups()))
        if tag:
            result.add(tag)
    return result


def direct_generic_attributes(element: ET.Element) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = defaultdict(list)
    for child in element:
        if local_name(child.tag) != "GenericAttributes":
            continue
        for item in child:
            if local_name(item.tag) != "GenericAttribute":
                continue
            name = str(item.attrib.get("Name", "")).strip().casefold()
            value = str(item.attrib.get("Value", "")).strip()
            if name and value:
                attributes[name].append(value)
    return dict(attributes)


def structured_tag_candidates(root: ET.Element) -> set[str]:
    """Extract tag candidates from semantic XML fields, not raster OCR."""

    candidates: set[str] = set()
    for element in root.iter():
        attrs = direct_generic_attributes(element)
        prefixes = attrs.get(GENERIC_PREFIX, [])
        sequences = attrs.get(GENERIC_SEQUENCE, [])
        suffixes = attrs.get(GENERIC_SUFFIX, [""])
        for prefix in prefixes:
            for sequence in sequences:
                for suffix in suffixes or [""]:
                    tag = normalize_tag(f"{prefix}{sequence}{suffix}")
                    if tag:
                        candidates.add(tag)

        # Several legacy vendor exports use these two fields for instrument
        # loops rather than the newer DEXPI prefix/sequence attributes.
        complete = attrs.get("complete function", [])
        tag_names = attrs.get("tagname", [])
        for value in tag_names:
            tag = normalize_tag(value)
            if tag:
                candidates.add(tag)
        for prefix in complete:
            if not re.fullmatch(r"[A-Za-z]{1,8}", prefix.strip()):
                continue
            for number in tag_names:
                if re.fullmatch(r"\d+(?:[.\-/]\d+)*[A-Za-z]?", number.strip()):
                    tag = normalize_tag(f"{prefix}{number}")
                    if tag:
                        candidates.add(tag)
    return candidates


def graphic_texts(root: ET.Element) -> list[str]:
    values = []
    for element in root.iter():
        if local_name(element.tag) != "Text":
            continue
        value = str(element.attrib.get("String", element.text or "")).strip()
        if value:
            values.append(html.unescape(value))
    return values


def visible_structured_tags(root: ET.Element) -> tuple[set[str], dict[str, Any]]:
    candidates = structured_tag_candidates(root)
    texts = graphic_texts(root)
    direct = set().union(*(tags_in_text(value) for value in texts)) if texts else set()

    # Split instrument labels are common (for example, ``P`` and ``001`` in
    # adjacent Text elements).  The normalized XML graphics stream provides a
    # second deterministic confirmation for semantic candidates.  A candidate
    # is still impossible to create without a structured tag field above.
    graphics_blob = re.sub(r"[^A-Za-z0-9]", "", " ".join(texts)).casefold()
    visible = {
        tag
        for tag in candidates
        if tag in direct
        or re.sub(r"[^A-Za-z0-9]", "", tag).casefold() in graphics_blob
    }
    return visible, {
        "structured_candidate_count": len(candidates),
        "xml_graphic_text_count": len(texts),
        "direct_graphic_tag_count": len(direct),
        "visible_structured_tag_count": len(visible),
        "confirmation": "structured XML tag field intersected with XML graphic Text declarations",
    }


def logical_group(path: Path) -> str:
    match = LOGICAL_GROUP_RE.search(path.name)
    return match.group(1).upper() if match else path.stem.casefold()


def discover_exact_pairs(source_root: Path) -> list[tuple[Path, Path]]:
    by_directory: dict[Path, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    for path in files:
        by_directory[path.parent][path.stem.casefold()].append(path)
    pairs: list[tuple[Path, Path]] = []
    for image in files:
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "doc" in image.stem.casefold() or "specification" in image.stem.casefold():
            continue
        xmls = [
            path
            for path in by_directory[image.parent][image.stem.casefold()]
            if path.suffix.lower() == ".xml"
        ]
        if len(xmls) == 1:
            pairs.append((image, xmls[0]))
    return sorted(pairs, key=lambda pair: pair[0].as_posix().casefold())


def extract_pdf_text(path: Path) -> str:
    process = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "1", "-layout", str(path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.decode("utf-8", errors="replace")


def render_image(source: Path, destination: Path) -> tuple[int, int]:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".pdf":
        temporary_stem = destination.with_suffix("")
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                "1",
                "-singlefile",
                "-png",
                "-scale-to",
                "4096",
                str(source),
                str(temporary_stem),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        generated = temporary_stem.with_suffix(".png")
        if generated != destination:
            generated.replace(destination)
    else:
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="PNG", optimize=True)
    with Image.open(destination) as image:
        return int(image.width), int(image.height)


def choose_sources(candidates: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    """Maximize logical-test-case coverage before adding vendor variants."""

    format_rank = {".pdf": 0, ".png": 1, ".jpg": 2, ".jpeg": 3}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_group[str(row["logical_group"])].append(row)
    for rows in by_group.values():
        rows.sort(
            key=lambda row: (
                format_rank.get(str(row["image_suffix"]), 99),
                str(row["selection_hash"]),
            )
        )
    selected = [by_group[group][0] for group in sorted(by_group)]
    remaining = [
        row
        for group in sorted(by_group)
        for row in by_group[group][1:]
    ]
    remaining.sort(key=lambda row: str(row["selection_hash"]))
    selected.extend(remaining[: max(0, max_images - len(selected))])
    return sorted(selected[:max_images], key=lambda row: str(row["selection_hash"]))


def prefix_map(tags: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for tag in sorted(set(tags)):
        grouped[tag_prefix(tag)].append(tag)
    return {
        prefix: values
        for prefix, values in sorted(grouped.items())
        if 1 <= len(values) <= 12 and prefix.upper() not in EXCLUDED_PREFIXES
    }


def choose_prefixes(source_id: str, grouped: dict[str, list[str]], limit: int) -> list[str]:
    return sorted(
        grouped,
        key=lambda prefix: hashlib.sha256(f"{source_id}|{prefix}".encode()).hexdigest(),
    )[:limit]


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def source_derangement(rows: list[dict[str, Any]], seed: int = 8122026) -> dict[str, str]:
    """Find a reproducible one-to-one cross-test-case low-overlap mapping."""

    if len(rows) < 2:
        raise ValueError("At least two sources are required for a derangement")
    originals = list(rows)
    rng = random.Random(seed)
    best: tuple[float, list[dict[str, Any]]] | None = None
    for _ in range(50_000):
        candidate = list(rows)
        rng.shuffle(candidate)
        if any(
            left["source_id"] == right["source_id"]
            or left["logical_group"] == right["logical_group"]
            for left, right in zip(originals, candidate)
        ):
            continue
        score = sum(
            jaccard(set(left["tags"]), set(right["tags"]))
            for left, right in zip(originals, candidate)
        )
        if best is None or score < best[0]:
            best = (score, candidate)
            if score == 0:
                break
    if best is None:
        raise ValueError("No cross-logical-group source derangement was found")
    return {
        str(left["source_id"]): str(right["source_id"])
        for left, right in zip(originals, best[1])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--min-images", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-prefixes-per-image", type=int, default=3)
    parser.add_argument(
        "--public-dir", default="data/processed/rineng_v8_dexpi_external"
    )
    parser.add_argument(
        "--hidden", default="data/answer_store/rineng_v8_dexpi_external_hidden.jsonl"
    )
    parser.add_argument(
        "--plan", default="data/manifests/rineng_v8_dexpi_external_plan.json"
    )
    parser.add_argument(
        "--report", default="reports/generated/rineng_v8_dexpi_external_audit.json"
    )
    args = parser.parse_args()
    if args.min_images < 2 or args.max_images < args.min_images:
        raise ValueError("Invalid image-count bounds")
    if args.max_prefixes_per_image <= 0:
        raise ValueError("max-prefixes-per-image must be positive")

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    asset_root = Path(args.asset_root).resolve()
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError(f"DEXPI source is not a git checkout: {source_root}")
    license_path = source_root / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(license_path)
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    audited: list[dict[str, Any]] = []
    image_hashes: set[str] = set()
    failures: list[dict[str, str]] = []
    for image_path, xml_path in discover_exact_pairs(source_root):
        try:
            root = ET.parse(xml_path).getroot()
            visible, diagnostics = visible_structured_tags(root)
            grouped = prefix_map(visible)
            retained_tags = sorted({tag for values in grouped.values() for tag in values})
            if not retained_tags:
                continue
            image_digest = sha256(image_path)
            if image_digest in image_hashes:
                continue
            image_hashes.add(image_digest)
            pdf_text_tags: set[str] = set()
            if image_path.suffix.lower() == ".pdf":
                pdf_text_tags = tags_in_text(extract_pdf_text(image_path))
            relative_image = image_path.relative_to(source_root).as_posix()
            relative_xml = xml_path.relative_to(source_root).as_posix()
            selection_hash = hashlib.sha256(
                f"{relative_image}|{image_digest}|{sha256(xml_path)}".encode()
            ).hexdigest()
            audited.append(
                {
                    "source_image": relative_image,
                    "source_xml": relative_xml,
                    "source_image_sha256": image_digest,
                    "source_xml_sha256": sha256(xml_path),
                    "image_suffix": image_path.suffix.lower(),
                    "logical_group": logical_group(image_path),
                    "selection_hash": selection_hash,
                    "tags": retained_tags,
                    "prefixes": grouped,
                    "pdf_embedded_text_confirmed_tag_count": len(
                        set(retained_tags) & pdf_text_tags
                    ),
                    **diagnostics,
                }
            )
        except Exception as exc:  # retain a complete machine-readable audit
            failures.append(
                {
                    "image": image_path.relative_to(source_root).as_posix(),
                    "xml": xml_path.relative_to(source_root).as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    selected = choose_sources(audited, args.max_images)
    if len(selected) < args.min_images:
        raise RuntimeError(
            f"Only {len(selected)} exact image/XML pairs passed the visibility audit; "
            f"at least {args.min_images} are required"
        )

    asset_root.mkdir(parents=True, exist_ok=True)
    selected_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(selected, start=1):
        source_id = f"dexpi-v8-{index:03d}-{str(row['source_image_sha256'])[:10]}"
        asset_path = asset_root / f"{source_id}.png"
        size = render_image(source_root / str(row["source_image"]), asset_path)
        row.update(
            {
                "source_id": source_id,
                "asset_path": asset_path.as_posix(),
                "asset_sha256": sha256(asset_path),
                "asset_size": list(size),
            }
        )
        selected_by_id[source_id] = row

    mapping = source_derangement(selected)
    public_correct: list[dict[str, Any]] = []
    public_shuffled: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for source_number, row in enumerate(selected, start=1):
        source_id = str(row["source_id"])
        prefixes = choose_prefixes(
            source_id, dict(row["prefixes"]), args.max_prefixes_per_image
        )
        shuffled_source = selected_by_id[mapping[source_id]]
        for question_number, prefix in enumerate(prefixes, start=1):
            instance_id = f"dexpi-v8-{source_number:03d}-{question_number:02d}"
            question = (
                "Which equipment or instrumentation tags beginning with "
                f"{prefix.upper()} are visible in this P&ID? "
                "Return only a comma-separated tag list; return [] if none are visible."
            )
            base = {
                "instance_id": instance_id,
                "source_id": source_id,
                "source_sheet": str(row["logical_group"]),
                "task": "value",
                "question": question,
                "fields": {"Prefix": prefix.upper()},
            }
            correct = {
                **base,
                "image_path": str(row["asset_path"]),
                "image_source_id": source_id,
            }
            shuffled = {
                **base,
                "image_path": str(shuffled_source["asset_path"]),
                "image_source_id": str(shuffled_source["source_id"]),
            }
            reference = {
                **base,
                "answer": list(row["prefixes"][prefix]),
                "reference_provenance": {
                    "source_xml": row["source_xml"],
                    "source_xml_sha256": row["source_xml_sha256"],
                    "visibility_rule": row["confirmation"],
                },
            }
            public_correct.append(correct)
            public_shuffled.append(shuffled)
            hidden.append(reference)

    public_dir = output_root / args.public_dir
    correct_path = public_dir / "dexpi_external_v8_correct_public.jsonl"
    shuffled_path = public_dir / "dexpi_external_v8_shuffled_public.jsonl"
    hidden_path = output_root / args.hidden
    write_jsonl(correct_path, public_correct)
    write_jsonl(shuffled_path, public_shuffled)
    write_jsonl(hidden_path, hidden)

    if any("answer" in key.casefold() for row in public_correct for key in row):
        raise AssertionError("Correct public manifest contains an answer-bearing field")
    if any("answer" in key.casefold() for row in public_shuffled for key in row):
        raise AssertionError("Shuffled public manifest contains an answer-bearing field")
    if [row["instance_id"] for row in public_correct] != [
        row["instance_id"] for row in public_shuffled
    ]:
        raise AssertionError("Correct/shuffled membership mismatch")

    plan_path = output_root / args.plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "rineng-v8-dexpi-external",
        "date": "2026-08-12",
        "status": "frozen_before_inference",
        "external_source": {
            "name": "DEXPI Public Example PIDs",
            "url": "https://gitlab.com/dexpi/TrainingTestCases",
            "commit": commit,
            "license": "CC BY 4.0",
            "license_path": "LICENSE",
            "license_sha256": sha256(license_path),
        },
        "selection": {
            "rule": (
                "exact same-directory image/XML stem; structured-tag/graphic-text "
                "intersection; unique image SHA-256; one per logical test case first, "
                "then SHA-ranked vendor variants"
            ),
            "candidate_exact_pair_count": len(discover_exact_pairs(source_root)),
            "visibility_qualified_unique_pair_count": len(audited),
            "selected_image_count": len(selected),
            "logical_test_case_count": len({row["logical_group"] for row in selected}),
            "question_count": len(public_correct),
            "minimum_requested_images": args.min_images,
            "maximum_requested_images": args.max_images,
            "model_output_used_for_selection": False,
        },
        "datasets": [
            {
                "dataset_id": "dexpi_external_v8",
                "correct_input": correct_path.relative_to(output_root).as_posix(),
                "correct_sha256": sha256(correct_path),
                "shuffled_input": shuffled_path.relative_to(output_root).as_posix(),
                "shuffled_sha256": sha256(shuffled_path),
                "record_count": len(public_correct),
                "source_count": len(selected),
                "logical_group_count": len({row["logical_group"] for row in selected}),
            }
        ],
        "conditions": ["correct", "shuffled", "text_only", "paddleocr_full_image"],
        "frozen_inference": {
            "qwen_model_label": "qwen3vl8b",
            "qwen_model_path": "models/Qwen3-VL-8B-Instruct-modelscope",
            "prompt_id": "p0",
            "max_image_side": 3072,
            "max_new_tokens": 512,
            "do_sample": False,
            "ocr": "PaddleOCR 2.8.1 / PaddlePaddle 2.6.2 English full-image, no crop",
        },
        "answer_isolation": {
            "public_manifests_answer_bearing_field_count": 0,
            "hidden_reference": hidden_path.relative_to(output_root).as_posix(),
            "hidden_reference_sha256": sha256(hidden_path),
            "scorer_reference_read_by_inference": False,
            "status": "pass",
        },
        "shuffled_control": {
            "mapping": mapping,
            "one_to_one": len(set(mapping.values())) == len(mapping),
            "fixed_point_count": sum(key == value for key, value in mapping.items()),
            "same_logical_test_case_count": sum(
                selected_by_id[key]["logical_group"]
                == selected_by_id[value]["logical_group"]
                for key, value in mapping.items()
            ),
            "rule": "50,000-seed-search one-to-one cross-test-case minimum tag-Jaccard derangement",
        },
        "analysis_boundary": (
            "DEXPI examples are public engineering exchange test cases, not an independent "
            "real-plant population. Vendor variants share logical test cases and must be "
            "grouped by source_sheet for uncertainty estimates."
        ),
    }
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "version": "rineng-v8-dexpi-external-audit",
        "status": "pass",
        "plan": plan_path.relative_to(output_root).as_posix(),
        "plan_sha256": sha256(plan_path),
        "source_root": str(source_root),
        "asset_root": str(asset_root),
        "failed_pair_count": len(failures),
        "failed_pairs": failures,
        "selected_sources": selected,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "selected_images": len(selected),
                "logical_groups": plan["selection"]["logical_test_case_count"],
                "questions": len(public_correct),
                "plan": str(plan_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
