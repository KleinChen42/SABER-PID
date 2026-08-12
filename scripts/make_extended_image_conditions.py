"""Generate the fixed lightweight R4 image-quality conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter

from pidbench.io import read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--condition", choices=("blur", "jpeg35", "center_crop"), required=True)
    parser.add_argument("--image-output-root", required=True)
    parser.add_argument("--records-output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    project_root = Path.cwd().resolve()
    output_root = Path(args.image_output_root)
    transformed_paths: dict[str, Path] = {}
    for record in records:
        sheet = str(record["source_sheet"])
        if sheet in transformed_paths:
            continue
        source = Path(str(record["image_path"]))
        target = output_root / f"{sheet}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as loaded:
            image = loaded.convert("RGB")
            image.thumbnail((1536, 1536))
            if args.condition == "blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
                image.save(target, format="JPEG", quality=95)
            elif args.condition == "jpeg35":
                image.save(target, format="JPEG", quality=35, optimize=True)
            else:
                width, height = image.size
                crop_width = round(width * 0.7)
                crop_height = round(height * 0.7)
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                image = image.crop((left, top, left + crop_width, top + crop_height)).resize(
                    (width, height), Image.Resampling.LANCZOS
                )
                image.save(target, format="JPEG", quality=95)
        transformed_paths[sheet] = target.resolve()

    conditioned: list[dict] = []
    for record in records:
        target = transformed_paths[str(record["source_sheet"])]
        try:
            image_path = target.relative_to(project_root).as_posix()
        except ValueError:
            image_path = target.as_posix()
        conditioned.append({**record, "image_path": image_path, "image_condition": args.condition})
    write_jsonl(args.records_output, conditioned)
    summary = {
        "condition": args.condition,
        "record_count": len(conditioned),
        "source_count": len(transformed_paths),
        "answer_isolated": all("answer" not in row for row in conditioned),
        "image_output_root": str(output_root),
        "max_image_side_before_condition": 1536,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
