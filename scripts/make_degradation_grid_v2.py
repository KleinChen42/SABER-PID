"""Create the frozen Set-B severity grid for F5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageFilter


CONDITIONS = {
    "blur_r1": {"family": "gaussian_blur", "severity": 1.0},
    "blur_r2": {"family": "gaussian_blur", "severity": 2.0},
    "blur_r4": {"family": "gaussian_blur", "severity": 4.0},
    "jpeg_q70": {"family": "jpeg", "severity": 70},
    "jpeg_q35": {"family": "jpeg", "severity": 35},
    "jpeg_q15": {"family": "jpeg", "severity": 15},
    "downsample_s075": {"family": "downsample_restore", "severity": 0.75},
    "downsample_s050": {"family": "downsample_restore", "severity": 0.50},
    "downsample_s025": {"family": "downsample_restore", "severity": 0.25},
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--input", default="data/processed/main400_hashblind_set_b_public.jsonl")
    parser.add_argument("--image-root", default="data/raw/final_degradation/set_b")
    parser.add_argument("--records-root", default="data/processed/final_degradation")
    parser.add_argument("--manifest", default="data/manifests/degradation_grid_v2.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    records = read_jsonl(root / args.input)
    by_source = {str(row["source_sheet"]): row for row in records}
    image_root = root / args.image_root
    records_root = root / args.records_root
    output_meta = []
    for condition, spec in CONDITIONS.items():
        out_dir = image_root / condition
        transformed: dict[str, str] = {}
        for sheet, representative in sorted(by_source.items()):
            source = root / str(representative["image_path"])
            target = out_dir / f"{sheet}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as loaded:
                image = loaded.convert("RGB")
                image.thumbnail((1536, 1536))
                if spec["family"] == "gaussian_blur":
                    image = image.filter(ImageFilter.GaussianBlur(radius=float(spec["severity"])))
                    image.save(target, format="JPEG", quality=95)
                elif spec["family"] == "jpeg":
                    image.save(target, format="JPEG", quality=int(spec["severity"]), optimize=True)
                else:
                    scale = float(spec["severity"])
                    width = max(1, round(image.width * scale))
                    height = max(1, round(image.height * scale))
                    image = image.resize((width, height), Image.Resampling.LANCZOS).resize((image.width, image.height), Image.Resampling.LANCZOS)
                    image.save(target, format="JPEG", quality=95)
            transformed[sheet] = target.relative_to(root).as_posix()
        conditioned = [{**row, "image_path": transformed[str(row["source_sheet"])], "image_condition": condition, "degradation_family": spec["family"], "degradation_severity": spec["severity"]} for row in records]
        record_path = records_root / f"main400_hashblind_set_b_{condition}_public.jsonl"
        write_jsonl(record_path, conditioned)
        output_meta.append({"condition": condition, **spec, "record_count": len(conditioned), "source_count": len(transformed), "records_path": record_path.relative_to(root).as_posix(), "image_root": out_dir.relative_to(root).as_posix()})
    manifest = {"status": "pass", "set_id": "hashblind_set_B", "input": args.input, "source_count": len(by_source), "record_count": len(records), "max_side_before_transform": 1536, "conditions": output_meta, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
