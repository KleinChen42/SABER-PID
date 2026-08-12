"""Corrected F5 severity-grid generator (downsample restores original size)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageFilter


CONDITIONS = {
    "blur_r1": ("gaussian_blur", 1.0), "blur_r2": ("gaussian_blur", 2.0), "blur_r4": ("gaussian_blur", 4.0),
    "jpeg_q70": ("jpeg", 70), "jpeg_q35": ("jpeg", 35), "jpeg_q15": ("jpeg", 15),
    "downsample_s075": ("downsample_restore", 0.75), "downsample_s050": ("downsample_restore", 0.50), "downsample_s025": ("downsample_restore", 0.25),
}


def read(path: Path):
    with path.open("r", encoding="utf-8-sig") as h: return [json.loads(line) for line in h if line.strip()]


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows: h.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--input", default="data/processed/main400_hashblind_set_b_public.jsonl"); p.add_argument("--image-root", default="data/raw/final_degradation/set_b"); p.add_argument("--records-root", default="data/processed/final_degradation"); p.add_argument("--manifest", default="data/manifests/degradation_grid_v2.json"); a = p.parse_args()
    root = Path(a.root).resolve(); records = read(root / a.input); reps = {str(row["source_sheet"]): row for row in records}; image_root = root / a.image_root; records_root = root / a.records_root; meta = []
    for condition, (family, severity) in CONDITIONS.items():
        out = image_root / condition; paths = {}
        for sheet, row in sorted(reps.items()):
            source = root / str(row["image_path"]); target = out / f"{sheet}.jpg"; target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as loaded:
                image = loaded.convert("RGB"); image.thumbnail((1536, 1536)); original_size = image.size
                if family == "gaussian_blur": image.filter(ImageFilter.GaussianBlur(radius=float(severity))).save(target, format="JPEG", quality=95)
                elif family == "jpeg": image.save(target, format="JPEG", quality=int(severity), optimize=True)
                else:
                    small = image.resize((max(1, round(original_size[0] * float(severity))), max(1, round(original_size[1] * float(severity)))), Image.Resampling.LANCZOS); small.resize(original_size, Image.Resampling.LANCZOS).save(target, format="JPEG", quality=95)
            paths[sheet] = target.relative_to(root).as_posix()
        conditioned = [{**row, "image_path": paths[str(row["source_sheet"])], "image_condition": condition, "degradation_family": family, "degradation_severity": severity} for row in records]
        record_path = records_root / f"main400_hashblind_set_b_{condition}_public.jsonl"; write(record_path, conditioned); meta.append({"condition": condition, "family": family, "severity": severity, "records_path": record_path.relative_to(root).as_posix(), "image_root": out.relative_to(root).as_posix(), "record_count": len(conditioned), "source_count": len(paths)})
    manifest = {"status": "pass", "set_id": "hashblind_set_B", "input": a.input, "source_count": len(reps), "record_count": len(records), "max_side_before_transform": 1536, "conditions": meta, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; path = root / a.manifest; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
