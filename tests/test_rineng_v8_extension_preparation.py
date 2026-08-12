from pathlib import Path

import numpy as np
from PIL import Image

from prepare_quality_robustness_v8 import transform_image
from run_internvl_budget_matched_v8 import letterbox_grid_tensor


def runtime_dir() -> Path:
    path = Path("outputs/test_runtime_v8_unit")
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_quality_transforms_preserve_dimensions() -> None:
    directory = runtime_dir()
    source = directory / "quality_transform_fixture_v8_source.png"
    pixels = np.full((40, 80, 3), 255, dtype=np.uint8)
    pixels[10:30, 20:60] = 0
    Image.fromarray(pixels).save(source)
    specs = {
        "jpeg": {"family": "jpeg", "severity": 70},
        "blur": {"family": "gaussian_blur", "severity": 1.0},
        "down": {"family": "downsample_restore", "severity": 0.75},
    }
    for name, spec in specs.items():
        suffix = ".jpg" if name == "jpeg" else ".png"
        target = directory / f"quality_transform_fixture_v8_{name}{suffix}"
        metadata = transform_image(source, target, spec)
        assert metadata["original_size"] == [80, 40]
        assert metadata["output_size"] == [80, 40]
        assert len(metadata["output_sha256"]) == 64


def test_budget_matched_letterbox_has_exact_tensor_budget() -> None:
    source = runtime_dir() / "letterbox_fixture_v8_wide.png"
    Image.new("RGB", (160, 90), (240, 240, 240)).save(source)
    pixels, metadata = letterbox_grid_tensor(
        source, columns=9, rows=6, tile_side=448, add_thumbnail=False
    )
    assert tuple(pixels.shape) == (54, 3, 448, 448)
    assert pixels.numel() == 32_514_048
    assert metadata["tile_count"] == 54
    assert metadata["thumbnail_added"] is False
    assert metadata["letterbox_canvas_size"] == [4032, 2688]
