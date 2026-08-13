"""Build review-only contact sheets from the final V10 rendered PDF pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build(source: Path, output: Path, *, columns: int = 4, thumb_width: int = 360) -> int:
    pages = sorted(source.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No rendered pages in {source}")
    opened = [Image.open(path).convert("RGB") for path in pages]
    ratio = opened[0].height / opened[0].width
    thumb_height = round(thumb_width * ratio)
    label_height = 28
    margin = 18
    rows = (len(opened) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            margin + columns * (thumb_width + margin),
            margin + rows * (thumb_height + label_height + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, page) in enumerate(zip(pages, opened)):
        row, column = divmod(index, columns)
        x = margin + column * (thumb_width + margin)
        y = margin + row * (thumb_height + label_height + margin)
        thumb = page.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb_width, y + thumb_height), outline="#9AA0A6", width=1)
        draw.text((x, y + thumb_height + 6), path.stem, fill="#333333", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(150, 150))
    return len(pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", default="tmp/pdfs/final_validation_v10")
    parser.add_argument("--output-root", default="tmp/pdfs/contact_sheets_v10")
    args = parser.parse_args()
    render_root = Path(args.render_root)
    output_root = Path(args.output_root)
    for document in ("manuscript", "supplementary"):
        count = build(render_root / document, output_root / f"{document}_contact.png")
        print(f"{document}: {count} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
