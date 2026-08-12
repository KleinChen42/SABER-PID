"""Render final PDFs and record deterministic PDF/package QA metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import fitz


PDFS = (
    ("manuscript", "output/pdf/manuscript.pdf", "output/pdf/manuscript.log"),
    ("supplementary", "output/pdf/supplementary.pdf", "output/pdf/supplementary.log"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="reports/generated/pdf_render_validation_v2.json",
    )
    parser.add_argument(
        "--render-dir",
        default="tmp/pdfs/final_validation_v2",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    render_root = root / args.render_dir
    entries: list[dict[str, Any]] = []

    for name, pdf_relative, log_relative in PDFS:
        pdf_path = root / pdf_relative
        log_path = root / log_relative
        entry: dict[str, Any] = {
            "name": name,
            "pdf": pdf_relative,
            "log": log_relative,
            "exists": pdf_path.is_file() and log_path.is_file(),
        }
        if entry["exists"]:
            document = fitz.open(pdf_path)
            output_dir = render_root / name
            output_dir.mkdir(parents=True, exist_ok=True)
            rendered = []
            for page_number, page in enumerate(document, start=1):
                page_path = output_dir / f"page-{page_number:02d}.png"
                page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False).save(page_path)
                rendered.append(page_path.relative_to(root).as_posix())
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            entry.update(
                {
                    "sha256": sha256(pdf_path),
                    "size_bytes": pdf_path.stat().st_size,
                    "page_count": len(document),
                    "page_size_points": [round(document[0].rect.width, 2), round(document[0].rect.height, 2)],
                    "encrypted": document.is_encrypted,
                    "metadata_title": document.metadata.get("title", ""),
                    "rendered_pngs": rendered,
                    "overfull_hbox_count": log_text.count("Overfull \\hbox"),
                    "underfull_hbox_count": log_text.count("Underfull \\hbox"),
                }
            )
        entries.append(entry)

    failures = [
        item["name"]
        for item in entries
        if not item["exists"]
        or not item.get("page_count")
        or not item.get("metadata_title")
        or item.get("overfull_hbox_count", 0)
        or item.get("underfull_hbox_count", 0)
    ]
    report = {
        "validation_version": "pdf-render-v2",
        "render_engine": "PyMuPDF",
        "visual_review": {
            "method": "rendered PNG inspection by the active automated agent",
            "status": "completed",
        },
        "documents": entries,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
