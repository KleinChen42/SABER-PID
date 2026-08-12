"""Render every v4 PDF page and validate compile-log layout diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import fitz


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/generated/pdf_render_validation_v4.json")
    parser.add_argument("--render-dir", default="tmp/pdfs/final_validation_v4")
    parser.add_argument("--pdf-dir", default="output/pdf")
    parser.add_argument("--validation-version", default="pdf-render-v4")
    parser.add_argument("--visual-record", default="reports/generated/pdf_visual_inspection_v4.json")
    args = parser.parse_args()
    root = Path(args.root).resolve(); render_root = root / args.render_dir
    pdf_dir = Path(args.pdf_dir).as_posix().rstrip("/")
    pdfs = tuple(
        (name, f"{pdf_dir}/{name}.pdf", f"{pdf_dir}/{name}.log")
        for name in ("manuscript", "supplementary")
    )
    documents: list[dict[str, Any]] = []
    for name, pdf_relative, log_relative in pdfs:
        pdf_path = root / pdf_relative; log_path = root / log_relative
        entry: dict[str, Any] = {"name": name, "pdf": pdf_relative, "log": log_relative, "exists": pdf_path.is_file() and log_path.is_file()}
        if entry["exists"]:
            document = fitz.open(pdf_path); output_dir = render_root / name; output_dir.mkdir(parents=True, exist_ok=True)
            rendered = []
            for number, page in enumerate(document, start=1):
                path = output_dir / f"page-{number:02d}.png"; page.get_pixmap(matrix=fitz.Matrix(1.8,1.8), alpha=False).save(path); rendered.append(path.relative_to(root).as_posix())
            log = log_path.read_text(encoding="utf-8", errors="replace")
            entry.update({
                "sha256": sha256(pdf_path), "size_bytes": pdf_path.stat().st_size, "page_count": len(document),
                "page_size_points": [round(document[0].rect.width,2), round(document[0].rect.height,2)],
                "encrypted": document.is_encrypted, "metadata_title": document.metadata.get("title", ""), "rendered_pngs": rendered,
                "overfull_hbox_count": log.count("Overfull \\hbox"), "underfull_hbox_count": log.count("Underfull \\hbox"),
                "undefined_reference_count": log.lower().count("undefined reference"), "undefined_citation_count": log.lower().count("undefined citation"),
            })
        documents.append(entry)
    failures = [entry["name"] for entry in documents if not entry["exists"] or not entry.get("page_count") or not entry.get("metadata_title") or entry.get("encrypted") or entry.get("overfull_hbox_count",0) or entry.get("underfull_hbox_count",0) or entry.get("undefined_reference_count",0) or entry.get("undefined_citation_count",0)]
    report = {"validation_version": args.validation_version, "render_engine": "PyMuPDF", "visual_review": {"status": "separate inspection record required", "record": args.visual_record}, "documents": documents, "status": "pass" if not failures else "fail", "failures": failures}
    output = root / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True)); return 0 if report["status"] == "pass" else 1


if __name__ == "__main__": raise SystemExit(main())
