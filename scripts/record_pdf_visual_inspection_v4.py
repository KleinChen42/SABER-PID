"""Record completed all-page visual inspection after rendered PNG review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value=hashlib.sha256(); value.update(path.read_bytes()); return value.hexdigest()


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",default="."); parser.add_argument("--render-report",default="reports/generated/pdf_render_validation_v4.json"); parser.add_argument("--output",default="reports/generated/pdf_visual_inspection_v4.json"); parser.add_argument("--validation-version",default="pdf-visual-inspection-v4"); parser.add_argument("--status",choices=("pass","fail"),required=True); parser.add_argument("--note",required=True); args=parser.parse_args()
    root=Path(args.root).resolve(); render= json.loads((root/args.render_report).read_text(encoding="utf-8")); pages=[]
    for document in render["documents"]:
        for relative in document.get("rendered_pngs",[]):
            path=root/relative
            if not path.is_file(): raise FileNotFoundError(path)
            pages.append({"document":document["name"],"path":relative,"sha256":digest(path),"inspected":True})
    if not pages: raise ValueError("No rendered pages available")
    report={"validation_version":args.validation_version,"status":args.status,"method":"all rendered pages inspected by the active automated agent","note":args.note,"page_count":len(pages),"pages":pages,"pdf_sha256":{document["name"]:document.get("sha256") for document in render["documents"]}}
    output=root/args.output; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"status":args.status,"pages":len(pages)},sort_keys=True)); return 0 if args.status=="pass" else 1


if __name__=="__main__": raise SystemExit(main())
