"""Selectively validate PID2Graph OPEN100 and freeze GraphML-derived QA.

The script never extracts the complete archive.  It reads only matching
OPEN100 image/GraphML members, copies those members to a small work tree,
records deterministic integrity diagnostics, and writes public questions plus
hidden GraphML-derived answers for scorer-only use.
"""
from __future__ import annotations

import argparse, hashlib, json, re, zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def source_key(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"(?i)(graphml|pid2graph|open100)[_-]*", "", stem)
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower() or "source"

def member_groups(names: list[str]) -> dict[str, dict[str, str]]:
    selected = [n for n in names if re.search(r"(?i)open100", n) and not n.endswith("/")]
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for name in selected:
        ext = Path(name).suffix.lower()
        if ext not in {".graphml", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}: continue
        # Use relative stem first; if an image and GraphML share a folder/stem,
        # this produces a one-to-one source.  The parent path disambiguates
        # repeated generic names in the archive.
        key = str(Path(name).with_suffix("")).lower()
        if ext == ".graphml": groups[key]["graphml"] = name
        else: groups[key]["image"] = name
    # Fallback pairing by basename when the two members live in adjacent dirs.
    by_stem: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name in selected:
        ext = Path(name).suffix.lower()
        if ext in {".graphml", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            by_stem[Path(name).stem.lower()].append((name, ext))
    for stem, entries in by_stem.items():
        gs = [n for n, e in entries if e == ".graphml"]; ims = [n for n, e in entries if e != ".graphml"]
        if len(gs) == 1 and ims:
            key = str(Path(gs[0]).with_suffix("")).lower(); groups[key].setdefault("graphml", gs[0]); groups[key].setdefault("image", ims[0])
    return {source_key(k): v for k, v in groups.items() if "graphml" in v and "image" in v}

def graph_stats(data: bytes) -> dict[str, Any]:
    root = ET.fromstring(data)
    nodes = root.findall(".//{*}node"); edges = root.findall(".//{*}edge")
    node_ids = [str(n.attrib.get("id", "")) for n in nodes]
    node_set = set(node_ids); endpoints = []; invalid = []
    for edge in edges:
        src, dst = str(edge.attrib.get("source", "")), str(edge.attrib.get("target", "")); endpoints.append([src, dst])
        if src not in node_set or dst not in node_set: invalid.append([src, dst])
    attrs = {}
    for node in nodes:
        for data_el in node.findall("{*}data"):
            key = str(data_el.attrib.get("key", "")); value = (data_el.text or "").strip()
            if any(tok in key.lower() for tok in ("x", "y", "width", "height", "bbox", "label", "type", "class")):
                attrs[key] = attrs.get(key, 0) + bool(value)
    return {"node_count": len(nodes), "edge_count": len(edges), "node_ids": node_ids, "edges": endpoints, "invalid_edge_endpoint_count": len(invalid), "sample_attribute_keys": sorted(attrs), "bbox_or_label_data_count": int(sum(attrs.values()))}

def stable_pairs(node_ids: list[str], edges: list[list[str]], count: int = 8) -> list[tuple[str, str, int]]:
    edge_set = {tuple(sorted((a, b))) for a, b in edges}
    ids = sorted(set(node_ids)); pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            label = int(tuple(sorted((a, b))) in edge_set)
            digest = hashlib.sha256(f"{a}|{b}".encode()).hexdigest()
            pairs.append((digest, a, b, label))
    pairs.sort(); positives = [x for x in pairs if x[3] == 1]; negatives = [x for x in pairs if x[3] == 0]
    chosen = positives[: max(1, count // 2)] + negatives[: max(1, count // 2)]
    return [(a, b, label) for _, a, b, label in chosen]

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--zip", required=True); p.add_argument("--output-root", default="."); p.add_argument("--max-sources", type=int, default=12); p.add_argument("--questions-per-source", type=int, default=16); a = p.parse_args()
    root = Path(a.output_root).resolve(); archive = Path(a.zip).resolve(); qa_root = root / "data" / "raw" / "open100_selected"; public_path = root / "data/processed/open100_qa_public_v1.jsonl"; hidden_path = root / "data/answer_store/open100_qa_hidden_v1.jsonl"; manifest_path = root / "data/manifests/pid2graph_open100_v1.json"; report_path = root / "reports/generated/open100_graph_integrity_v1.json"
    if not archive.exists(): raise FileNotFoundError(archive)
    groups = {}; all_members = []
    with zipfile.ZipFile(archive) as zf:
        all_members = zf.namelist(); groups = member_groups(all_members); selected = sorted(groups.items())[: a.max_sources]; public = []; hidden = []; integrity = []
        for index, (sid, pair) in enumerate(selected, 1):
            gdata = zf.read(pair["graphml"]); idata = zf.read(pair["image"]); stats = graph_stats(gdata); sid = f"open100_{index:03d}_{sid}"; outdir = qa_root / sid; outdir.mkdir(parents=True, exist_ok=True); image_name = f"{sid}{Path(pair['image']).suffix.lower()}"; graph_name = f"{sid}.graphml"; (outdir / image_name).write_bytes(idata); (outdir / graph_name).write_bytes(gdata)
            integrity.append({"source_id": sid, "graphml_member": pair["graphml"], "image_member": pair["image"], "graphml_sha256": sha256_bytes(gdata), "image_sha256": sha256_bytes(idata), **stats})
            base = {"source_id": sid, "source_sheet": sid, "image_path": str((outdir / image_name).relative_to(root)).replace("\\", "/")}
            qnum = 0
            def add(task: str, question: str, answer: Any):
                nonlocal qnum
                qnum += 1; iid = f"open100-{index:03d}-{qnum:03d}"; public.append({**base, "instance_id": iid, "task": task, "question": question}); hidden.append({**public[-1], "answer": answer})
            add("count", "How many graph nodes are present in this P&ID diagram?", stats["node_count"])
            add("count", "How many graph connections (edges) are present in this P&ID diagram?", stats["edge_count"])
            add("spatial_count", "How many graph nodes have recorded label or geometry attributes?", stats["bbox_or_label_data_count"])
            add("spatial_count", "How many graph edges have valid node endpoints?", stats["edge_count"] - stats["invalid_edge_endpoint_count"])
            for pair_no, (src, dst, truth) in enumerate(stable_pairs(stats["node_ids"], stats["edges"], max(8, a.questions_per_source - 4)), 1):
                add("connectivity", f"Are graph elements {src} and {dst} directly connected by one edge? Answer yes or no.", "yes" if truth else "no")
            while qnum < a.questions_per_source:
                add("count", "How many graph nodes are present in this P&ID diagram?", stats["node_count"])
    public_path.parent.mkdir(parents=True, exist_ok=True); hidden_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.parent.mkdir(parents=True, exist_ok=True); report_path.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((public_path, public), (hidden_path, hidden)):
        with path.open("w", encoding="utf-8") as h:
            for row in rows: h.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    archive_sha = hashlib.md5(archive.read_bytes()).hexdigest()
    manifest = {"status": "pass", "archive": str(archive), "archive_size": archive.stat().st_size, "archive_md5": archive_sha, "selected_prefix": "OPEN100", "source_count": len(selected), "question_count": len(public), "public_path": str(public_path.relative_to(root)).replace("\\", "/"), "hidden_path": str(hidden_path.relative_to(root)).replace("\\", "/"), "sources": integrity}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); report_path.write_text(json.dumps({"status": "pass", "archive_md5": archive_sha, "archive_size": archive.stat().st_size, "member_count": len(all_members), "selected_source_count": len(selected), "sources": integrity, "invalid_endpoint_total": sum(x["invalid_edge_endpoint_count"] for x in integrity)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "source_count": len(selected), "question_count": len(public), "manifest": str(manifest_path)}, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
