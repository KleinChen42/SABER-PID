"""Freeze the 100-question stratified Set-B subset for efficiency repeats."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

TASKS = ("connectivity", "count", "spatial_count", "value")

def read(path: Path):
    with path.open("r", encoding="utf-8-sig") as h: return [json.loads(x) for x in h if x.strip()]

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--input", default="data/processed/main400_hashblind_set_b_public.jsonl"); p.add_argument("--per-task", type=int, default=25); a = p.parse_args()
    root = Path(a.root).resolve(); rows = read(root / a.input); chosen = []
    for task in TASKS:
        group = [r for r in rows if str(r.get("task")) == task]
        group.sort(key=lambda r: hashlib.sha256(f"efficiency-v2|{r['instance_id']}".encode()).hexdigest())
        chosen.extend(group[: a.per_task])
    chosen.sort(key=lambda r: (TASKS.index(str(r["task"])), str(r["source_id"]), str(r["instance_id"])))
    remote = []
    for r in chosen:
        x = dict(r); x["image_path"] = f"data/raw/main400_flat/{r['source_sheet']}.jpg"; remote.append(x)
    out_jsonl = root / "data/processed/efficiency_subset_v2_public.jsonl"; out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as h:
        for r in remote: h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(out_jsonl.read_bytes()).hexdigest(); manifest = {"status": "pass", "selection_rule": "within-task SHA-256(instance_id) ascending", "seed_label": "efficiency-v2", "source_input": a.input, "record_count": len(remote), "per_task": {t: sum(str(r["task"]) == t for r in remote) for t in TASKS}, "source_count": len({str(r["source_id"]) for r in remote}), "public_path": str(out_jsonl.relative_to(root)).replace("\\", "/"), "public_sha256": digest, "image_root_contract": "data/raw/main400_flat/{source_sheet}.jpg"}
    mpath = root / "data/manifests/efficiency_subset_v2.json"; mpath.parent.mkdir(parents=True, exist_ok=True); mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
