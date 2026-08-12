"""Aggregate efficiency-repeat telemetry and verify the fixed-run contract."""
from __future__ import annotations
import argparse, csv, json, math, random, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from pidbench.pidqa_metrics import normalize_pidqa_answer

TASKS = ("connectivity", "count", "spatial_count", "value")

def read(path: Path):
    with path.open("r", encoding="utf-8-sig") as h: return [json.loads(x) for x in h if x.strip()]

def quantile(values, q):
    if not values: return None
    x = sorted(float(v) for v in values); pos = (len(x)-1)*q; lo = math.floor(pos); hi = math.ceil(pos)
    return x[lo] if lo == hi else x[lo] + (x[hi]-x[lo])*(pos-lo)

def ci(values: dict[str, float], reps=5000, seed=1706):
    keys = sorted(values); rng = random.Random(seed); out = []
    for _ in range(reps): out.append(sum(values[rng.choice(keys)] for _ in keys)/len(keys))
    out.sort(); return out[round((len(out)-1)*.025)], out[round((len(out)-1)*.975)]

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--telemetry", default="outputs/telemetry/efficiency_repeats_v2.jsonl"); p.add_argument("--hidden", default="data/answer_store/main400_hashblind_set_b_hidden.jsonl"); p.add_argument("--output-dir", default="reports/generated"); a = p.parse_args()
    root = Path(a.root).resolve(); rows = read(root/a.telemetry); hidden = {str(x["instance_id"]): x for x in read(root/a.hidden)}; measured = [x for x in rows if x.get("phase") == "measure"]; groups = defaultdict(list)
    for row in measured: groups[(str(row.get("family")), str(row.get("condition")), int(row.get("max_image_side")))].append(row)
    output_rows = []; details = {}
    for key, group in sorted(groups.items()):
        family, condition, side = key; correct = []
        for row in group:
            truth = hidden.get(str(row.get("instance_id"))); pred = normalize_pidqa_answer(row.get("answer"), str(truth.get("task"))) if truth else None; gold = normalize_pidqa_answer(truth.get("answer"), str(truth.get("task"))) if truth else None; correct.append(int(row.get("status") == "ok" and pred == gold))
        lat = [float(x["cpu_latency_seconds"]) for x in group if x.get("cpu_latency_seconds") is not None]; gen = [float(x["generate_latency_seconds"]) for x in group if x.get("generate_latency_seconds") is not None]; acc = sum(correct)/max(1,len(correct)); by_source = defaultdict(list)
        for row, ok in zip(group, correct): by_source[str(row.get("source_id"))].append((float(row.get("cpu_latency_seconds") or 0), ok))
        src_acc = {s: sum(v for _,v in vals)/len(vals) for s, vals in by_source.items()}; src_lat = {s: sum(v for v,_ in vals)/len(vals) for s, vals in by_source.items()}; acc_ci = ci(src_acc); lat_ci = ci(src_lat)
        out = {"family": family, "condition": condition, "max_image_side": side, "record_count": len(group), "repeat_count": len({int(x.get("repeat",0)) for x in group}), "accuracy": acc, "accuracy_bootstrap_ci95_low": acc_ci[0], "accuracy_bootstrap_ci95_high": acc_ci[1], "cpu_latency_mean_seconds": statistics.mean(lat) if lat else None, "cpu_latency_median_seconds": statistics.median(lat) if lat else None, "cpu_latency_p95_seconds": quantile(lat,.95), "cpu_latency_source_bootstrap_ci95_low": lat_ci[0], "cpu_latency_source_bootstrap_ci95_high": lat_ci[1], "generate_latency_mean_seconds": statistics.mean(gen) if gen else None, "generate_latency_median_seconds": statistics.median(gen) if gen else None, "generate_latency_p95_seconds": quantile(gen,.95), "requests_per_second": 1/statistics.mean(lat) if lat and statistics.mean(lat)>0 else None, "seconds_per_correct_result": statistics.mean(lat)/acc if lat and acc>0 else None, "output_token_mean": statistics.mean([float(x["output_token_count"]) for x in group if x.get("output_token_count") is not None]) if any(x.get("output_token_count") is not None for x in group) else None, "peak_memory_allocated_gib": max([float(x.get("gpu_peak_memory_allocated_bytes",0))/2**30 for x in group] or [0]), "peak_memory_reserved_gib": max([float(x.get("gpu_peak_memory_reserved_bytes",0))/2**30 for x in group] or [0]), "gpu_uuid_count": len({str(x.get("gpu_uuid")) for x in group}), "invalid_count": sum(str(x.get("status")) != "ok" for x in group)}; output_rows.append(out); details[f"{family}|{condition}|{side}"] = {"rows": group, "correct": correct}
    expected = 8*3*100; keys = {(x.get("family"),x.get("condition"),x.get("max_image_side"),x.get("repeat"),x.get("instance_id")) for x in measured}; audit = {"status": "pass" if len(measured) == expected and len(keys) == len(measured) else "fail", "expected_measurement_rows": expected, "actual_measurement_rows": len(measured), "unique_measurement_keys": len(keys), "duplicate_measurement_rows": len(measured)-len(keys), "warmup_rows": sum(x.get("phase") == "warmup" for x in rows), "condition_count": len(groups), "power_measurement_included": False, "notes": "GPU power was not sampled; no joules/request claim is made."}
    outdir = root/a.output_dir; outdir.mkdir(parents=True, exist_ok=True); csv_path = outdir/"efficiency_frontier_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(output_rows[0]) if output_rows else ["family"]); w.writeheader(); w.writerows(output_rows)
    (outdir/"efficiency_measurement_audit_v2.json").write_text(json.dumps({"audit": audit, "groups": output_rows}, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    render_frontier(output_rows, outdir/"efficiency_frontier_v2.svg"); print(json.dumps({"status": audit["status"], "groups": len(output_rows), "measurements": len(measured), "csv": str(csv_path)}, indent=2, sort_keys=True)); return 0

def render_frontier(rows, path: Path):
    width,height=900,540; left,top,right,bottom=90,55,30,75; xs=[float(r["cpu_latency_mean_seconds"] or 0) for r in rows]; ys=[float(r["accuracy"]) for r in rows]; xmax=max(xs or [1])*1.15; ymax=max(ys or [.1])*.1+max(ys or [.1]); ymin=0
    def esc(x): return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    def txt(x,y,v,s=12,a="start",w="400"): return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{s}" text-anchor="{a}" font-weight="{w}">{esc(v)}</text>'
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',txt(25,28,"Accuracy–latency efficiency frontier (fixed subset, 3 repeats)",20,w="700")]
    out += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',txt(left, height-35,"mean CPU latency (s)",13),txt(20,(top+height-bottom)/2,"accuracy",13,"middle")]
    for r in rows:
        x=left+(float(r["cpu_latency_mean_seconds"] or 0)/xmax)*(width-left-right); y=height-bottom-(float(r["accuracy"])/max(ymax,.01))*(height-top-bottom); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1f77b4"/>'); out.append(txt(x+7,y-6,f"{r['family']}:{r['condition']}",10))
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(out+["</svg>"]) + "\n",encoding="utf-8")

if __name__ == "__main__": raise SystemExit(main())
