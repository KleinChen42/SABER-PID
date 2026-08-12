"""Score F5 degradation conditions with Set-B hidden answers and source CIs."""
from __future__ import annotations
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
from typing import Any
from pidbench.pidqa_metrics import normalize_pidqa_answer

TASKS = ("connectivity", "count", "spatial_count", "value")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as h:
        return [json.loads(x) for x in h if x.strip()]

def correct(rec: dict[str, Any], pred: dict[str, Any] | None) -> int:
    if pred is None: return 0
    action = str(pred.get("action", "ANSWER"))
    if "action" not in pred and str(pred.get("status", "ok")) == "ok": action = "ANSWER"
    return int(action == "ANSWER" and normalize_pidqa_answer(pred.get("answer"), str(rec["task"])) == normalize_pidqa_answer(rec.get("answer"), str(rec["task"])))

def tag_counts(rec: dict[str, Any], pred: dict[str, Any] | None) -> tuple[int, int, int]:
    truth = set(normalize_pidqa_answer(rec.get("answer"), "value") or ())
    got = set(normalize_pidqa_answer(pred.get("answer"), "value") or ()) if pred else set()
    if pred and str(pred.get("action", "ANSWER")) != "ANSWER": got = set()
    return len(truth & got), len(got - truth), len(truth - got)

def metrics(records: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(x.get("instance_id")): x for x in preds}
    good = {str(r["instance_id"]): correct(r, by_id.get(str(r["instance_id"]))) for r in records}
    task = {t: sum(good[str(r["instance_id"])] for r in records if str(r["task"]) == t) / max(1, sum(str(r["task"]) == t for r in records)) for t in TASKS}
    src: dict[str, list[int]] = defaultdict(list)
    for r in records: src[str(r["source_id"])].append(good[str(r["instance_id"])])
    tp = fp = fn = 0
    for r in records:
        if str(r["task"]) == "value":
            a, b, c = tag_counts(r, by_id.get(str(r["instance_id"]))); tp += a; fp += b; fn += c
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"record_count": len(records), "prediction_count": len(by_id), "missing_prediction_count": len(set(str(r["instance_id"]) for r in records) - set(by_id)), "invalid_prediction_count": sum(str(x.get("action", "ANSWER")) != "ANSWER" for x in preds), "overall_accuracy": sum(good.values()) / max(1, len(records)), "task_accuracy": task, "source_macro_accuracy": sum(sum(v) / len(v) for v in src.values()) / max(1, len(src)), "source_accuracy": {s: sum(v) / len(v) for s, v in sorted(src.items())}, "value_tag_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "value_tag_precision": precision, "value_tag_recall": recall}

def bootstrap(values: dict[str, float], reps: int = 10000, seed: int = 1702) -> tuple[float, float]:
    keys = sorted(values); rng = random.Random(seed); draws = [sum(values[rng.choice(keys)] for _ in keys) / len(keys) for _ in range(reps)]; draws.sort()
    return draws[round((len(draws) - 1) * .025)], draws[round((len(draws) - 1) * .975)]

def render_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    cols = ["connectivity_accuracy", "count_accuracy", "spatial_count_accuracy", "value_accuracy", "overall_accuracy"]; names = ["Connectivity", "Count", "Spatial", "Value", "Overall"]; left, top, cw, ch = 245, 78, 125, 42; width = 960; height = top + len(rows) * ch + 50
    def esc(x: str) -> str: return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    def txt(x: float, y: float, value: str, size: int = 13, anchor: str = "start", weight: str = "400") -> str: return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{esc(value)}</text>'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', txt(30, 30, "F5 degradation severity — Qwen3-VL 8B, Set B, 1536", 20, weight="700")]
    for j, name in enumerate(names): svg.append(txt(left + j * cw + cw / 2, top - 15, name, 12, "middle", "700"))
    for i, row in enumerate(rows):
        svg.append(txt(left - 12, top + i * ch + 25, str(row["label"]), 12, "end", "700"))
        for j, key in enumerate(cols):
            val = float(row[key]); q = max(0.0, min(1.0, val / .6)); red = int(247 - 130*q); green = int(247 - 70*q); blue = int(247 - 25*q); x = left + j*cw; y = top + i*ch
            svg.append(f'<rect x="{x}" y="{y}" width="{cw-4}" height="{ch-4}" fill="rgb({red},{green},{blue})" stroke="white"/>'); svg.append(txt(x + (cw-4)/2, y+24, f"{val*100:.1f}%", 13, "middle", "700"))
    svg.append(txt(30, height-18, "Exact/task accuracy; source-cluster CIs against clean Set-B anchor are in the CSV/JSON.", 12)); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(svg + ["</svg>"]) + "\n", encoding="utf-8")

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--manifest", default="data/manifests/degradation_grid_v2.json"); p.add_argument("--prediction-dir", default="outputs/final_degradation"); p.add_argument("--output-dir", default="reports/generated"); p.add_argument("--clean", required=True); a = p.parse_args()
    root = Path(a.root).resolve(); manifest = json.loads((root / a.manifest).read_text(encoding="utf-8")); hidden = read_jsonl(root / "data/answer_store/main400_hashblind_set_b_hidden.jsonl"); clean_path = root / a.clean; clean = metrics(hidden, read_jsonl(clean_path)); rows = []; details = {}; family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest["conditions"]:
        condition = item["condition"]; pred_path = root / a.prediction_dir / f"qwen8_set_b_{condition}.jsonl"; m = metrics(hidden, read_jsonl(pred_path)); row = {"label": condition, "family": item["family"], "severity": item["severity"], **{f"{t}_accuracy": m["task_accuracy"][t] for t in TASKS}, "overall_accuracy": m["overall_accuracy"], "source_macro_accuracy": m["source_macro_accuracy"], "value_tag_f1": m["value_tag_f1"], "overall_vs_clean": m["overall_accuracy"] - clean["overall_accuracy"], "value_f1_vs_clean": m["value_tag_f1"] - clean["value_tag_f1"], "missing_prediction_count": m["missing_prediction_count"], "invalid_prediction_count": m["invalid_prediction_count"]}
        diffs = {s: m["source_accuracy"].get(s, 0.0) - clean["source_accuracy"].get(s, 0.0) for s in clean["source_accuracy"]}; row["source_bootstrap_ci95_low"], row["source_bootstrap_ci95_high"] = bootstrap(diffs); rows.append(row); family_rows[item["family"]].append(row); details[condition] = {"manifest": item, "prediction_path": str(pred_path.relative_to(root)).replace("\\", "/"), **m}
    severity = []
    for family, group in sorted(family_rows.items()):
        ordered = sorted(group, key=lambda x: float(x["severity"])); xs = [float(x["severity"]) for x in ordered]; ys = [float(x["overall_accuracy"]) for x in ordered]; xm = sum(xs)/len(xs); ym = sum(ys)/len(ys); den = sum((x-xm)**2 for x in xs); sl = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/den if den else 0.0
        severity.append({"family": family, "conditions_in_ascending_severity": [x["label"] for x in ordered], "overall_values": ys, "overall_slope_per_severity_unit": sl, "monotonic_nonincreasing": all(ys[i] >= ys[i+1] for i in range(len(ys)-1))})
    out = root / a.output_dir; out.mkdir(parents=True, exist_ok=True); csv_path = out / "degradation_severity_table_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as h: w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    render_heatmap(rows, out / "degradation_task_heatmap_v2.svg"); (out / "degradation_severity_analysis_v2.json").write_text(json.dumps({"status": "pass", "manifest": a.manifest, "clean_anchor": {"path": str(clean_path.relative_to(root)).replace("\\", "/"), **clean}, "rows": rows, "details": details, "severity_summary": severity, "bootstrap_method": "paired source-cluster bootstrap", "bootstrap_reps": 10000, "bootstrap_seed": 1702}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"status": "pass", "conditions": len(rows), "table": str(csv_path)}, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
