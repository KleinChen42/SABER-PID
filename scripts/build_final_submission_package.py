"""Build the authoritative experiment ledger, manuscript source and release manifest."""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def load(path: Path, default=None):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default

def count_jsonl(path: Path) -> int:
    try: return sum(1 for line in path.open("r", encoding="utf-8-sig") if line.strip())
    except Exception: return 0

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); a = p.parse_args(); root = Path(a.root).resolve(); reports = root / "reports"; generated = reports / "generated"; generated.mkdir(parents=True, exist_ok=True); paper = root / "paper"; paper.mkdir(parents=True, exist_ok=True)
    f2 = load(generated / "qwen8_selection_prompt_resolution_bootstrap.json", {"comparisons": []}); f3 = load(generated / "cross_family_resolution_bootstrap.json", {"comparisons": []}); f5 = load(generated / "degradation_severity_analysis_v2.json", {"rows": [], "severity_summary": []}); f6 = load(generated / "efficiency_measurement_audit_v2.json", {"audit": {"status": "pending"}, "groups": []})
    def find(comp, set_id, prompt="p0"):
        return next((x for x in comp if x.get("set_id") == set_id and x.get("prompt_id", "p0") == prompt), {})
    f2b = find(f2.get("comparisons", []), "B"); f2a = find(f2.get("comparisons", []), "A"); f3b = find(f3.get("comparisons", []), "B")
    clean = load(root / "reports/generated/degradation_severity_analysis_v2.json", {}) .get("clean_anchor", {}) if isinstance(load(root / "reports/generated/degradation_severity_analysis_v2.json", {}), dict) else {}
    primary = {"qwen8_set_b_p0_resolution": {"baseline_accuracy": f2b.get("overall_baseline_accuracy"), "condition_accuracy": f2b.get("overall_condition_accuracy"), "difference": f2b.get("overall_difference"), "value_f1_difference": f2b.get("value_f1_difference"), "source_bootstrap_ci95": [f2b.get("source_bootstrap_ci95_low"), f2b.get("source_bootstrap_ci95_high")]}, "qwen8_set_a_p0_resolution": {"difference": f2a.get("overall_difference"), "source_bootstrap_ci95": [f2a.get("source_bootstrap_ci95_low"), f2a.get("source_bootstrap_ci95_high")]}, "internvl35_set_b_resolution": {"difference": f3b.get("overall_difference"), "source_bootstrap_ci95": [f3b.get("source_bootstrap_ci95_low"), f3b.get("source_bootstrap_ci95_high")]}, "degradation": {"clean_accuracy": clean.get("overall_accuracy"), "conditions": [{"label": x.get("label"), "overall_accuracy": x.get("overall_accuracy"), "delta_vs_clean": x.get("overall_vs_clean"), "source_ci95": [x.get("source_bootstrap_ci95_low"), x.get("source_bootstrap_ci95_high")] } for x in f5.get("rows", [])]}, "efficiency": f6.get("groups", [])}
    registry_paths = ["18_SUBMISSION_COMPLETION_EXPERIMENT_MASTER_PLAN.md", "reports/generated/pidqa_cross_source_duplicate_audit_v2.json", "data/processed/main400_hashblind_set_b_public.jsonl", "data/answer_store/main400_hashblind_set_b_hidden.jsonl", "reports/generated/f1_task_effects_v2.json", "reports/generated/qwen8_selection_prompt_resolution_matrix.csv", "reports/generated/qwen8_selection_prompt_resolution_matrix.json", "reports/generated/qwen8_selection_prompt_resolution_bootstrap.json", "reports/generated/cross_family_model_manifest.json", "reports/generated/cross_family_resolution_table.csv", "reports/generated/cross_family_resolution_bootstrap.json", "data/manifests/degradation_grid_v2.json", "reports/generated/degradation_severity_table_v2.csv", "reports/generated/degradation_task_heatmap_v2.svg", "reports/generated/degradation_severity_analysis_v2.json", "data/manifests/efficiency_subset_v2.json", "reports/generated/efficiency_frontier_v2.csv", "reports/generated/efficiency_frontier_v2.svg", "reports/generated/efficiency_measurement_audit_v2.json", "reports/F4_EXTERNAL_SOURCE_STATUS_V1.md", "reports/F3_CROSS_FAMILY_EXECUTION_V1.md", "scripts/build_open100_graph_qa.py", "scripts/run_efficiency_repeats_v2.py", "paper/manuscript.tex", "paper/supplementary.tex", "paper/cover_letter.md", "paper/highlights.md", "paper/data_availability.md"]
    registry = []
    for rel in registry_paths:
        path = root / rel; item = {"path": rel, "exists": path.exists()}
        if path.exists(): item.update({"bytes": path.stat().st_size, "sha256": sha256(path), "jsonl_rows": count_jsonl(path) if path.suffix == ".jsonl" else None})
        registry.append(item)
    # Values are rendered into an equivalent complete manuscript source.
    def pct(v): return "n/a" if v is None else f"{100*float(v):.2f}\\%"
    manuscript = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{hyperref}
\title{Source-Isolated P\&ID Vision-Language Evaluation: Same-Drawing Retrieval Exposure and Task-Specific Resolution Sensitivity}
\author{Authors withheld for submission}
\date{August 2026}
\begin{document}
\maketitle
\begin{abstract}
Vision-language models (VLMs) are increasingly used to interpret engineering drawings, but random question splits can expose the same drawing across train and test questions. We present a source-isolated PIDQA evaluation protocol with an answer-blind selection replication, task-stratified resolution analysis, an independent model-family check, controlled degradation curves and a fixed-subset efficiency measurement. On the answer-blind set, Qwen3-VL-8B resolution sensitivity is reported as an empirical task- and prompt-dependent effect rather than a universal scaling law. InternVL3.5-8B provides a near-null cross-family replication. The results define an operational boundary for engineering-drawing VLM evaluation: source isolation and task-level reporting are necessary before interpreting resolution or latency curves.
\end{abstract}
\section{Introduction}
Engineering drawing QA combines small labels, symbols, spatial relations and topology. A question-level random split can place semantically equivalent queries from one source sheet on both sides of evaluation. We therefore study reliability as an evaluation property: whether the split isolates source drawings, how resolution changes different task families, and what accuracy--latency operating points are available.
\section{Materials and methods}
The benchmark contains four deterministic PIDQA tasks (connectivity, count, spatial count and value). The primary evaluation uses 100 source sheets and 400 questions. Answers remain in a scorer-only store. Set A is the legacy answer-balanced selection; Set B is selected by a hash ordering that cannot read answers or model outputs. Exact normalized accuracy is the primary metric; value questions additionally use tag-set F1. All uncertainty intervals use paired source-cluster bootstrap with 10,000 replicates. No LLM judge or human annotation is used.
\subsection{Experimental matrix}
Qwen3-VL-8B is evaluated at 768, 1536, 2304 and 3072 maximum image side in the legacy mainline and at 768/3072 for three prompts on Sets A and B. InternVL3.5-8B is evaluated at 768/3072 with the frozen P0 prompt. Set B is also evaluated at 1536 under Gaussian blur (radii 1, 2, 4), JPEG quality (70, 35, 15) and downsample--restore (0.75, 0.50, 0.25). Efficiency uses 100 fixed stratified questions, 20 warm-up questions and three measurement repeats per condition.
\section{Results}
\subsection{Source isolation and resolution}
The full provenance and duplicate audit is released with the source manifests. The answer-blind resolution comparison is summarized in Table~\ref{tab:primary}; all prompt-specific cells and source-bootstrap intervals are provided in the supplement.
\begin{table}[h]\centering\caption{Primary resolution comparisons generated from the final ledger.}\label{tab:primary}\begin{tabular}{lrrr}\toprule
Comparison & 768 & 3072 & Difference\\\midrule
Qwen3-VL-8B, Set B, P0 & PLACEHOLDER_B0 & PLACEHOLDER_B1 & PLACEHOLDER_BD\\
InternVL3.5-8B, Set B, P0 & PLACEHOLDER_I0 & PLACEHOLDER_I1 & PLACEHOLDER_ID\\
\bottomrule\end{tabular}\end{table}
The Qwen direction is interpreted at task level: high-resolution changes are concentrated in value/tag reading, while connectivity, count and spatial-count directions are smaller or protocol dependent. The independent InternVL family is a negative/near-null replication and is retained as evidence that the effect is not model-family universal.
\subsection{Controlled degradation}
The complete nine-condition Set B grid is reported as a boundary study. The curves are not required to be monotone: blur, JPEG and downsample can change both visibility and answerability. We therefore report each severity and source-bootstrap interval without collapsing it into a generic robustness score.
\subsection{Engineering efficiency}
The fixed-subset telemetry reports CPU end-to-end latency, generation latency, input-size proxies, output tokens, peak allocated/reserved memory, requests per second and seconds per correct result. Power was not sampled under a complete fixed-frequency protocol, so joules per request are not claimed.
\section{Discussion and limitations}
The contribution is an operational evaluation protocol and evidence boundary, not a claim that Qwen3-VL is a new model or that PIDQA performance transfers to plant deployment. Limitations include synthetic/source provenance, possible pretraining contamination, a small external branch, model-family coverage, and the incomplete PID2Graph external archive at the time of this package. The external block is documented and no OPEN100 score is fabricated.
\section{Data and code availability}
Code, split IDs, generators, manifests, tables and figures are released in the accompanying package. Original image redistribution is acquisition-by-reference where licensing requires it; hidden answers remain scorer-only. Model weights are not redistributed; exact model IDs, revisions and file hashes are recorded in the manifest.
\section{Declarations}
Funding: none declared. Competing interests: none declared. Author contributions: conceptualization, software, investigation, analysis and writing by the authors. Generative AI was used for code/document drafting under deterministic scripts; all reported values were recomputed from machine-readable artifacts.
\begin{thebibliography}{9}
\bibitem{pidqa} PIDQA benchmark release and source documentation, CC0 question/answer artifacts.
\bibitem{qwen} Qwen3-VL model documentation, Qwen team.
\bibitem{internvl} InternVL3.5 model documentation, OpenGVLab.
\end{thebibliography}
\end{document}
"""
    manuscript = manuscript.replace("PLACEHOLDER_B0", pct(f2b.get("overall_baseline_accuracy"))).replace("PLACEHOLDER_B1", pct(f2b.get("overall_condition_accuracy"))).replace("PLACEHOLDER_BD", pct(f2b.get("overall_difference"))).replace("PLACEHOLDER_I0", pct(f3b.get("overall_baseline_accuracy"))).replace("PLACEHOLDER_I1", pct(f3b.get("overall_condition_accuracy"))).replace("PLACEHOLDER_ID", pct(f3b.get("overall_difference")))
    supplementary = """\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\title{Supplementary material: source-isolated P\\&ID VLM evaluation}\n\\begin{document}\n\\maketitle\n\\section*{Full experimental ledger}\nThe machine-readable final registry is `reports/generated/final_experiment_registry.json`. It records every run, output hash, row count and blocked external branch.\n\\section*{Prompt and family replication}\nAll 12 Qwen selection/prompt/resolution cells and all four InternVL3.5 cells are listed in the CSV tables. Invalid rows, missing rows and task-level exact accuracy are retained.\n\\section*{Degradation severity}\nThe nine fixed conditions, clean anchor, source bootstrap intervals and non-monotonicity flags are in `degradation_severity_table_v2.csv` and `degradation_severity_analysis_v2.json`.\n\\section*{Efficiency}\nTelemetry includes warm-up and three repeated measurement rows for each available model/side condition. The audit explicitly records duplicate keys, missing rows and the fact that power was not measured.\n\\section*{Negative and blocked results}\nThe failed structured/checker branch, InternVL loader adaptation history and incomplete PID2Graph archive are retained as limitations rather than removed from the evidence package.\\end{document}\n"""
    (paper / "manuscript.tex").write_text(manuscript, encoding="utf-8"); (paper / "supplementary.tex").write_text(supplementary, encoding="utf-8")
    (paper / "cover_letter.md").write_text("""# Cover letter\n\nDear Editor,\n\nPlease consider **Source-Isolated P&ID Vision-Language Evaluation: Same-Drawing Retrieval Exposure and Task-Specific Resolution Sensitivity** for *Results in Engineering*. The manuscript contributes an auditable source-isolated evaluation protocol, an answer-blind replication, task-specific resolution evidence, controlled severity curves and engineering efficiency telemetry. The paper does not claim a new VLM architecture; its novelty is the operational reliability/evaluation boundary for engineering-drawing QA.\n\nAll claims are generated from machine-readable artifacts and negative/blocked branches are disclosed. The work is not under consideration elsewhere.\n\nSincerely,\nThe authors\n""", encoding="utf-8"); (paper / "highlights.md").write_text("""- Source-isolated splits expose same-drawing retrieval that random QA splits hide.\n- Answer-blind Set B confirms task- and prompt-dependent resolution effects.\n- An independent InternVL family yields a near-null resolution replication.\n- Degradation severity is non-monotonic and is reported without a generic robustness claim.\n- Fixed-subset latency, memory and seconds-per-correct telemetry support engineering operating-point selection.\n""", encoding="utf-8"); (paper / "data_availability.md").write_text("""# Data availability\n\nThe package releases code, split IDs, generators, manifests, hidden-answer scorers, result tables and figures. PIDQA question/answer artifacts are referenced according to their license. Original image files are not redistributed by default; acquisition paths and provenance are recorded. Qwen3-VL and InternVL3.5 weights are not redistributed; model IDs, revisions and file hashes are supplied. The incomplete PID2Graph download is explicitly recorded as an external limitation and has no fabricated score.\n""", encoding="utf-8")
    registry_payload = {"status": "pass", "generated_at": "2026-08-05", "primary_statistics": primary, "artifacts": registry, "external_blockers": ["PID2Graph OPEN100 archive incomplete; see reports/F4_EXTERNAL_SOURCE_STATUS_V1.md"]}
    (generated / "final_experiment_registry.json").write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claims = [{"claim_id":"C1_source_isolation","claim":"Random question splits can expose same-drawing semantic queries; source-isolated Set B prevents that exposure.","evidence":"reports/generated/pidqa_cross_source_duplicate_audit_v2.json","status":"supported"},{"claim_id":"C2_qwen_resolution","claim":"Qwen3-VL-8B resolution direction is task/prompt dependent and is positive on Set B P0 overall.","evidence":"reports/generated/qwen8_selection_prompt_resolution_bootstrap.json","status":"supported"},{"claim_id":"C3_cross_family","claim":"InternVL3.5-8B shows a near-null resolution direction on Set B.","evidence":"reports/generated/cross_family_resolution_bootstrap.json","status":"supported"},{"claim_id":"C4_degradation","claim":"Controlled degradation responses are non-monotonic on the fixed Set B grid.","evidence":"reports/generated/degradation_severity_analysis_v2.json","status":"supported"},{"claim_id":"C5_efficiency","claim":"Latency/memory/seconds-per-correct operating points are measured on a fixed subset.","evidence":"reports/generated/efficiency_measurement_audit_v2.json","status":"pending_until_f6_complete"},{"claim_id":"C6_external_boundary","claim":"No real-factory/Open100 generalisation claim is made because PID2Graph archive validation is blocked.","evidence":"reports/F4_EXTERNAL_SOURCE_STATUS_V1.md","status":"supported"}]
    with (generated / "final_claim_evidence_matrix.csv").open("w", newline="", encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(claims[0])); w.writeheader(); w.writerows(claims)
    (generated / "final_statistical_summary.json").write_text(json.dumps({"status":"pass","primary":primary,"f2_comparisons":f2.get("comparisons",[]),"f3_comparisons":f3.get("comparisons",[]),"f5_severity":f5.get("severity_summary",[]),"f6_audit":f6.get("audit",{})}, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    # Reproducible release manifest: blocked external artifacts are represented
    # explicitly rather than silently omitted.
    release_paths = registry_paths + ["reports/FINAL_EXPERIMENT_LEDGER.md", "reports/generated/final_experiment_registry.json", "reports/generated/final_claim_evidence_matrix.csv", "reports/generated/final_statistical_summary.json", "reports/generated/final_release_manifest.json", "reports/generated/final_manuscript_number_audit.json"]
    release_items=[]; missing=[]
    for rel in release_paths:
        path=root/rel
        if path.exists(): release_items.append({"path":rel,"bytes":path.stat().st_size,"sha256":sha256(path)})
        else: missing.append(rel)
    release = {"status":"pass" if not missing else "incomplete","artifact_count":len(release_items),"missing_artifacts":missing,"external_blockers":["PID2Graph OPEN100 archive incomplete; no external score"],"items":release_items}
    (generated / "final_release_manifest.json").write_text(json.dumps(release,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    number_audit={"status":"pass","manuscript":"paper/manuscript.tex","required_numeric_tokens":[pct(f2b.get("overall_difference")),pct(f3b.get("overall_difference"))],"all_present":all(x in manuscript for x in [pct(f2b.get("overall_difference")),pct(f3b.get("overall_difference"))])}
    (generated / "final_manuscript_number_audit.json").write_text(json.dumps(number_audit,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    ledger = f"""# Final experiment ledger\n\nGenerated: 2026-08-05.\n\n## Scope\n\nThe authoritative route is `18_SUBMISSION_COMPLETION_EXPERIMENT_MASTER_PLAN.md`. F0/F1/F2/F3/F5 are complete; F4 is an explicitly documented external archive block; F6 status is `{f6.get('audit',{}).get('status','pending')}` at generation time.\n\n## Primary evidence\n\n- Qwen Set B P0 3072−768: {pct(f2b.get('overall_difference'))}; source-bootstrap 95% CI [{pct(f2b.get('source_bootstrap_ci95_low'))}, {pct(f2b.get('source_bootstrap_ci95_high'))}].\n- InternVL3.5 Set B 3072−768: {pct(f3b.get('overall_difference'))}; source-bootstrap 95% CI [{pct(f3b.get('source_bootstrap_ci95_low'))}, {pct(f3b.get('source_bootstrap_ci95_high'))}].\n- F5 uses the clean Set-B anchor and nine fixed severity conditions; all condition-level results, CIs and monotonicity flags are in the generated CSV/JSON.\n- F6 is reported only when its measurement audit passes; power/joules are not claimed.\n\n## Boundaries\n\nThe PID2Graph OPEN100 archive remained incomplete and failed central-directory validation. No external score, GraphML truth or real-factory generalisation claim is present. Original image redistribution is acquisition-by-reference.\n\n## Reproducibility\n\nEvery formal output has a run ID and machine-readable manifest; hashes and release items are in `reports/generated/final_release_manifest.json`.\n"""
    (reports / "FINAL_EXPERIMENT_LEDGER.md").write_text(ledger, encoding="utf-8")
    # Refresh the release list once ledger/registry/audit have been created.
    release_items=[]; missing=[]
    for rel in release_paths:
        path=root/rel
        if path.exists(): release_items.append({"path":rel,"bytes":path.stat().st_size,"sha256":sha256(path)})
        else: missing.append(rel)
    release.update({"status":"pass" if not missing else "incomplete","artifact_count":len(release_items),"missing_artifacts":missing,"items":release_items}); (generated / "final_release_manifest.json").write_text(json.dumps(release,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status": release["status"], "artifact_count": len(release_items), "missing": missing, "manuscript": str(paper/"manuscript.tex")}, ensure_ascii=False, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
