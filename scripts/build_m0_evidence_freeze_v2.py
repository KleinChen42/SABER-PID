"""Freeze the manuscript claim/evidence contract after E1--E6 and F4.

The builder reads machine-readable artifacts only.  It does not score model
outputs, alter raw JSONL, select a result, or infer an unsupported claim.  It
emits the M0 deliverables named by the authoritative submission charter:
``final_statistical_summary_v2.json``, ``final_claim_evidence_matrix_v2.csv``,
and ``PAPER_EVIDENCE_STRENGTHENING_CLOSEOUT.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "claim_id",
        "status",
        "manuscript_statement",
        "numerical_support",
        "evidence_artifacts",
        "scope_boundary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison(report: dict[str, Any], label: str, metric: str, task: str) -> dict[str, Any]:
    for row in report["comparisons"]:
        if row["comparison"] == label and row["metric"] == metric and row["task"] == task:
            return row
    raise KeyError(f"Missing comparison {label}/{metric}/{task}")


def f(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def ci(row: dict[str, Any], digits: int = 4) -> str:
    return f"[{f(row['source_bootstrap_ci95_low'], digits)}, {f(row['source_bootstrap_ci95_high'], digits)}]"


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a mean of no values")
    return statistics.mean(values)


def r1_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    result: dict[str, Any] = {}
    for split in ("random", "source"):
        selected = [row for row in rows if row["split"] == split and row["method"] == "L3_image_semantic"]
        if len(selected) != 5:
            raise RuntimeError(f"Expected five L3 rows for {split}, found {len(selected)}")
        coverage = [float(row["coverage"]) for row in selected]
        accuracy = [float(row["overall_accuracy"]) for row in selected]
        result[split] = {
            "seed_count": len(selected),
            "coverage_mean": mean(coverage),
            "coverage_min": min(coverage),
            "coverage_max": max(coverage),
            "overall_accuracy_mean": mean(accuracy),
        }
    return result


def source_seed_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for seed in (29, 31):
        label = f"e6_seed{seed}_3072_minus_768"
        strict_overall = comparison(report, label, "strict_correct", "overall")
        semantic_overall = comparison(report, label, "semantic_correct", "overall")
        strict_value = comparison(report, label, "strict_correct", "value")
        semantic_value = comparison(report, label, "semantic_correct", "value")
        strict_f1 = comparison(report, label, "strict_value_tag_f1", "value")
        rows[str(seed)] = {
            "strict_overall": strict_overall,
            "semantic_overall": semantic_overall,
            "strict_value_exact": strict_value,
            "semantic_value_exact": semantic_value,
            "strict_value_tag_f1": strict_f1,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/generated")
    parser.add_argument("--closeout", default="reports/PAPER_EVIDENCE_STRENGTHENING_CLOSEOUT.md")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    generated = root / "reports" / "generated"

    r1 = r1_summary(read_json(generated / "pidqa_input_retrieval_seed_sweep.json"))
    prior = read_json(generated / "set_b_task_prior_v2.json")
    e2 = read_json(generated / "qwen8_value_budget_sensitivity_v1.json")
    e3 = read_json(generated / "image_dependence_control_v1.json")
    e4 = read_json(generated / "internvl_tile_budget_v1.json")
    e5 = read_json(generated / "ontology_visibility_effect_v1.json")
    e6 = read_json(generated / "source_seed_resolution_sensitivity_v1.json")
    f4 = read_json(generated / "pid2graph_recheck_v1.json")
    input_isolation = read_json(generated / "evidence_input_answer_isolation_audit_v1.json")
    if input_isolation.get("status") != "pass":
        raise RuntimeError("Evidence input answer-isolation audit did not pass")

    e2_value = comparison(e2, "e2_512_3072_minus_768", "strict_value_tag_f1", "value")
    e3_value = comparison(e3, "e3_shuffled_minus_correct_3072", "strict_value_tag_f1", "value")
    e3_structural = {
        task: comparison(e3, "e3_shuffled_minus_correct_3072", "strict_correct", task)
        for task in ("connectivity", "count", "spatial_count")
    }
    e4_strict = comparison(e4, "e4_high_minus_low_actual_tile_budget", "strict_correct", "overall")
    e4_semantic = comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "overall")
    e4_spatial = comparison(e4, "e4_high_minus_low_actual_tile_budget", "semantic_correct", "spatial_count")
    e5_spatial_768 = comparison(e5, "e5_ontology_visible_minus_raw_768", "semantic_correct", "spatial_count")
    e5_spatial_3072 = comparison(e5, "e5_ontology_visible_minus_raw_3072", "semantic_correct", "spatial_count")
    e5_overall_768 = comparison(e5, "e5_ontology_visible_minus_raw_768", "semantic_correct", "overall")
    e5_overall_3072 = comparison(e5, "e5_ontology_visible_minus_raw_3072", "semantic_correct", "overall")
    e6_seeds = source_seed_summary(e6)

    prior_metrics = prior["metrics"]
    qwen_b_3072 = e3["cells"]["qwen8_b_p0_correct_3072"]["metrics"]
    seed_text = "; ".join(
        f"seed {seed}: strict overall {f(row['strict_overall']['difference_condition_minus_baseline'])} {ci(row['strict_overall'])}, "
        f"value F1 {f(row['strict_value_tag_f1']['difference_condition_minus_baseline'])} {ci(row['strict_value_tag_f1'])}"
        for seed, row in e6_seeds.items()
    )

    claims = [
        {
            "claim_id": "C0",
            "status": "PROTOCOL_VALIDATED",
            "manuscript_statement": "E2, E3, E5, and E6 inference used answer-isolated public manifests; the output `answer` field is an alias of the generated raw response, not a reference label.",
            "numerical_support": "Five public input manifests each contain 400 unique records with no `answer` or `cypher` field; all 10 audited output files have only successful rows and `answer == raw` for every row.",
            "evidence_artifacts": "reports/generated/evidence_input_answer_isolation_audit_v1.json; scripts/audit_evidence_input_isolation.py",
            "scope_boundary": "This validates the audited runner and manifests; it cannot rule out model pretraining exposure or establish broader benchmark generalization.",
        },
        {
            "claim_id": "C1",
            "status": "SUPPORTED",
            "manuscript_statement": "Question-random PIDQA splits retain a measurable same-image semantic retrieval route that source isolation removes.",
            "numerical_support": f"Five-seed L3 coverage: random {f(r1['random']['coverage_mean'], 6)} (range {f(r1['random']['coverage_min'], 6)}--{f(r1['random']['coverage_max'], 6)}), source {f(r1['source']['coverage_mean'], 6)}.",
            "evidence_artifacts": "reports/generated/pidqa_input_retrieval_seed_sweep.json; reports/R1_INPUT_RETRIEVAL_CLOSEOUT.md",
            "scope_boundary": "This is an input-recoverable exposure diagnostic, not proof that every trained VLM score is inflated.",
        },
        {
            "claim_id": "C2",
            "status": "SUPPORTED",
            "manuscript_statement": "Source isolation alone does not establish visual reasoning beyond a matched task prior.",
            "numerical_support": f"Set-B training-only task prior strict accuracy {f(prior_metrics['strict_accuracy'])}; Qwen P0 3072 strict accuracy {f(qwen_b_3072['strict_accuracy'])}. Prior task accuracies: connectivity {f(prior_metrics['task']['connectivity']['strict_accuracy'])}, count {f(prior_metrics['task']['count']['strict_accuracy'])}, spatial-count {f(prior_metrics['task']['spatial_count']['strict_accuracy'])}, value {f(prior_metrics['task']['value']['strict_accuracy'])}.",
            "evidence_artifacts": "reports/generated/set_b_task_prior_v2.json; reports/E1_EVIDENCE_AUDIT_CLOSEOUT.md",
            "scope_boundary": "Aggregate accuracy must not be presented as a general topology-understanding result.",
        },
        {
            "claim_id": "C3",
            "status": "SUPPORTED",
            "manuscript_statement": "For the frozen Qwen setup, the high-side value/tag-reading advantage persists under a 512-token budget and depends on the correctly paired P&ID image.",
            "numerical_support": f"E2 3072-768 strict value F1 {f(e2_value['difference_condition_minus_baseline'])} {ci(e2_value)}; E3 shuffled-correct 3072 strict value F1 {f(e3_value['difference_condition_minus_baseline'])} {ci(e3_value)}.",
            "evidence_artifacts": "reports/generated/qwen8_value_budget_sensitivity_v1.json; reports/generated/image_dependence_control_v1.json; reports/E2_VALUE_BUDGET_CLOSEOUT.md; reports/E3_IMAGE_DEPENDENCE_CONTROL_CLOSEOUT.md",
            "scope_boundary": "The evidence is task-specific tag reading, not uniform P&ID reasoning or a universal resolution law.",
        },
        {
            "claim_id": "C4",
            "status": "SUPPORTED",
            "manuscript_statement": "The same image-shuffle control leaves evaluated structural tasks near their task-prior-constrained behavior.",
            "numerical_support": "; ".join(f"{task} strict delta {f(row['difference_condition_minus_baseline'])} {ci(row)}" for task, row in e3_structural.items()),
            "evidence_artifacts": "reports/generated/image_dependence_control_v1.json; reports/E3_IMAGE_DEPENDENCE_CONTROL_CLOSEOUT.md",
            "scope_boundary": "Near-null changes do not prove no image use in every structural task; they rule out the claimed general structural interpretation on this setup.",
        },
        {
            "claim_id": "C5",
            "status": "SUPPORTED",
            "manuscript_statement": "Numeric class-ontology visibility is a material evaluation condition for PIDQA spatial-count questions.",
            "numerical_support": f"Semantic spatial-count ontology-raw: 768 {f(e5_spatial_768['difference_condition_minus_baseline'])} {ci(e5_spatial_768)}; 3072 {f(e5_spatial_3072['difference_condition_minus_baseline'])} {ci(e5_spatial_3072)}. Semantic overall: 768 {f(e5_overall_768['difference_condition_minus_baseline'])} {ci(e5_overall_768)}; 3072 {f(e5_overall_3072['difference_condition_minus_baseline'])} {ci(e5_overall_3072)}.",
            "evidence_artifacts": "reports/generated/ontology_visibility_effect_v1.json; reports/E5_ONTOLOGY_VISIBILITY_CLOSEOUT.md; reports/E5_ONTOLOGY_PROVENANCE_V1.md",
            "scope_boundary": "The fixed visual key does not establish human semantic understanding or a pure semantic intervention independent of added image budget.",
        },
        {
            "claim_id": "C6",
            "status": "SUPPORTED_NEGATIVE_BOUNDARY",
            "manuscript_statement": "Actual visual-budget effects are model- and task-dependent; the corrected InternVL control does not replicate a universal high-budget gain.",
            "numerical_support": f"E4 high-low strict overall {f(e4_strict['difference_condition_minus_baseline'])} {ci(e4_strict)}; semantic overall {f(e4_semantic['difference_condition_minus_baseline'])} {ci(e4_semantic)}; semantic spatial-count {f(e4_spatial['difference_condition_minus_baseline'])} {ci(e4_spatial)} for actual 1 versus 7 tiles.",
            "evidence_artifacts": "reports/generated/internvl_tile_budget_v1.json; reports/INTERNVL_CORRECTED_REPLICATION_CLOSEOUT.md",
            "scope_boundary": "Neither legacy F3 nor E4 is a positive cross-family replication of Qwen.",
        },
        {
            "claim_id": "C7",
            "status": "DESCRIPTIVE_SENSITIVITY",
            "manuscript_statement": "Two pre-specified source-split seeds quantify how the Qwen high-low pattern varies across source partitions.",
            "numerical_support": seed_text,
            "evidence_artifacts": "reports/generated/source_seed_resolution_sensitivity_v1.json; data/manifests/source_seed29_resolution_v1.json; data/manifests/source_seed31_resolution_v1.json",
            "scope_boundary": "Seeds are reported separately and are not pooled, selected, or presented as independent training repetitions.",
        },
        {
            "claim_id": "C8",
            "status": "LIMITATION",
            "manuscript_statement": "No PID2Graph/OPEN100 external score is reported unless the official archive passes its size, MD5, and ZIP checks.",
            "numerical_support": f"F4 status: {f4.get('status')}; expected bytes {f4.get('expected_bytes')}; observed bytes {f4.get('observed_bytes')}; MD5 {f4.get('observed_md5')}.",
            "evidence_artifacts": "reports/generated/pid2graph_recheck_v1.json; reports/F4_EXTERNAL_SOURCE_STATUS_V1.md",
            "scope_boundary": "The external branch is not a blocker for the PIDQA audit and cannot support real-factory or OPEN100 generalization claims.",
        },
    ]

    summary = {
        "status": "pass",
        "protocol": {
            "primary_unit": "source_id",
            "interval": "paired source-cluster bootstrap 95% CI",
            "bootstrap_reps": 10000,
            "scoring": "strict and deterministic semantic scores reported side by side",
        },
        "retrieval_exposure": r1,
        "task_prior": prior_metrics,
        "e2_value_budget": e2_value,
        "e3_image_dependence": {"value_f1": e3_value, "structural_strict": e3_structural},
        "e4_actual_tile_budget": {"strict_overall": e4_strict, "semantic_overall": e4_semantic, "semantic_spatial_count": e4_spatial},
        "e5_ontology_visibility": {"spatial_768": e5_spatial_768, "spatial_3072": e5_spatial_3072, "overall_768": e5_overall_768, "overall_3072": e5_overall_3072},
        "e6_source_seed_sensitivity": e6_seeds,
        "f4_pid2graph": f4,
        "input_answer_isolation": input_isolation,
        "claims": claims,
    }
    output_dir = root / args.output_dir
    write_json(output_dir / "final_statistical_summary_v2.json", summary)
    write_csv(output_dir / "final_claim_evidence_matrix_v2.csv", claims)

    lines = [
        "# Paper Evidence-Strengthening Closeout",
        "",
        "Status: `COMPLETED` after E1--E6 and the one bounded F4 status recheck. This document is generated from machine-readable artifacts by `scripts/build_m0_evidence_freeze_v2.py`.",
        "",
        "## Frozen manuscript claim contract",
        "",
    ]
    for claim in claims:
        lines.extend(
            [
                f"### {claim['claim_id']} — {claim['status']}",
                "",
                claim["manuscript_statement"],
                "",
                f"**Numerical support:** {claim['numerical_support']}",
                "",
                f"**Evidence:** {claim['evidence_artifacts']}",
                "",
                f"**Boundary:** {claim['scope_boundary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Non-negotiable wording constraints",
            "",
            "- Do not claim a universal resolution benefit, a positive InternVL cross-family replication, generic corruption robustness, pure visual-encoding latency, energy efficiency, PID2Graph/OPEN100 generalization, real-factory deployment, or human-semantic P&ID understanding.",
            "- Keep strict and semantic scores distinct; raw outputs remain immutable.",
            "- Describe E6 as two pre-specified source-split sensitivities, not a selection exercise or independent model-training replication.",
        ]
    )
    closeout = root / args.closeout
    closeout.parent.mkdir(parents=True, exist_ok=True)
    closeout.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": str(output_dir / "final_statistical_summary_v2.json"), "claims": str(output_dir / "final_claim_evidence_matrix_v2.csv"), "closeout": str(closeout)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
