# START Package 01 — Source-Isolated, Graph-Grounded Reliability Evaluation for P&IDs

> **Active submission route (2026-08-12):** the validated V7 replication layer remains the baseline, and the V8 high-value evidence closeout is executing under `22_RINENG_V8_HIGH_VALUE_EVIDENCE_AND_SUBMISSION_CLOSEOUT.md`. V8 adds a cost-sensitive operating rule, paired 3072/512 quality robustness, a closest-safe 54-tile InternVL boundary, and a second public DEXPI family. Maintenance-safe archives and resume instructions are recorded in `reports/RINENG_V8_PUBLIC_BACKUP_STATUS.md` and `reports/RINENG_V8_H200_MAINTENANCE_RESUME.md`.

**Frozen planning date:** 2026-08-04  
**Target journal:** *Results in Engineering*  
**Primary recommendation:** Experimental Plan B  
**Fallback:** Plan A  
**High-risk extension:** Plan C only after Plan B quality gates are satisfied

## Project mission

This package defines an implementation-ready research program for evaluating and improving the reliability of vision–language models on piping and instrumentation diagrams (P&IDs). The project is centered on five non-negotiable principles:

1. **Source-level isolation:** all questions, crops, renders, and derivatives from one source artifact remain in one split.
2. **Answer isolation:** hidden graph truth, executable queries, and outcome labels never enter the model-facing pipeline.
3. **Objective evaluation:** central claims use deterministic graph, path, attribute, evidence, and abstention metrics rather than unvalidated LLM judging.
4. **Engineering failure analysis:** topology, flow-direction, connectivity, missing-information, and evidence failures are reported separately.
5. **Reproducibility:** every run is tied to immutable data, split, model, prompt, environment, and output hashes.

## Primary paper concept

> A source-isolated counterfactual study showing when high-detail vision--language processing reads source-specific P&ID tags, and how transparent OCR--VLM combinations provide coverage- and precision-oriented candidate-tag operating modes.

The manuscript presents **SABER-PID**---Source-isolated, Answer-hidden,
Budget-recorded, Evidence-counterfactual, Reported-by-task---as a demonstrated
evaluation instrument rather than a universal benchmark protocol. The frozen
V7 extension contains 54 complete cells and 16,560 predictions across three
models, three pairwise source-disjoint PIDQA subsets, two prompts, and three
evidence conditions. All 36 correct-minus-control value-tag F1 intervals have
positive lower bounds. Revalidate the recovered outputs and rebuild the V7
tables, figures, and manifests without new model inference with:

```text
python scripts/reproduce_rineng_overnight_v7.py --root .
```

## Recommended execution path

1. Complete the license, provenance, integrity, and source-group audit.
2. Reproduce direct-answer and image-to-graph baselines.
3. Run the decisive random-QA-split versus source-isolated-split pilot.
4. Continue to Plan B only if the pilot produces a coherent diagnostic signal.
5. Freeze calibration thresholds, prompts, task definitions, and severity policies.
6. Run the locked final evaluation once.
7. Generate all paper artifacts from immutable run manifests.

## Package map

| File | Purpose |
|---|---|
| `01_PROJECT_DEFINITION.md` | Thesis, research questions, hypotheses, contributions, exclusions |
| `02_RELATED_WORK_AND_POSITIONING.md` | Literature map and defensible novelty boundary |
| `03_DATA_BENCHMARK_AND_LICENSES.md` | Dataset roles, split rules, leakage controls, licenses |
| `04_METHOD_SPECIFICATION.md` | Formal problem, architecture, algorithm, pseudocode |
| `05_EXPERIMENT_PLAN_A_CONSERVATIVE.md` | Lower-risk benchmark and audit paper |
| `06_EXPERIMENT_PLAN_B_RECOMMENDED.md` | Recommended benchmark-plus-reliability-system paper |
| `07_EXPERIMENT_PLAN_C_HIGH_NOVELTY.md` | Synthetic-to-real structured fine-tuning extension |
| `08_METRICS_STATISTICS_ABLATIONS.md` | Metrics, statistical design, robustness, ablations |
| `09_MULTI_GPU_EXECUTION.md` | 8×H200 scheduling, serving, caching, recovery |
| `10_REPOSITORY_AND_MODULE_CONTRACTS.md` | Repository tree, module contracts, configs, CLI, tests |
| `11_CODEX_BACKLOG.md` | Dependency-aware implementation tasks for Codex |
| `12_EXPERIMENT_GOVERNANCE.md` | Manifests, quality gates, frozen-evaluation rules |
| `13_PAPER_BLUEPRINT.md` | Manuscript outline, abstract blueprint, evidence map |
| `14_REVIEWER_RISK_REGISTER.md` | Anticipated criticism and prevention strategy |
| `15_STARTUP_SEQUENCE.md` | First ten tasks, first decisive experiment, MVP/full paper |
| `16_REFERENCES.md` | Primary papers, repositories, model cards, specifications |
| `18_SUBMISSION_COMPLETION_EXPERIMENT_MASTER_PLAN.md` | Historical completion route for the first final experiment package |
| `19_PAPER_EVIDENCE_STRENGTHENING_AND_SUBMISSION_EXECUTION_CHARTER.md` | Historical corrected-evidence and submission route completed before v4 |
| `20_POSITIVE_NARRATIVE_SELF_REVIEW_AND_REVISION_CHARTER.md` | Historical E7/E8 and positive-narrative route completed before v4 |
| `21_RINENG_OVERNIGHT_HIGH_VALUE_GPU_EXECUTION_V7.md` | Frozen V7 three-model counterfactual execution plan |
| `review/SABER_PID_model_based_editorial_review.md` | Supplied model-based editorial critique that motivated the v4 revision |
| `reports/MODEL_BASED_EDITORIAL_REVISION_CLOSEOUT_V4.md` | Frozen item-by-item v4 review resolution and evidence boundary |
| `reports/POSITIVE_NARRATIVE_SELF_REVIEW_AND_JOURNAL_STRATEGY_V5.md` | Active title, narrative, evidence hierarchy, figure plan, experiment decisions, and journal ladder |
| `reports/POSITIVE_NARRATIVE_REVISION_CLOSEOUT_V5.md` | Machine-generated v5 implementation closeout |
| `reports/RINENG_OVERNIGHT_V7_CLOSEOUT.md` | Completed V7 execution, evidence, validation, and claim boundary |
| `reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.json` | Local recovery and SHA-256 inventory for 118 artifacts |
| `scripts/reproduce_submission_v5.py` | One-command, inference-free rebuild of scores, figures, TeX, PDFs, and validation records |
| `scripts/reproduce_rineng_overnight_v7.py` | Independent V7 recomputation and paper-artifact rebuild without inference |
| `templates/` | Run, decision, results-ledger, and dataset-card templates |

## Scope freeze

The initial project does **not** claim:

- the first P&ID question-answering dataset;
- the first P&ID graph dataset;
- the first engineering-drawing VLM benchmark;
- the first topology-preserving synthetic P&ID generator;
- industrial deployment safety;
- universal severity weights;
- real-plant validation without public real-plant evidence.

The project can claim novelty only where the evidence supports source isolation, answer isolation, unified graph-grounded reliability evaluation, selective prediction, and reproducible engineering diagnostics.
