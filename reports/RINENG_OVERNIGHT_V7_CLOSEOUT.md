# RINENG overnight V7 closeout

Status: **COMPLETE AND INDEPENDENTLY VALIDATED**  
Date: 2026-08-12  
Frozen plan SHA-256: `2042505735da31f51d98a12fe8edebc01858e525f9f5059282dd01fd1b13b799`

## Execution closeout

- Three models completed: Qwen3-VL-8B, Qwen3-VL-32B, and InternVL3.5-8B.
- The frozen matrix contains three pairwise source-disjoint PIDQA subsets, two pre-existing prompts, three evidence conditions, and all four PIDQA tasks.
- All 54 cells passed: 16,560/16,560 predictions, zero inference errors, zero duplicate IDs within cells, and zero true `test_answer_used` flags.
- The original score watcher failed only because `scripts/run_e1_evidence_audit.py` had not been synchronized. The dependency was SHA-256 matched, imported successfully, and the CPU-only scorer was rerun without repeating inference.
- Final scorer status is `pass`: 54 scored cells and 36 counterfactual comparisons.
- Independent validation recomputed every cell metric, comparison, source-bootstrap interval, prompt sensitivity, raw-output hash, and CSV value without importing the scorer. It passed with zero errors and maximum numerical disagreement 0.0.
- The recovered raw prediction payload contains 54 JSONL files, 16,560 rows, and 20,889,084 bytes. Runner summaries, control markers, and remote logs were also recovered.
- Manuscript and supplement PDFs compile without undefined references, overfull/underfull boxes, or layout errors. Model-based visual review passed all 23 rendered pages, with detailed 240-dpi inspection of the new main figure and supplementary V7 tables/figure.

## Primary evidence

All 36 pre-specified correct-minus-control value-tag F1 intervals have positive lower bounds.

| Model | Correct-image F1 range (median) | Correct minus shuffled range | Correct minus no image range | Positive intervals |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 0.5253--0.6587 (0.6229) | +0.5191--+0.6437 | +0.5253--+0.6587 | 12/12 |
| Qwen3-VL-32B | 0.5697--0.7361 (0.6512) | +0.5639--+0.7284 | +0.5697--+0.7361 | 12/12 |
| InternVL3.5-8B | 0.0090--0.0257 (0.0124) | +0.0090--+0.0239 | +0.0090--+0.0257 | 12/12 |

The requested-drawing direction therefore replicates across all tested models, subsets, and frozen prompts. Practically useful magnitude is concentrated in Qwen3-VL; the InternVL direction is statistically detected but too small to qualify comparable retrieval performance.

Eight of nine P1-minus-P0 intervals include zero. The single exception is Qwen3-VL-32B on seed 29, where P1 is lower by 0.0487 with interval [-0.1044, -0.0025]. Both prompts remain reported; no best-prompt selection was applied.

## Paper artifacts

- Main replication figure: `paper/figures/figure_v7_cross_model_counterfactual_replication.pdf`
- Prompt-sensitivity figure: `paper/figures/figure_s_v7_prompt_sensitivity.pdf`
- Full counterfactual table: `paper/tables/table_rineng_overnight_v7_counterfactual.tex`
- All-task boundary table: `paper/tables/table_rineng_overnight_v7_task_accuracy.tex`
- Machine-readable paper summary: `reports/generated/rineng_overnight_v7_paper_summary.json`
- Scorer output: `reports/generated/rineng_overnight_v7_score.json`
- Independent validation: `reports/generated/rineng_overnight_v7_validation.json`
- Complete SHA-256 inventory: `reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.json`
- Compiled manuscript: `output/pdf/v7/manuscript.pdf`
- Compiled supplement: `output/pdf/v7/supplementary.pdf`

The manuscript now presents the V7 matrix as cross-scale, cross-prompt, source-disjoint within-family replication. The full matrices remain visible in the supplement, including the small InternVL magnitude and task-level boundary.

## Reproduction

From the repository root, with recovered raw outputs present:

```text
python scripts/reproduce_rineng_overnight_v7.py --root .
```

This performs scoring-independent validation and rebuilds the V7 tables, figures, paper summary, and artifact manifest. It does not run GPU inference.

## Claim boundary

All three subsets belong to PIDQA. The V7 matrix strengthens requested-drawing dependence, Qwen scale stability, and prompt robustness within PIDQA; it does not establish transport to a different drawing family, topology reasoning, or field deployment.
