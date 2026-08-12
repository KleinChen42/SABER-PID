# RINENG overnight high-value GPU execution v7

Date frozen: 2026-08-12

Status: `COMPLETE -- INDEPENDENTLY VALIDATED`

Machine-readable plan: `data/manifests/rineng_overnight_v7_public_plan.json`

Frozen plan SHA-256: `2042505735da31f51d98a12fe8edebc01858e525f9f5059282dd01fd1b13b799`

## Purpose

Use otherwise idle H200 capacity for evidence that materially answers likely
reviewer objections without reopening the paper into a broad tuning study. The
matrix tests whether requested-drawing dependence survives model scale, model
family, prompt wording, and two additional source-disjoint PIDQA subsets.

This is deliberately not an intermediate-budget sweep, learned fusion search,
best-prompt selection, or new-data download.

## Experiment matrix

| GPU | Model | Scientific role | Rows |
|---:|---|---|---:|
| 0 | Qwen3-VL-32B-Instruct | Model-scale counterfactual and prompt robustness | 5,520 |
| 1 | InternVL3.5-8B | Independent model-family boundary and prompt robustness | 5,520 |
| 2 | Qwen3-VL-8B-Instruct | Same-protocol anchor for the qualified model | 5,520 |

Each model evaluates:

- three source sets: Set B (100 sources / 400 questions), seed-29 strict
  subset (65 / 260), and seed-31 strict subset (65 / 260);
- zero source overlap among the three sets;
- two pre-existing F2 prompts (`p0`, `p1`), fixed before V7 inference;
- correct, source-shuffled, and text-only conditions;
- all four PIDQA tasks;
- greedy decoding and a common 512-token output cap;
- Qwen maximum image side 3072; InternVL dynamic-preprocess maximum 12 tiles.

Total inference rows: 16,560.

## Evidence hierarchy

Primary contrasts are correct-minus-shuffled and correct-minus-text-only strict
value-tag F1 within every model, dataset, and prompt. Intervals use paired
10,000-replicate source bootstrap resampling.

Secondary outputs are task-wise strict accuracy, source-macro accuracy,
`p1-minus-p0` sensitivity with both prompts retained, latency, and directional
model-family/model-scale comparisons. No prompt is selected after seeing the
answers.

Positive results strengthen within-PIDQA operating-point transport. Null or
negative results remain model-, prompt-, or subset-specific boundaries. All
drawings remain synthetic PIDQA drawings, so even uniformly positive results
cannot establish real-plant or external-family transport.

## Isolation and reproducibility

Inference manifests are projected onto an allowlist and contain no field whose
name includes `answer` or `cypher`. Inference runners never open
`pidqa_records.jsonl`; the scorer-only reference is opened only after inference
has stopped.

Every row records dataset, prompt hash, plan hash, condition, image source,
input budget, output cap, latency, output token count, device, and
`test_answer_used=false`. Outputs are atomically rewritten after each instance,
and completed rows are resumable with `--skip-existing`.

## Detached execution record

- Remote host: `hd03-gpu2-0002`
- Launch time: `2026-08-12T01:26:06+08:00`
- Qwen3-VL-8B completed: `2026-08-12T03:56:35+08:00`
- Qwen3-VL-32B completed: `2026-08-12T03:58:50+08:00`
- InternVL3.5-8B completed: `2026-08-12T04:07:11+08:00`
- Final CPU scorer completed: `2026-08-12T12:02:10+08:00`
- Assigned GPUs 0--2 were released after inference; GPUs 3--7 were never
  modified.

All three workers wrote `FINISHED` and `COMPLETE`. Each produced 18 cells and
5,520 rows with zero inference error. The combined matrix contains 54 JSONL
files and 16,560 rows.

The original detached watcher reached all three `FINISHED` markers but failed
at scoring because `scripts/run_e1_evidence_audit.py` had not been synchronized.
No raw output was affected. The missing dependency was transferred with a
matching SHA-256, its import was verified, and only the CPU scorer was rerun.
No GPU inference was repeated.

## Final results and artifacts

- Scorer: `pass`, 54 cells and 36 counterfactual comparisons.
- Independent recomputation: `pass`, zero errors, maximum numerical difference
  0.0 for both counterfactual and prompt-sensitivity quantities.
- All 36 pre-specified correct-minus-control value-tag F1 intervals have
  positive lower bounds.
- Eight of nine P1-minus-P0 intervals include zero; both prompts are retained.
- Qwen3-VL-8B correct-image value-tag F1: 0.5253--0.6587.
- Qwen3-VL-32B correct-image value-tag F1: 0.5697--0.7361.
- InternVL3.5-8B correct-image value-tag F1: 0.0090--0.0257.

Canonical outputs:

- `reports/generated/rineng_overnight_v7_score.json`
- `reports/generated/rineng_overnight_v7_score.csv`
- `reports/generated/rineng_overnight_v7_validation.json`
- `reports/RINENG_OVERNIGHT_V7_CLOSEOUT.md`
- `reports/RINENG_OVERNIGHT_V7_ARTIFACT_MANIFEST.json`

The raw outputs and remote logs are recovered locally. Raw payloads remain
ignored by Git; their per-file hashes and row counts are fixed in the artifact
manifest.
