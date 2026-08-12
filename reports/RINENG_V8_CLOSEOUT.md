# RINENG V8 high-value evidence and submission closeout

Status: **PASS — experiment, evidence, manuscript, and reproducibility route complete**

Date: 2026-08-12

Target: *Results in Engineering*

## Scope completed

The frozen V8 route completed all four requested evidence extensions without
changing the bounded candidate-tag-retrieval claim:

1. an exact cost-sensitive lower envelope derived from the existing Set B
   TP/FP/FN vectors;
2. 3072-side/512-token Qwen quality robustness for clean, JPEG Q70, radius-1
   blur, and 0.75-times downsample-and-restore inputs, each paired with a
   source-shuffled control across three mutually source-disjoint subsets;
3. a closest-safe 54-tile/512-token InternVL3.5-8B comparison on the same 230
   disjoint sources; and
4. a second licensed public P&ID family from 35 DEXPI images, 26 logical test
   cases, and 65 prefix-conditioned questions, with correct, shuffled,
   no-image, and frozen full-image OCR conditions.

The immutable V8 inference scope contains 37 raw files and 10,380 prediction
rows. Every row passed status, membership, plan-hash, duplicate-ID, and
answer-isolation checks. PID2Graph/OPEN100 was automatically audited and
retained as a structural resource because its verified GraphML does not expose
an explicit visible-text/tag reference; no score was fabricated.

## Headline results

- The Set B decision rule selects intersection for
  `0 <= C_FN/C_FP < 0.5283`, joined OCR for `0.5283 < r < 0.6250`, OCR-first
  for `0.6250 < r < 2.0143`, and union for `r > 2.0143`; adjacent modes tie at
  the finite switch points.
- Across 230 disjoint PIDQA sources, clean/JPEG/blur/downsample correct-minus-
  shuffled F1 effects are respectively `+0.5578`, `+0.5722`, `+0.5754`, and
  `+0.5482`. All four 95% paired source-bootstrap intervals exclude zero;
  every degradation-minus-clean interval includes zero at the declared mild
  severity.
- The matched-budget InternVL pass reaches pooled correct-image F1 `0.4846`
  and a correct-minus-shuffled effect of `+0.4715` with 95% interval
  `[0.4064, 0.5332]`. Its 54-tile-minus-native-12 gain is `+0.4681`
  `[0.4043, 0.5290]`.
- On DEXPI, correct-image Qwen reaches precision `0.9231`, recall `0.8889`,
  F1 `0.9057`, and exact-set accuracy `0.8615`; shuffled and no-image F1 are
  both zero. Full-image OCR reaches F1 `0.7424`.

## Validation and publication artifacts

- Independent recomputation imports neither V8 scorer, uses independently
  derived bootstrap seeds, reports zero errors, a maximum point discrepancy of
  `4.34e-17`, and a maximum CI-endpoint discrepancy of `0.00609`.
- The inference-free reproduction chain rescored all outputs, rebuilt the V8
  tables and figures, ran independent validation, and passed its test suite.
- Submission validation passes: 247-word abstract, six keywords, five
  highlights, 19 cited/19 bibliography entries, no undefined citation or
  reference, and no missing required artifact.
- The 19-page manuscript and 13-page supplement have zero overfull or
  underfull box warnings. All 32 rendered pages were inspected at original
  resolution and passed for legibility, clipping, overlap, and completeness.
- Final PDF SHA-256 values are
  `cc508ebfc7ae6a48344dec9bb5572201f63047e2dd7baa6f0db3b524a01c2302`
  (manuscript) and
  `87cf3570291f879451486bc0d263caea7e5dc2b66541dd3c32d1a5180d62578d`
  (supplement).

The authoritative machine-readable inputs are
`reports/generated/rineng_v8_extension_score.json`,
`reports/generated/rineng_v8_dexpi_external_score.json`, and
`reports/generated/rineng_v8_independent_validation.json`. The final artifact
inventory and public-release validation are built after this closeout record
and must retain `status: pass`.

## Recovery and remaining submitter fields

The pre-maintenance full project archive is stored under
`/kwkj-k8s/hera_pid_reliability_backups`, and the public V8 run root retains
raw cells, logs, checkpoints, overlays, and hashes. Exact recovery identifiers
are recorded in `reports/RINENG_V8_PUBLIC_BACKUP_STATUS.md`.

No further experiment or optional human audit is a submission gate. Before the
journal upload, the submitter must supply author names and affiliations,
funding and competing-interest declarations, and the public archive DOI/URL.
Those are ownership-dependent administrative fields and cannot be truthfully
invented by the automated workflow.
