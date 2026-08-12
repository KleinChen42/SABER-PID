# RINENG V8 high-value evidence and submission closeout charter

Status: active execution charter

Frozen date: 2026-08-12

Target: *Results in Engineering*

Decision default: **GO** for safe, answer-isolated, automatically verifiable work

## 1. Objective

Close the remaining high-value evidence gaps without changing the paper's
bounded claim. The main claim remains candidate-tag retrieval under declared
drawing-family, model, prompt, and visual-budget conditions. V8 adds an
executable cost rule, paired quality robustness, the closest safe non-Qwen
visual-budget comparison, and a second public P&ID family.

No optional human audit is a gate. All central checks use frozen manifests,
immutable raw outputs, deterministic scoring, grouped bootstrap uncertainty,
independent machine recomputation, and explicit limitations.

## 2. Cost-sensitive operating modes

Use existing Set B per-source TP/FP/FN values only; do not rerun inference.
For each transparent mode, compute

\[
L=C_{\mathrm{FN}}FN+C_{\mathrm{FP}}FP,
\qquad r=C_{\mathrm{FN}}/C_{\mathrm{FP}}.
\]

Report the exact lower envelope, adjacent switch ratios, 10,000 source-cluster
bootstrap minimum-loss probabilities, and switch-ratio intervals. Integrate a
decision figure and the exact piecewise rule into the paper and supplement.

## 3. Qualified-setting quality robustness

Retain Qwen3-VL-8B, prompt P0, 3072-side processing, a 512-token output cap,
and greedy decoding. Cross three pairwise source-disjoint PIDQA subsets (100,
65, and 65 drawings) with:

- clean;
- JPEG quality 70;
- Gaussian blur radius 1;
- 0.75-times downsample and restore.

Every condition has a source-shuffled partner. The frozen scope is 24 cells
and 7,360 predictions. The primary estimand is correct-minus-shuffled strict
value-tag F1. For each degradation, also report a paired difference-in-
differences relative to clean. Pooling is allowed because the 230 source IDs
are mutually disjoint; dataset-specific rows remain visible.

## 4. Closest-safe non-Qwen visual budget

Use InternVL3.5-8B with 54 non-overlapping 448-pixel tiles arranged as a
9-by-6 white-letterbox canvas, no thumbnail, and a 512-token output cap. The
realized input contains 32,514,048 tensor elements versus 35,979,264 for the
qualified Qwen processor, a 9.63% lower count. This is a closest-safe
tensor-element comparison, not encoder or information-throughput equivalence.

Run correct, source-shuffled, and no-image cells on the same three disjoint
subsets: nine cells and 2,760 predictions. Compare correct-minus-controls and
54-tile correct-image F1 against the earlier native-12-tile boundary. A fatal
CUDA peer-memory fault triggers a bounded fresh-process restart with valid
rows resumed by instance ID; non-accelerator failures are retained and
reported rather than hidden.

## 5. Second public P&ID family

Use the official DEXPI Public Example PIDs repository at frozen commit
`a23d61e2e089eb2ca464cd552f9ae580a2785963` under CC BY 4.0. Accept only exact
same-directory image/XML stem pairs for which a structured XML tag value also
appears in an XML graphic-text declaration. Remove duplicate image hashes,
cover logical test cases before vendor variants, and make no selection using
model outputs.

The frozen scope is 35 images, 26 logical test cases, and 65 prefix-conditioned
questions. Run Qwen correct, cross-case source-shuffled, and no-image cells at
3072-side/512 tokens plus the frozen full-image OCR family. Bootstrap over
logical test cases so vendor variants never count as independent drawings.
Describe these as public engineering exchange examples, not real-plant data.

PID2Graph/OPEN100 is audited separately. Its verified GraphML has structural
node/edge classes and bounding boxes but no explicit visible-text/tag field;
internal node IDs are not tag references. Therefore it receives no fabricated
tag-retrieval score and remains a documented structural resource.

## 6. Execution and maintenance recovery

Use only a confirmed idle project GPU and never terminate another user's
process. Formal outputs and logs write directly to:

`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812`

The verified full backup, V8 snapshots, latest code overlay, hashes, and resume
instructions are recorded in `reports/RINENG_V8_PUBLIC_BACKUP_STATUS.md` and
`reports/RINENG_V8_H200_MAINTENANCE_RESUME.md`. All inference runners resume
valid rows and skip complete cells.

## 7. Scoring, independent validation, and reporting

After inference:

1. copy all raw public-disk outputs to the local canonical paths;
2. verify cell membership, duplicate IDs, status, frozen plan hashes, answer-
   isolation flags, tensor-element budgets, file hashes, and expected row counts;
3. compute TP/FP/FN, precision, recall, strict tag F1, exact-set accuracy, and
   source/logical-group uncertainty;
4. run an independent validator that imports neither V8 scorer and uses
   different SHA-256-derived 10,000-replicate bootstrap seeds;
5. generate tables and figures only from passing score reports;
6. align abstract, results, discussion, limitations, conclusion, highlights,
   cover letter, captions, supplement, and availability statements with the
   validated evidence;
7. compile both PDFs, render and inspect every page, build a SHA-256 artifact
   inventory, build and validate the public release candidate, and commit only
   the intended V8 files.

## 8. Definition of done

V8 is complete when all safely automatable work above is finished; every
reported number traces to immutable raw output; independent validation passes;
figures and tables match their machine-readable sources; the manuscript claim
does not exceed the evidence; PDFs pass all-page visual inspection; the public
release archive passes membership, hash, CRC, timestamp, privacy, and raw-cell
scope checks; and the intended changes are in version control.
