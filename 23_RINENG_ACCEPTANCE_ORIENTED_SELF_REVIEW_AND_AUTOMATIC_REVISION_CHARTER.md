# 23. Results in Engineering acceptance-oriented self-review and automatic revision charter

Status: **COMPLETED**

Frozen date: 2026-08-12

Target: *Results in Engineering* full research paper

Decision default: **GO** for all safe, evidence-preserving, automatically
verifiable editorial work

## 0. Executive editorial verdict

The V8 evidence package is scientifically sufficient for submission. The
remaining acceptance risk is primarily editorial: the manuscript currently
reads as a sequence of audits and boundary checks, while its strongest
contribution is an engineering qualification and operating-decision framework.
The amount of evidence is no longer the bottleneck. The manuscript needs a
major narrative compression and hierarchy reset, not another model zoo.

The recommended decision is:

> **Revise the story, consolidate the main figures, update the closest-work
> positioning, validate the complete package, and submit. Do not launch new
> GPU inference before first submission.**

The target identity is an engineering-methodology paper that answers two
practical questions:

1. When does a vision--language model's tag output demonstrably depend on the
   requested P&ID rather than source overlap, priors, or an inadequate visual
   budget?
2. Once qualified, which transparent OCR--VLM operating mode minimizes the
   expected cost of missed tags and false candidates?

## 1. Scientific-integrity boundary for positive narrative

Positive framing is legitimate when it changes emphasis, not evidence.

Allowed and required:

- lead with the strongest predeclared or frozen results;
- keep the main text focused on results that answer the central engineering
  questions;
- move complete secondary, null, and boundary analyses to the supplement;
- replace long defensive passages with concise, accurate scope statements;
- omit failed downloads, smoke runs, superseded configurations, and execution
  history from the submission narrative;
- use effect sizes, uncertainty, transfer, robustness, and engineering
  decisions as the visual and verbal center;
- preserve every raw output and full numerical report in the reproducibility
  release.

Not allowed:

- removing a result whose omission would materially change interpretation of
  the central claim;
- describing the DEXPI examples as real-plant or field data;
- describing tensor-element matching as architectural equivalence;
- implying that native-12 InternVL performance was strong;
- selecting prompts, subsets, or transformations because they scored well;
- converting a confidence interval containing zero into evidence of
  improvement or equivalence;
- presenting a model-based critique as human peer review.

The native-12 InternVL boundary, aggregate/prior behavior, prompt sensitivity,
overlap sensitivity, structural-task controls, and PID2Graph task-fit audit
must remain available in the supplement and release. They need not occupy the
title, abstract, highlights, main figures, or first-order Results structure.

## 2. Journal-fit diagnosis

The current *Results in Engineering* scope includes AI and ML Engineering but
requires novelty beyond applying an existing AI algorithm to a known problem.
The manuscript must therefore make its reusable engineering knowledge explicit:

1. **Qualification contract:** source isolation, answer-hidden inference,
   correct/shuffled/no-image interventions, recorded budgets, task-specific
   estimators, and grouped uncertainty form a falsifiable acceptance rule.
2. **Budget finding:** visual-input adequacy is experimentally shown to govern
   whether a model family supplies useful requested-drawing evidence.
3. **Transfer finding:** the qualification succeeds on a second licensed P&ID
   family and on a second model family after closest-safe budget matching.
4. **Decision finding:** complementary OCR and VLM errors become an exact,
   cost-sensitive operating rule rather than an informal recommendation.

This is not a leaderboard paper and not a new recognition architecture. Its
novelty is the linked qualification-to-decision workflow and the engineering
knowledge produced by applying it.

## 3. Selected paper identity

### 3.1 Selected title

> **SABER-PID: Source-Isolated Qualification and Cost-Aware Operation of
> Vision--Language Models for P&ID Tag Retrieval**

This title is preferred because it names the reusable method, identifies the
engineering output, and foregrounds both scientific qualification and
operational value. It avoids leading with a defensive word such as “audit” or
with a generic benchmark score.

### 3.2 One-sentence thesis

> SABER-PID turns raw P&ID model scores into an engineering decision by
> requiring requested-drawing counterfactual evidence at a declared visual
> budget, then selecting transparent OCR--VLM modes from explicit miss and
> false-candidate costs.

### 3.3 Four headline contributions

1. A source-isolated, answer-hidden, counterfactually controlled qualification
   contract for engineering-diagram tag retrieval.
2. Robust requested-drawing effects under mild image-quality shifts and across
   Qwen and closest-safe-budget InternVL inputs.
3. External transfer to visibility-qualified public DEXPI drawings with exact
   image/XML provenance and logical-case uncertainty.
4. A cost-sensitive lower envelope that converts precision--coverage tradeoffs
   into an executable operating-mode rule.

### 3.4 Supported and unsupported claims

Supported: candidate-tag retrieval under the evaluated PIDQA and public DEXPI
families, frozen prompts/models, declared visual budgets, and strict tag-set
scoring.

Unsupported: topology reconstruction, arbitrary P&ID question answering,
architectural equivalence, severe degradation robustness, real-plant
deployment, and autonomous engineering acceptance.

## 4. Result hierarchy after revision

| Evidence | Main text role | Supplement role | Decision |
|---|---|---|---|
| Correct vs shuffled/no-image PIDQA tag F1 | Headline qualification | Full counts and estimators | Lead |
| 3072 vs 768 at common cap | Qualification/budget finding | Runtime/cap diagnostics | Lead |
| Mild clean/JPEG/blur/downsample matrix | Robustness finding | Per-subset table | Lead |
| InternVL 54-tile correct vs controls | Cross-model budget transfer | All nine cells and native-12 comparison | Lead |
| DEXPI correct/shuffled/no-image/OCR | External-family transfer | Provenance and group-bootstrap detail | Lead |
| OCR/VLM union/intersection/fallback modes | Engineering operating points | Full family and workload | Lead |
| Cost-sensitive lower envelope | Engineering decision rule | Switch-ratio table and full grid | Lead |
| Source-split diagnostic | Motivation and one concise result | Five-seed details | Supporting |
| 54-cell Qwen/InternVL native-budget matrix | One scope sentence | Full figure/table | Supplement |
| Prompt sensitivity | None or one sentence | Full figure | Supplement |
| Union overlap-exclusion checks | One Discussion sentence | Full table/figure | Supplement |
| OCR join ablation/error taxonomy/gallery | One mechanism sentence | Full tables/gallery | Supplement |
| Aggregate structural-task boundaries | One scope sentence | Full task table/figure | Supplement |
| PID2Graph/OPEN100 task-fit audit | One limitation sentence | Full provenance audit | Supplement |

No result is deleted from the evidence chain. Main-text space is allocated by
relevance to the paper's two engineering questions.

## 5. Revised manuscript architecture

### 5.1 Abstract

Use 220--245 words and the order problem -> method -> four strongest results ->
engineering conclusion. Remove the detailed 54-cell/native-budget inventory
from the abstract. Retain these numerical anchors:

- PIDQA correct/shuffled/no-image F1;
- matched-budget InternVL effect;
- DEXPI correct and control F1;
- cost-aware operating choice or union/intersection endpoints;
- mild-quality effect range.

The last sentence must state positive utility first and scope second.

### 5.2 Introduction

Open with the engineering need for reliable tag candidate retrieval and the
cost of unqualified model output. Do not open Results-level narration with the
aggregate failure. Establish the evaluation gap, then introduce SABER-PID as
the solution. End with four contributions matching Section 3.3.

### 5.3 Related work

Organize by three routes:

1. image digitization and tag/text extraction;
2. graph/DEXPI/LLM retrieval from already structured representations;
3. document-VQA grounding, counterfactual controls, and grouped evaluation.

Add and accurately distinguish the following closest work:

- Alimin et al. (2025), *Talking like Piping and Instrumentation Diagrams*,
  arXiv:2502.18928;
- Alimin and Schweidtmann (2026), *GraphRAG for Engineering Diagrams:
  ChatP&ID Enables LLM Interaction with P&IDs*, arXiv:2603.22528;
- Zhu et al. (2026), *From P&ID Drawings to Process Graphs: A Multimodal
  Language Model Approach*, arXiv:2607.19568.

The distinction is that these works target structured graph interaction or
process-graph construction, whereas SABER-PID asks whether raw-image tag
retrieval is source-grounded and operationally qualified.

### 5.4 Methods

Compress to five subsections:

1. data families, source units, and answer isolation;
2. SABER-PID qualification contract and frozen conditions;
3. visual-budget, quality, and external-transfer matrices;
4. OCR--VLM operating modes and cost loss;
5. estimators, grouped uncertainty, and reproducibility.

Move the eight-row evidence-phase table to the supplement. Preserve exact
counts, conditions, estimators, and grouping units.

### 5.5 Results

Use five subsections in this order:

1. **SABER-PID qualifies requested-drawing tag retrieval on unseen PIDQA
   sources.** Include source-split motivation, correct/shuffled/no-image, and
   matched-cap visual-budget evidence.
2. **Qualified visual budgets sustain grounding across quality shifts and
   model families.** Combine mild-quality and 54-tile InternVL evidence.
3. **The qualification transfers to a second public P&ID family.** Lead with
   DEXPI F1, exact accuracy, and zero controls.
4. **Complementary OCR--VLM errors enable cost-aware operating modes.** Present
   operating points, workload, and exact cost rule together.
5. **Sensitivity analyses preserve the bounded interpretation.** Summarize
   native-budget, prompt, overlap, structural, and task-fit boundaries in one
   short subsection, with full results in the supplement.

### 5.6 Discussion

Use three subsections:

1. engineering finding and novelty;
2. deployment/benchmark design implications;
3. transportability and limitations.

Avoid repeating every numerical result. Interpret why budget matching,
source isolation, external transfer, and the loss rule matter for engineering
systems.

### 5.7 Limitations and conclusion

Limitations should be two compact paragraphs covering family diversity,
budgets/models, degradation range, OCR specificity, and deployment boundary.
Move byte-range and archive implementation detail to the supplement.

Conclusion should be one strong paragraph: qualification -> transfer ->
decision rule -> bounded application.

## 6. Main-figure and table redesign

### 6.1 Main figures

Target four main figures:

1. **Qualification-to-decision overview.** Replace the outdated defensive
   Figure 1. Show the four-stage SABER-PID path and four passed evidence cards:
   PIDQA requested-drawing effect, mild-quality stability, matched-budget
   InternVL transfer, and DEXPI transfer. End with cost-aware operating modes.
   Remove the obsolete statement that cross-model qualification failed.
2. **Qualified budgets across quality and model families.** Retain the current
   three-panel V8 figure with clearer compact labels.
3. **Second-family DEXPI transfer.** Retain the current DEXPI figure.
4. **Cost-aware OCR--VLM operation.** Consolidate the precision--recall
   operating-point panel with the deterministic cost envelope and bootstrap
   decision-stability panel. Move overlap-exclusion intervals to the
   supplement.

Move the full 54-cell native-budget cross-model matrix, prompt sensitivity,
source-exclusion stability, structural controls, and error gallery to
supplementary figures.

### 6.2 Main tables

Target two main tables:

1. a compact qualification scorecard across PIDQA Qwen, quality robustness,
   matched-budget InternVL, and DEXPI;
2. operating modes with TP/FP/FN, precision, recall, F1, and median candidate
   workload.

Move the evidence-phase table, per-quality detail, external full table, and
overlap-exclusion table to the supplement. Avoid duplicating the same numbers
in prose, table, and figure.

### 6.3 Optional graphical abstract

Do not create a separate graphical abstract unless the submission system asks
for one. The deterministic overview figure can be exported separately if
needed. No generative image model will be used for any submission figure.

## 7. Experiment decision

### 7.1 No new GPU experiment before submission

The three previously highest-value gaps are now closed: a second public P&ID
family, qualified-setting quality robustness, and a budget-comparable non-Qwen
test. Additional models or severity sweeps would dilute the frozen story and
increase multiplicity without addressing the current desk-review risk.

### 7.2 Allowed inference-free additions

- Build the consolidated qualification scorecard from passing JSON reports.
- Build the consolidated operating-decision figure from existing per-source
  TP/FP/FN and cost-grid artifacts.
- Recompute all displayed values and figure hashes.
- Add the three recent closest-work references from authoritative primary
  records.

### 7.3 Reviewer-response reserve only

Hold real/scanned proprietary drawings, severe-degradation sweeps, additional
model families, topology tasks, and learned fusion for a concrete reviewer
request or a separate follow-up study.

## 8. Section-level revision register

| Priority | Location | Current weakness | Required action |
|---:|---|---|---|
| P0 | Title | Accurate but procedural and low-energy | Adopt selected SABER-PID title |
| P0 | Abstract | Too many configuration details; strongest story arrives late | Rewrite around four outcomes |
| P0 | Figure 1 | Outdated cross-model boundary and defensive framing | Replace with current V8 qualification-to-decision overview |
| P0 | Results | Eleven subsections fragment the story | Consolidate to five subsections |
| P0 | Related work | Misses three direct 2025--2026 P&ID LLM/VLM papers | Add and distinguish them |
| P0 | Main figures | Seven figures compete for attention | Reduce to four; move boundaries to supplement |
| P0 | Discussion | Repeats results and foregrounds caveats | Interpret reusable engineering knowledge |
| P1 | Methods | Evidence-phase detail interrupts flow | Move phase table to supplement and compress |
| P1 | Tables | Detail is duplicated across prose and figures | Use scorecard + operating table only |
| P1 | Limitations | Archive/task-fit detail is too prominent | Compress and transfer detail to supplement |
| P1 | Conclusion | Exhaustive inventory weakens final message | One focused qualification-to-operation paragraph |
| P1 | Cover letter | Strong but long and configuration-heavy | Rewrite for editor-facing novelty and fit |
| P1 | Highlights | Good results but no cost-aware framework identity | Align with four contributions |
| P1 | AI disclosure | Does not exactly follow current Elsevier model wording | Align tools, purposes, author review, responsibility |
| P2 | Supplement | Current ordering follows history, not reader questions | Reorder to mirror main claim and preserve boundaries |
| P2 | Release | V8 release predates this editorial revision | Rebuild manifests, PDFs, and deterministic ZIP |

## 9. Acceptance-risk register after revision

| Risk | Residual level | Mitigation |
|---|---|---|
| “Only applies existing VLMs” desk rejection | Medium -> Low | Foreground reusable qualification and decision methods |
| Synthetic-family overgeneralization | Medium | DEXPI transfer plus explicit field boundary |
| Cherry-picking/negative-result suppression | Low | Full supplement, frozen matrices, immutable release |
| Cross-model comparability | Medium | Closest-safe label and no architecture-equivalence claim |
| Small DEXPI sample | Medium | Logical-case bootstrap and exact provenance |
| Too many controls/poor readability | High -> Low | Four figures, two tables, five Results subsections |
| Outdated closest-work positioning | High -> Low | Add 2025--2026 direct P&ID LLM/VLM work |
| Reproducibility mismatch after editing | Low | Full inference-free rebuild and SHA-256 release validation |

## 10. Automated execution order

1. Freeze this charter and selected title/story.
2. Add and verify closest-work references.
3. Generate the new overview and consolidated operating-decision figures from
   passing machine-readable evidence.
4. Rewrite manuscript title, abstract, Introduction, Related Work, Methods,
   Results, Discussion, Limitations, and Conclusion.
5. Reorder and update the supplement so every demoted result remains complete.
6. Update highlights, title page, cover letter, captions, figure manifest,
   data availability, and AI disclosure.
7. Recompute tables/figures and run independent metric agreement checks.
8. Compile both PDFs; render and visually inspect every page at final layout.
9. Rebuild artifact inventory and deterministic public release; validate
   hashes, CRC, timestamps, membership, local-path privacy, and raw scope.
10. Run the full test suite and submission validators.
11. Commit only intended files and write a final public-disk recovery bundle.

## 11. Definition of done

The revision is complete only when:

- the selected title and central thesis are consistent across all files;
- main-text results contain four central findings and no material
  contradiction is absent from the supplement/release;
- recent closest work is accurately cited and differentiated;
- main figures are reduced to four and main tables to two;
- every displayed value is generated from a passing machine-readable report;
- abstract, keywords, highlights, references, source inputs, and declarations
  pass the RINENG submission validator;
- the latest manuscript and supplement PDFs are rendered and every page is
  visually inspected with zero clipping, overlap, missing content, or
  unintended blank pages;
- the deterministic public release passes its integrity and privacy checks;
- the intended revision is committed and backed up to the public disk;
- only submitter-owned administrative placeholders remain: author names,
  affiliations, CRediT roles, funding, competing interests, and public DOI/URL.

## 12. Execution closeout

Completed: 2026-08-12

All automatable work in this charter is complete:

- selected title adopted across manuscript, supplement, cover letter, title
  page, highlights, declarations, and citation metadata;
- abstract reduced to 230 words and Results reorganized around requested-
  drawing qualification, quality/model transfer, DEXPI transfer, and
  cost-aware operation;
- recent direct 2025--2026 P\&ID language-model work added and differentiated;
- main-text hierarchy reduced to four figures and two tables;
- complete native-budget, prompt, structural-task, overlap, and task-fit
  boundary evidence retained in the 16-page supplement and public release;
- all V7/V8 scores and 10,000-replicate intervals reproduced without
  inference; all 53 repository tests passed;
- both final PDFs compiled without overfull/underfull boxes or unresolved
  citations/references, and all 13 manuscript plus 16 supplementary pages
  were visually inspected;
- the V9 submission validator and public-release privacy/integrity validator
  both passed;
- the deterministic public archive contains 272 inventoried project artifacts,
  including 91 immutable prediction files and 26,940 raw rows; unlike V8, it
  includes all 54 raw V7 cells;
- public archive size is 8,200,491 bytes with SHA-256
  `39f61f3b122e9d60633f6c681fcd7594767609c69e5aa801576954b2dead3493`;
- the identical archive hash was verified after backup to
  `/kwkj-k8s/hera_pid_reliability_backups/active_v9_20260812/submission/`.

The only unresolved items require submitter-owned facts or external account
actions: author and affiliation metadata, CRediT roles, funding and competing-
interest declarations, the public repository DOI/URL, and the journal-system
submission click. No new GPU inference is recommended before first submission.
