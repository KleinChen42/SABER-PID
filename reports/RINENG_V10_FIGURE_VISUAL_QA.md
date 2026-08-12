# RINENG V10 figure visual-QA record

Status: **PASS**  
Scope: all four main-text figures and all five supplementary figures  
Review method: two-pass direct visual inspection at original PNG resolution, renderer-level text-bounding-box audit, and final embedded-PDF page inspection.

## Fixed publication-style contract

- Final design width: 7.15 in (approximately the Results in Engineering double-column width).
- Typeface: one sans-serif family throughout; 8 pt normal text and 8.5 pt panel headings at final size.
- Vector-first delivery: PDF is the manuscript asset; 600 dpi PNG is retained for review and fallback.
- Color encoding: Okabe--Ito-derived color-blind-safe palette, reinforced by marker shape or line role.
- Layout: lowercase panel labels, concise panel headings, no redundant figure-level title inside the artwork, light reference grids, restrained borders, and consistent line weights.
- Validation: every active figure must have zero unintended text-box collisions and zero figure-boundary clipping before it is accepted.

The style contract follows Elsevier's artwork and sizing guidance and is cross-checked against Nature's research-figure guidance for final-size typography, compact panels, vector export, and accessible visual encoding.

## Figure-by-figure disposition

| Figure | V9 / legacy issue found | V10 action | Final status |
|---|---|---|---|
| Figure 1 | Card-heavy slide aesthetic, oversized text blocks, weak visual hierarchy, and an initially distorted step-node shape | Rebuilt as a compact three-panel qualification-to-operation diagram; replaced coordinate-space circles with display-space circular markers; separated pipeline, evidence, and cost rule | PASS |
| Figure 2 | Panel-C tensor-budget text collided with the left axis region; several labels were too close to plot boundaries | Rebuilt as aligned interval panels with concise two-line labels, explicit ticks, and increased left margin | PASS |
| Figure 3 | Redundant artwork title and a crowded metric-group display | Rebuilt as a dot-and-line operating-point panel plus a logical-case contrast panel; retained all conditions and intervals | PASS |
| Figure 4 | Oversized title, crowded point labels, and overlapping legend/title regions | Rebuilt as a precision--recall panel, exact lower-loss envelope, and decision-stability panel; moved the Qwen label away from the lower boundary and separated titles from legends | PASS |
| Figure S1 | Dense multi-purpose layout and hard-to-parse ungrouped input counts | Rebuilt as two aligned forest plots and a compact zebra table; added thousands separators and distinct markers for control type | PASS |
| Figure S2 | Large legacy titles and excessive annotation scale | Rebuilt as two compact qualification forest plots with common zero-reference grammar | PASS |
| Figure S3 | Oversized labels, lower-boundary label crowding, and an unnecessary workload footer | Rebuilt as a clean precision--coverage panel and a frozen-rule exclusion forest plot; removed duplicated workload content and repositioned Qwen | PASS |
| Figure S4 | Very dense 54-cell matrix with weak row/column hierarchy | Rebuilt as a 3-by-2 small-multiple matrix; model rows, counterfactual columns, subset color, and prompt shape are encoded consistently | PASS |
| Figure S5 | Excessive unused space and weak group separation | Rebuilt as one grouped interval plot with model blocks separated by subtle rules | PASS |

## Machine-verifiable acceptance evidence

- Builder report: `reports/generated/rineng_v10_publication_figures.json`
- Source and asset hashes: `paper/figures/figure_metadata_v10.json`
- Text layout audit: `reports/generated/rineng_figure_layout_audit_v10.json`
- Embedded-page audit: `reports/generated/pdf_render_validation_v10.json` and `reports/generated/pdf_visual_inspection_v10.json`
- Required audit outcome: 9 figures, 0 clipped text boxes, 0 unintended text collisions.
- Font outcome: all nine vector figures and both compiled submission PDFs report zero non-embedded fonts.
- Rebuild command: `python scripts/build_rineng_v10_publication_figures.py --root .`

Legacy assets remain in the repository as provenance, but the manuscript and supplement reference only the V10 figure set.
