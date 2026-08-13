# Figure manifest

| Figure | PDF | PNG | Deterministic source |
|---|---|---|---|
| Figure 1 | paper/figures/figure_1_saber_pid_overview_v10.pdf | paper/figures/figure_1_saber_pid_overview_v10.png | Validated PIDQA, quality, closest-budget InternVL, DEXPI, and cost-rule reports |
| Figure 2 | paper/figures/figure_2_quality_and_budget_v10.pdf | paper/figures/figure_2_quality_and_budget_v10.png | Frozen 7,360-row quality matrix and 2,760-row InternVL 54-tile recovery |
| Figure 3 | paper/figures/figure_3_dexpi_external_v10.pdf | paper/figures/figure_3_dexpi_external_v10.png | Frozen 35-image DEXPI Qwen/OCR matrix and logical-case bootstrap contrasts |
| Figure 4 | paper/figures/figure_4_cost_aware_operation_v10.pdf | paper/figures/figure_4_cost_aware_operation_v10.png | Set B TP/FP/FN, operating workload, exact loss envelope, and 10,000 source-bootstrap decision probabilities |
| Figure S1 | paper/figures/figure_s1_boundary_controls_v10.pdf | paper/figures/figure_s1_boundary_controls_v10.png | Structural-task, initial InternVL, and runtime boundaries |
| Figure S2 | paper/figures/figure_s2_qualification_effects_v10.pdf | paper/figures/figure_s2_qualification_effects_v10.png | Complete primary paired counterfactual and matched-output-cap effects |
| Figure S3 | paper/figures/figure_s3_operating_modes_v10.pdf | paper/figures/figure_s3_operating_modes_v10.png | Complete operating family and overlap exclusions |
| Figure S4 | paper/figures/figure_s4_cross_model_replication_v10.pdf | paper/figures/figure_s4_cross_model_replication_v10.png | Complete native-budget 54-cell counterfactual matrix, including low InternVL magnitudes |
| Figure S5 | paper/figures/figure_s5_prompt_sensitivity_v10.pdf | paper/figures/figure_s5_prompt_sensitivity_v10.png | Frozen P1-minus-P0 sensitivity for every model/subset pair |

All figures are produced by `scripts/build_rineng_v10_publication_figures.py` from validated machine-readable reports. No figure uses a generative image model, answer-guided crop, or hand-drawn numerical annotation. The V10 metadata records source hashes and final asset hashes; the V10 layout audit checks every rendered text bounding box for clipping and unintended overlap.
