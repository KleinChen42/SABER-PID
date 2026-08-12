# Cover letter

Dear Editor,

Please consider our manuscript, **Qualifying Image-Grounded Tag Retrieval in Piping and Instrumentation Diagrams with Source-Isolated Counterfactual Evaluation**, for publication as a Research Article in *Results in Engineering*.

The paper addresses an engineering deployment question: when can candidate tags produced from a piping and instrumentation diagram be treated as image-grounded retrieval rather than as an aggregate benchmark effect driven by source repetition or task priors? We introduce SABER-PID, a source-isolated, answer-hidden, budget-recorded, and counterfactual evaluation workflow, and apply it to 100 unseen PIDQA source drawings.

The study produces a clear qualification decision. Frozen Qwen3-VL-8B at the high-detail operating point reaches pooled tag F1 0.5549 and source-macro F1 0.5810. Source-shuffled and no-image controls reduce pooled F1 to 0.0062 and 0.0000, while a common-output-cap comparison isolates a +0.5250 high-minus-low visual-input-budget effect. Aggregate strict accuracy would not reveal this capability because task-prior and no-image diagnostics exceed the correct-image aggregate score.

A frozen 54-cell extension crosses Qwen3-VL-8B, Qwen3-VL-32B, and InternVL3.5-8B with three pairwise source-disjoint PIDQA subsets, two pre-existing prompts, and correct, source-shuffled, and no-image conditions. All 36 correct-minus-control value-tag F1 intervals have positive lower bounds. Median correct-image F1 is 0.6229 and 0.6512 for the two Qwen scales, while InternVL remains at 0.0124; the paper therefore demonstrates broad directional replication without overstating model-invariant magnitude.

The paper also translates the evidence into practical retrieval modes. A frozen full-image OCR comparator and Qwen recover complementary tags. Prediction-only union reaches recall 0.7040 and pooled F1 0.6339; intersection reaches precision 0.9907 with a median of one candidate per drawing. TP/FP/FN counts, source-macro sensitivity, per-drawing workload, deterministic OCR joining, error taxonomy, and overlap-audited frozen-rule checks are all reported. The latter retain positive point estimates after Set B and mutual-source exclusions, while their asymmetric uncertainty is stated explicitly.

This combination of engineering information retrieval, controlled multimodal evaluation, usable operating modes, and reproducible decision boundaries is well aligned with *Results in Engineering*. The manuscript does not generalize the result to topology reasoning, arbitrary model families, or real-plant deployment. A locally validated public-release archive accompanies the submission package; the persistent DOI/URL and author-owned declarations will be inserted before upload.

The submitting authors must confirm originality, authorship, funding, competing interests, and exclusive submission in the journal system.

Sincerely,

The authors
