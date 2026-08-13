# Cover letter

Dear Editor,

Please consider our manuscript, **SABER-PID: Source-Isolated Qualification and Cost-Aware Operation of Vision--Language Models for P&ID Tag Retrieval**, as a Research Article in *Results in Engineering*.

The paper addresses an engineering decision that aggregate benchmark accuracy cannot answer: when is a candidate tag retrieved from the requested piping and instrumentation diagram, and which operating mode should be used under different missed-tag and review costs? SABER-PID combines drawing-level source isolation, answer-hidden inference, correct/source-shuffled/no-image interventions, recorded visual budgets, and source-cluster uncertainty estimates.

The main PIDQA study qualifies a high-detail Qwen operating point and translates complementary Qwen/OCR errors into usable modes. Union reaches recall 0.7040 and F1 0.6339, whereas intersection reaches precision 0.9907. A scorer-only loss envelope gives an explicit rule for selecting intersection, joined OCR, OCR-first, or union as the missed-tag/false-candidate cost ratio changes. The central narrative is therefore a complete qualification-to-operation path rather than a benchmark-score comparison.

Three extensions strengthen the evidence without changing the task after results were observed. First, a 7,360-row paired matrix at the qualified 3072-side/512-token budget retains correct-minus-shuffled F1 effects of 0.548--0.575 under clean, JPEG-Q70, mild blur, and downsample-and-restore conditions. Second, a closest-safe 54-tile InternVL pass reaches pooled correct-image F1 0.4846 and a +0.4715 correct-minus-shuffled effect [0.4064, 0.5332], showing that visual-budget adequacy is a first-class cross-model qualification variable. Third, a separately licensed public DEXPI family provides 35 visibility-qualified images and 65 tag questions. Qwen reaches precision 0.9231, recall 0.8889, and F1 0.9057 on correct pairs, compared with zero F1 after cross-case shuffling and with no image. Source selection, tag visibility, image/XML pairing, and derangement are deterministic and independent of model outputs.

The contribution is an engineering qualification and decision workflow, not a new model architecture. It differs from recent P&ID graph-construction and graph-retrieval systems by asking whether raw-image tag output follows the requested drawing before it enters a higher-level engineering pipeline. All claims are regenerated from immutable outputs, 10,000-replicate clustered intervals, deterministic validation code, and a SHA-256-inventoried public-release candidate. The manuscript explicitly limits its claim to candidate-tag retrieval under the evaluated PIDQA and DEXPI conditions. This combination of reproducible engineering evidence, external-family transfer, robustness at a declared operating point, and cost-sensitive operating choices is closely aligned with *Results in Engineering*.

The authors declare no competing interests and no specific funding for this work. Zhuo Chen is the corresponding author. The authors will complete the originality confirmation and public DOI/URL fields in the journal system.

Sincerely,

Zhuo Chen, corresponding author, on behalf of all authors (zhuoc@chalmers.se)
