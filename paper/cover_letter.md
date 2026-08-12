# Cover letter

Dear Editor,

Please consider our manuscript, **Qualifying Image-Grounded Tag Retrieval in Piping and Instrumentation Diagrams with Source-Isolated Counterfactual Evaluation**, as a Research Article in *Results in Engineering*.

The paper addresses an engineering decision that aggregate benchmark accuracy cannot answer: when is a candidate tag retrieved from the requested piping and instrumentation diagram, and which operating mode should be used under different missed-tag and review costs? SABER-PID combines drawing-level source isolation, answer-hidden inference, correct/source-shuffled/no-image interventions, recorded visual budgets, and source-cluster uncertainty estimates.

The main PIDQA study qualifies a high-detail Qwen operating point and translates complementary Qwen/OCR errors into usable modes. Union reaches recall 0.7040 and F1 0.6339, whereas intersection reaches precision 0.9907. A scorer-only loss envelope gives an explicit rule for selecting intersection, joined OCR, OCR-first, or union as the missed-tag/false-candidate cost ratio changes. A 54-cell frozen extension tests two Qwen scales, InternVL3.5-8B, three source-disjoint subsets, two prompts, and both shuffled and no-image controls without selecting a favorable prompt or subset.

Three new extensions materially strengthen the evidence. First, a 7,360-row paired matrix at the qualified 3072-side/512-token budget shows correct-minus-shuffled F1 effects of 0.548--0.575 under clean, JPEG-Q70, mild blur, and downsample-and-restore conditions; every quality-minus-clean interval includes zero. Second, a closest-safe 54-tile InternVL pass reaches pooled correct-image F1 0.4846 and a +0.4715 correct-minus-shuffled effect [0.4064, 0.5332], improving by +0.4681 [0.4043, 0.5290] over native 12 tiles. This turns visual-budget adequacy into an experimentally demonstrated cross-model qualification variable. Third, a separately licensed public DEXPI family provides 35 visibility-qualified images and 65 tag questions. Qwen reaches precision 0.9231, recall 0.8889, and F1 0.9057 on correct pairs, compared with zero F1 after cross-case shuffling and with no image. Frozen full-image OCR reaches F1 0.7424. Source selection, tag visibility, image/XML pairing, and derangement are deterministic and independent of model outputs.

The contribution is an engineering qualification and decision workflow, not a new model architecture. All claims are regenerated from immutable outputs, 10,000-replicate clustered intervals, independent validation code, and a SHA-256 inventoried public-release candidate. The manuscript explicitly limits its claim to candidate-tag retrieval under the evaluated PIDQA and DEXPI conditions and does not claim topology understanding or field deployment. This combination of reproducible engineering evidence, external-family transfer, robustness at a declared operating point, and cost-sensitive deployment choices is closely aligned with *Results in Engineering*.

The submitting authors will complete the author, funding, competing-interest, originality, and public DOI/URL fields in the journal system.

Sincerely,

The authors
