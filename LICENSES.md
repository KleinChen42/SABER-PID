# License and release audit matrix

This file is a machine-assisted release inventory, not legal advice. It records
what is present in this workspace and what may be redistributed. Official
license text and model-card terms must be rechecked at submission/release time.

| Item | Evidence in workspace | Observed term/status | Release action |
|---|---|---|---|
| PIDQA data and derived split IDs | `licenses/PIDQA_LICENSE.txt`, `data/processed/`, `data/answer_store/` | Vendored license text is CC0 1.0 Universal | Code, manifests, split IDs and derived tables may be released with provenance |
| PIDQA raw images | `data/raw/PIDQA/`, `paper/assets/pidqa_sheet_282.jpg`, `paper/assets/pidqa_sheet_184.jpg` | The copied source license is preserved byte-for-byte in `licenses/PIDQA_LICENSE.txt` (SHA-256 `f4e7f373...17ad328`) | The full collection remains acquisition-by-reference; the two deterministic Figure 1 source drawings are included with provenance |
| Qwen3-VL 8B/32B weights | Remote H200 model directories; model-card URLs in `16_REFERENCES.md` | Weights are not stored in this repository; exact license text/revision is not vendored | Do not redistribute weights; publish model IDs, revision/hash metadata and download instructions |
| InternVL family | `reports/INTERNVL_CORRECTED_REPLICATION_CLOSEOUT.md`, `reports/generated/internvl_tile_budget_v1.json`, `outputs/editorial_revision/internvl_counterfactual_ladder_v1/`, `outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix/` | The E4 boundary and frozen InternVL3.5-8B value ladder were scored from a remote checkpoint; the warned first pass and tokenizer-corrected final pass are both retained, while weights are not stored here | Do not redistribute weights; retain model-path metadata, tokenizer correction, frozen outputs and scoring provenance |
| PaddleOCR comparator | `reports/generated/paddleocr_environment_v1.txt`, `outputs/editorial_revision/paddleocr_value_baseline_v1/` | PaddleOCR 2.8.1 and PaddlePaddle 2.6.2 were frozen for the full-image comparator; downloaded inference weights are not stored here | Release package versions, pipeline settings and frozen OCR outputs; do not vendor downloaded weights |
| DEXPI Public Example PIDs | `licenses/DEXPI_TRAINING_TEST_CASES_LICENSE.txt`, `data/manifests/rineng_v8_dexpi_external_plan.json`, `reports/generated/rineng_v8_dexpi_external_audit.json` | Official repository commit `a23d61e2e089eb2ca464cd552f9ae580a2785963`; CC BY 4.0 license SHA-256 `7e7170e3...1c8a2661` | Release frozen acquisition/selection provenance, derived manifests, scorer-only references, outputs, and attribution; keep the upstream collection acquisition-by-reference unless the final archive explicitly includes the selected licensed assets |
| PID2Graph/OPEN100 | `reports/generated/pid2graph_open100_complete_materialized_v8.json`, `reports/generated/rineng_v8_pid2graph_open100_audit.json` | Official Zenodo record declares CC BY-SA 4.0, 9,303,633,645 bytes, and MD5 `90f782220de97e7e249d2595c49ddc1c`; 24 members for 12 complete plans were sparsely materialized and SHA-256 inventoried | Do not redistribute the upstream archive in the core package; release the transport catalogue and automated task-fit audit. Do not report a tag score because the GraphML has no explicit visible-text/tag reference field |
| Project source code | `LICENSE`, `pyproject.toml`, `src/`, `scripts/` | Original SABER-PID project code is released under the MIT License | Preserve the root MIT license in source and release archives; third-party data, assets, and model outputs remain governed by their own terms |
| Python runtime dependencies | `pyproject.toml`, `requirements-analysis-v6.txt` | Python >=3.10; V10 analysis dependencies and the test extra are pinned to the recorded Python 3.10.12 environment | Install the declared project dependencies; PDF compilation additionally requires the external Tectonic executable |
| Generated tables/figures/manifests | `reports/generated/` | Derived from the source-isolated PIDQA run and model outputs | Release with provenance, hashes and the applicable dataset/model restrictions |

## Automated findings

- The PIDQA license text is preserved in `licenses/PIDQA_LICENSE.txt` (CC0 1.0)
  so the release archive does not depend on the excluded raw-data tree.
- No model weights are present in the repository tree.
- The E4 and v4 InternVL results are retained with raw-output provenance; no InternVL or PaddleOCR weights are vendored. No PID2Graph tag score is included because its verified structural GraphML does not provide an explicit visible-tag reference.
- The DEXPI license is preserved byte-for-byte, and the V8 external-family plan
  freezes the upstream commit, license hash, selected source hashes, exact
  image/XML pairing rule, answer isolation, and logical-case grouping.
- The safest release bundle is source code, deterministic scripts, manifests,
  split IDs, hashes, tables and figures; large/raw/model artifacts remain
  acquisition-by-reference unless their terms are rechecked.
- The root MIT License covers original SABER-PID software only. It does not
  relicense PIDQA, DEXPI, model weights, OCR/model outputs, or other third-party
  material; their recorded upstream terms continue to apply.
