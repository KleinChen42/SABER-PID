# SABER-PID

**Source-Isolated Qualification and Cost-Aware Operation of Vision--Language Models for P&ID Tag Retrieval**

SABER-PID is an answer-hidden, source-isolated evaluation and operating
workflow for candidate-tag retrieval from piping and instrumentation diagrams
(P&IDs). It tests whether a frozen vision--language model follows the requested
drawing under declared visual and output budgets, then converts fixed OCR--VLM
outputs into transparent precision-, balance-, or recall-oriented operating
modes.

This repository is the public reproducibility package for a submission to
*Results in Engineering*.

## Submission artifacts

- [Main manuscript](output/pdf/v10/manuscript.pdf)
- [Supplementary material](output/pdf/v10/supplementary.pdf)
- [Clean editable submission package](release/saber_pid_rineng_v10_submission_source.zip)
- [Public reproducibility release](release/saber_pid_rineng_v10_public_release.zip)
- [Data and code availability statement](paper/data_availability.md)
- [License audit](LICENSES.md)

## Evidence snapshot

- On 100 unseen PIDQA drawings, Qwen3-VL-8B reaches strict tag F1 0.5549 with
  the requested image, versus 0.0062 with a source-shuffled image and 0 without
  an image.
- Across 230 source-disjoint drawings, clean and three mild image-quality
  conditions retain correct-minus-shuffled effects of 0.548--0.575.
- A closest-safe 54-tile InternVL3.5-8B setting retains a +0.4715
  requested-drawing effect. This is a gross-input-budget comparison, not
  encoder or information-throughput equivalence.
- On 35 images from 26 public DEXPI test cases, Qwen reaches F1 0.9057; both
  cross-case shuffled and no-image controls score zero.
- The exact missed-tag/false-candidate loss envelope selects intersection,
  joined OCR, OCR-first, or union as the relative miss cost increases.

All pre-specified native-budget cells, including low-magnitude InternVL cells,
remain in the public evidence archive.

## Inference-free reproduction

The core scorer uses the Python standard library. The publication rebuild uses
the pinned analysis dependencies in `pyproject.toml` and
`requirements-analysis-v6.txt`.

```text
python -m pip install -e ".[test]"
python scripts/reproduce_rineng_submission_v10.py --root .
```

PDF compilation additionally requires Tectonic. The reproduction command:

1. rebuilds all nine active figures from frozen machine-readable reports;
2. runs the relevant deterministic tests;
3. compiles and renders the main and supplementary PDFs;
4. records all-page visual validation; and
5. validates authorship metadata, references, placeholders, licenses, and
   public-repository fields.

It performs no model inference and no external dataset download.

## Release validation

```text
python scripts/build_rineng_v10_submission_package.py --root .
python scripts/build_rineng_v10_artifact_manifest.py --root .
python scripts/build_rineng_public_release_v10.py --root . --force
python scripts/validate_rineng_public_release_v10.py --root .
```

The public release validator checks CRCs, member hashes and sizes, deterministic
timestamps, forbidden private/weight files, local-path leakage, required
artifacts, and the expected raw-cell scope.

## Scope

The evidence supports bounded candidate-tag retrieval under the declared
PIDQA and public DEXPI drawing families, models, quality conditions, and visual
budgets. It does not qualify topology reconstruction, arbitrary engineering
documents, autonomous field deployment, or encoder equivalence.

## Licensing and citation

Original SABER-PID software is available under the [MIT License](LICENSE).
PIDQA and DEXPI material retains the vendored upstream CC0 and CC BY 4.0 terms;
model weights are not redistributed. See [LICENSES.md](LICENSES.md) for the
release matrix.

Citation metadata is provided in [CITATION.cff](CITATION.cff). The public data,
code, release archives, and version history are available directly from this
GitHub repository.
