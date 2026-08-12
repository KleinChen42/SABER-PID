from pathlib import Path

from build_rineng_v8_artifact_manifest import artifact_category


def test_v8_artifact_categories() -> None:
    assert artifact_category(Path("outputs/rineng_v8/a.jsonl")) == "raw_prediction"
    assert artifact_category(Path("data/manifests/plan.json")) == "frozen_plan"
    assert artifact_category(Path("data/answer_store/ref.jsonl")) == "scorer_only_reference"
    assert artifact_category(Path("paper/figures/figure.pdf")) == "paper_figure"
    assert artifact_category(Path("output/pdf/v8/manuscript.pdf")) == "submission_pdf"
