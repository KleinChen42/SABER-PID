import pytest

from build_positive_narrative_hybrid_analysis_v5 import RULES, answer_set, fuse, render
from build_editorial_revision_submission_v4 import fusion_validation_narrative


def test_frozen_fusion_rule_family_is_complete_and_truth_free() -> None:
    qwen = {"ft-101", "pt-202"}
    ocr = {"pt-202", "tt-303"}

    assert RULES == (
        "set_union",
        "set_intersection",
        "ocr_if_nonempty_else_qwen",
        "qwen_if_nonempty_else_ocr",
    )
    assert fuse(qwen, ocr, "set_union") == {"ft-101", "pt-202", "tt-303"}
    assert fuse(qwen, ocr, "set_intersection") == {"pt-202"}
    assert fuse(qwen, ocr, "ocr_if_nonempty_else_qwen") == ocr
    assert fuse(qwen, ocr, "qwen_if_nonempty_else_ocr") == qwen


def test_fallback_rules_use_the_other_instrument_only_for_empty_predictions() -> None:
    qwen = {"ft-101"}
    ocr = {"pt-202"}

    assert fuse(qwen, set(), "ocr_if_nonempty_else_qwen") == qwen
    assert fuse(set(), ocr, "qwen_if_nonempty_else_ocr") == ocr


def test_answer_set_and_render_are_deterministic() -> None:
    parsed = answer_set({"answer": "PT-202, FT-101, PT-202"})
    assert parsed == {"pt-202", "ft-101"}
    assert render(parsed) == "ft-101, pt-202"
    assert render(set()) == "[]"


def test_unknown_fusion_rule_fails_closed() -> None:
    with pytest.raises(KeyError):
        fuse({"ft-101"}, {"pt-202"}, "best_on_test_set")


def test_validation_narrative_preserves_latex_and_within_family_boundary() -> None:
    def cell(f1: float, precision: float = 0.8, recall: float = 0.6) -> dict:
        return {
            "metrics": {
                "strict_value_tags": {"precision": precision, "recall": recall, "f1": f1},
                "task": {"value": {"strict_accuracy": 0.2}},
            }
        }

    seeds = []
    for seed, effect in ((29, 0.08), (31, 0.06)):
        seeds.append(
            {
                "seed": seed,
                "cells": {
                    "qwen": cell(0.55),
                    "paddleocr_geometry": cell(0.59),
                    "set_union": cell(0.63),
                    "set_intersection": cell(0.46, precision=0.99, recall=0.30),
                },
                "comparisons": [
                    {
                        "comparison": f"seed{seed}_union_minus_qwen",
                        "metric": "strict_value_tag_f1",
                        "task": "value",
                        "difference_condition_minus_baseline": effect,
                        "source_bootstrap_ci95_low": 0.01,
                        "source_bootstrap_ci95_high": 0.14,
                    },
                    {
                        "comparison": f"seed{seed}_union_minus_ocr",
                        "metric": "strict_value_tag_f1",
                        "task": "value",
                        "difference_condition_minus_baseline": 0.04,
                        "source_bootstrap_ci95_low": -0.02,
                        "source_bootstrap_ci95_high": 0.10,
                    },
                ],
            }
        )

    abstract, main, supplement, highlight = fusion_validation_narrative({"seeds": seeds})
    assert "prospective within-family confirmation" in abstract
    assert "not across an external dataset" in main
    assert r"\begin{table}" in supplement
    assert "\x08" not in supplement
    assert len(highlight) <= 85
