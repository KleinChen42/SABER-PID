from build_rineng_v8_tables import (
    POOLED,
    QUALITY_ORDER,
    build_external_table,
    build_internvl_table,
    build_quality_subset_table,
    build_quality_table,
)


def tag_metrics(tp: int, fp: int, fn: int) -> dict:
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def test_v8_tables_build_from_frozen_report_schema() -> None:
    datasets = ("set_b100", "seed29_strict65", "seed31_strict65")
    cells = {}
    quality_comparisons = []
    for quality_index, quality in enumerate(QUALITY_ORDER):
        for dataset in datasets:
            cells[f"quality|qwen3vl8b|{dataset}|{quality}|correct"] = {
                "metrics": {"strict_value_tags": tag_metrics(30 - quality_index, 5, 10)}
            }
            cells[f"quality|qwen3vl8b|{dataset}|{quality}|shuffled"] = {
                "metrics": {"strict_value_tags": tag_metrics(1, 3, 39)}
            }
            quality_comparisons.append(
                {
                    "dataset": dataset,
                    "quality": quality,
                    "contrast": "correct_minus_shuffled",
                    "value_f1_difference": 0.5 - 0.03 * quality_index,
                    "value_f1_source_bootstrap_ci95": [0.4, 0.6],
                    "source_count": 65 if dataset != "set_b100" else 100,
                }
            )
        quality_comparisons.append(
            {
                "dataset": POOLED,
                "quality": quality,
                "contrast": "correct_minus_shuffled",
                "value_f1_difference": 0.5 - 0.03 * quality_index,
                "value_f1_source_bootstrap_ci95": [0.4, 0.6],
                "source_count": 230,
            }
        )
        if quality != "clean":
            quality_comparisons.append(
                {
                    "dataset": POOLED,
                    "quality": quality,
                    "contrast": "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1",
                    "value_f1_difference_in_differences": -0.03 * quality_index,
                    "source_bootstrap_ci95": [-0.1, 0.02],
                    "source_count": 230,
                }
            )

    internvl_comparisons = []
    for dataset in (*datasets, POOLED):
        if dataset != POOLED:
            for condition, counts in {
                "correct": (4, 3, 36),
                "shuffled": (0, 1, 40),
                "text_only": (0, 0, 40),
            }.items():
                cells[f"internvl_budget54|{dataset}|{condition}"] = {
                    "metrics": {"strict_value_tags": tag_metrics(*counts)}
                }
        for contrast, point in {
            "correct_minus_shuffled": 0.12,
            "correct_minus_text_only": 0.13,
            "budget54_minus_native_tiles12_correct": 0.10,
        }.items():
            internvl_comparisons.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "value_f1_difference": point,
                    "value_f1_source_bootstrap_ci95": [0.03, 0.20],
                }
            )
    extension = {
        "cells": cells,
        "quality_comparisons": quality_comparisons,
        "internvl_comparisons": internvl_comparisons,
    }
    quality_tex, quality_rows = build_quality_table(extension)
    quality_subset_tex, quality_subset_rows = build_quality_subset_table(extension)
    internvl_tex, internvl_rows = build_internvl_table(extension)
    assert "Paired quality robustness" in quality_tex
    assert len(quality_rows) == 4
    assert "source-disjoint subset" in quality_subset_tex
    assert len(quality_subset_rows) == 3
    assert "Closest-safe InternVL" in internvl_tex
    assert len(internvl_rows) == 4

    metrics = tag_metrics(49, 2, 32)
    metrics.update(
        {
            "records": 65,
            "logical_groups": 26,
            "exact_set_accuracy": 0.6,
            "logical_group_macro_exact_accuracy": 0.58,
        }
    )
    external = {
        "metrics": {
            condition: dict(metrics)
            for condition in ("correct", "shuffled", "text_only", "paddleocr_full_image")
        }
    }
    external_tex, external_rows = build_external_table(external)
    assert "External-family tag retrieval" in external_tex
    assert len(external_rows) == 4
