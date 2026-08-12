from pathlib import Path

from build_rineng_v8_extension_figures import (
    build_external_figure,
    build_quality_figure,
    find_internvl,
    find_quality,
)


def test_find_frozen_extension_rows() -> None:
    report = {
        "quality_comparisons": [
            {
                "dataset": "pooled_three_source_disjoint_subsets",
                "quality": "clean",
                "contrast": "correct_minus_shuffled",
                "value_f1_difference": 0.5,
            }
        ],
        "internvl_comparisons": [
            {
                "dataset": "pooled_three_source_disjoint_subsets",
                "contrast": "correct_minus_shuffled",
                "value_f1_difference": 0.2,
            }
        ],
    }
    assert find_quality(report, "clean", "correct_minus_shuffled")["value_f1_difference"] == 0.5
    assert find_internvl(report, "correct_minus_shuffled")["value_f1_difference"] == 0.2


def test_extension_figures_render_from_complete_schema() -> None:
    pooled = "pooled_three_source_disjoint_subsets"
    quality_rows = []
    for index, quality in enumerate(("clean", "jpeg_q70", "blur_r1", "downsample_s075")):
        point = 0.55 - 0.04 * index
        quality_rows.append(
            {
                "dataset": pooled,
                "quality": quality,
                "contrast": "correct_minus_shuffled",
                "value_f1_difference": point,
                "value_f1_source_bootstrap_ci95": [point - 0.08, point + 0.08],
            }
        )
        if quality != "clean":
            change = -0.04 * index
            quality_rows.append(
                {
                    "dataset": pooled,
                    "quality": quality,
                    "contrast": "degraded_minus_clean_change_in_correct_minus_shuffled_value_f1",
                    "value_f1_difference_in_differences": change,
                    "source_bootstrap_ci95": [change - 0.06, change + 0.06],
                }
            )
    extension = {
        "quality_comparisons": quality_rows,
        "internvl_comparisons": [
            {
                "dataset": pooled,
                "contrast": "correct_minus_shuffled",
                "value_f1_difference": 0.12,
                "value_f1_source_bootstrap_ci95": [0.03, 0.21],
            }
        ],
    }
    external = {
        "selection": {
            "selected_image_count": 35,
            "logical_test_case_count": 26,
            "question_count": 65,
        },
        "metrics": {
            "correct": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
            "shuffled": {"precision": 0.2, "recall": 0.05, "f1": 0.08},
            "text_only": {"precision": 0.1, "recall": 0.0, "f1": 0.0},
            "paddleocr_full_image": {"precision": 0.96, "recall": 0.60, "f1": 0.74},
        },
        "comparisons": {
            "correct_minus_shuffled": {"difference": 0.67, "ci95": [0.5, 0.8]},
            "correct_minus_text_only": {"difference": 0.75, "ci95": [0.6, 0.85]},
            "correct_minus_paddleocr_full_image": {"difference": 0.01, "ci95": [-0.1, 0.12]},
            "ocr_minus_text_only": {"difference": 0.74, "ci95": [0.6, 0.82]},
        },
    }
    output = Path("outputs/test_runtime_v8_unit/figure_schema")
    quality_files, _ = build_quality_figure(extension, output)
    external_files, _ = build_external_figure(external, output)
    assert all(path.is_file() and path.stat().st_size > 0 for path in quality_files + external_files)
