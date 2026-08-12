from scripts.build_rineng_v9_editorial_assets import latex_escape, pooled_internvl


def test_latex_escape_handles_table_delimiters():
    assert latex_escape("P&ID 95%") == r"P\&ID 95\%"


def test_pooled_internvl_selects_named_contrast():
    report = {
        "internvl_comparisons": [
            {"dataset": "other", "contrast": "correct_minus_shuffled", "value": 0},
            {
                "dataset": "pooled_three_source_disjoint_subsets",
                "contrast": "correct_minus_shuffled",
                "value": 1,
            },
        ]
    }
    assert pooled_internvl(report, "correct_minus_shuffled")["value"] == 1
