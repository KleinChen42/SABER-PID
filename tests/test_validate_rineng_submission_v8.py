from validate_rineng_submission_v8 import (
    REQUIRED_FIGURES,
    REQUIRED_TABLE_INPUTS,
    TITLE,
)


def test_v8_submission_contract_names_final_extensions() -> None:
    assert "Source-Isolated Counterfactual Evaluation" in TITLE
    assert REQUIRED_FIGURES == (
        "figure_4_cost_sensitive_operating_modes_v8.pdf",
        "figure_5_quality_and_budget_matched_v8.pdf",
        "figure_6_dexpi_external_v8.pdf",
    )
    assert REQUIRED_TABLE_INPUTS == (
        "tables/table_rineng_v8_quality.tex",
        "tables/table_rineng_v8_quality_by_subset.tex",
        "tables/table_rineng_v8_internvl_budget54.tex",
        "tables/table_rineng_v8_dexpi_external.tex",
    )
