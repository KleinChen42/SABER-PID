from validate_rineng_submission_v6 import expand_tex_inputs, latex_ids


def test_expand_tex_inputs_exposes_table_labels() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        r"Table~\ref{tab:v7_counterfactual}. "
        r"\input{tables/table_rineng_overnight_v7_counterfactual.tex}"
    )

    expanded = expand_tex_inputs(root, source)

    assert "tab:v7_counterfactual" in latex_ids(expanded, "label")
    assert latex_ids(expanded, "ref") == {"tab:v7_counterfactual"}
