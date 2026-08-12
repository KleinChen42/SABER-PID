from validate_rineng_v8_extensions import (
    bootstrap_delta,
    bootstrap_did,
    external_cell,
    normalized_external_tags,
    pidqa_cell,
)


def test_independent_pidqa_and_bootstrap_helpers() -> None:
    references = [
        {
            "instance_id": "a",
            "source_id": "s1",
            "task": "value",
            "answer": ["P-100", "P-101"],
        },
        {
            "instance_id": "b",
            "source_id": "s2",
            "task": "value",
            "answer": ["T-200"],
        },
    ]
    correct = [
        {"instance_id": "a", "raw": "P-100, P-101", "status": "ok"},
        {"instance_id": "b", "raw": "T-200", "status": "ok"},
    ]
    control = [
        {"instance_id": "a", "raw": "P-100", "status": "ok"},
        {"instance_id": "b", "raw": "[]", "status": "ok"},
    ]
    correct_cell = pidqa_cell(references, correct)
    control_cell = pidqa_cell(references, control)
    assert correct_cell["metrics"]["f1"] == 1.0
    assert control_cell["metrics"]["f1"] < 1.0
    interval = bootstrap_delta(
        control_cell["counts"], correct_cell["counts"], reps=100, seed=2
    )
    assert interval[0] >= 0
    point, did_interval = bootstrap_did(
        correct_cell["counts"],
        control_cell["counts"],
        control_cell["counts"],
        control_cell["counts"],
        reps=100,
        seed=3,
    )
    assert point < 0
    assert did_interval[1] <= 0


def test_independent_external_parser_and_grouping() -> None:
    assert normalized_external_tags("PI 4712.01, T-200", "PI") == {"pi4712-01"}
    references = [
        {
            "instance_id": "a",
            "source_id": "s1",
            "source_sheet": "C01",
            "fields": {"Prefix": "PI"},
            "answer": ["pi4712-01"],
        }
    ]
    cell = external_cell(
        references,
        [{"instance_id": "a", "raw": "PI-4712/01", "status": "ok"}],
    )
    assert cell["metrics"]["tp"] == 1
    assert cell["metrics"]["f1"] == 1.0
