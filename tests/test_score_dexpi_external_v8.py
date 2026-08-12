from pathlib import Path

from score_dexpi_external_v8 import paired_group_bootstrap, report_path, score_prediction_rows


def test_external_scoring_and_group_bootstrap() -> None:
    references = [
        {
            "instance_id": "a",
            "source_id": "s1",
            "source_sheet": "C01",
            "fields": {"Prefix": "P"},
            "answer": ["p100", "p101"],
        },
        {
            "instance_id": "b",
            "source_id": "s2",
            "source_sheet": "C02",
            "fields": {"Prefix": "T"},
            "answer": ["t200"],
        },
    ]
    correct = [
        {"instance_id": "a", "raw": "P100, P101", "status": "ok", "test_answer_used": False},
        {"instance_id": "b", "raw": "T200", "status": "ok", "test_answer_used": False},
    ]
    control = [
        {"instance_id": "a", "raw": "P100", "status": "ok", "test_answer_used": False},
        {"instance_id": "b", "raw": "[]", "status": "ok", "test_answer_used": False},
    ]
    correct_metrics, correct_counts, _ = score_prediction_rows(references, correct)
    control_metrics, control_counts, _ = score_prediction_rows(references, control)
    assert correct_metrics["f1"] == 1.0
    assert control_metrics["f1"] < 1.0
    comparison = paired_group_bootstrap(
        control_counts, correct_counts, reps=100, seed=1
    )
    assert comparison["difference"] > 0
    assert len(comparison["ci95"]) == 2


def test_external_score_paths_are_portable_inside_repository() -> None:
    root = Path("C:/repo").resolve()
    assert report_path(root / "outputs" / "cell.jsonl", root) == "outputs/cell.jsonl"
