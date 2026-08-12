from pidbench.metrics import normalize_answer, score_predictions


def test_normalize_set_and_boolean_answers() -> None:
    assert normalize_answer("['B', 'A']") == ("A", "B")
    assert normalize_answer("True") is True
    assert normalize_answer("2") == 2


def test_score_predictions_reports_source_macro_accuracy() -> None:
    records = [
        {"instance_id": "a", "source_id": "s1", "task": "count", "answer": "2"},
        {"instance_id": "b", "source_id": "s2", "task": "count", "answer": "3"},
    ]
    predictions = [
        {"instance_id": "a", "action": "ANSWER", "answer": "2"},
        {"instance_id": "b", "action": "ANSWER", "answer": "0"},
    ]
    result = score_predictions(records, predictions)
    assert result["overall_accuracy"] == 0.5
    assert result["source_macro_accuracy"] == 0.5
