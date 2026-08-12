from pidbench.pidqa_metrics import normalize_pidqa_answer, score_pidqa_predictions


def test_value_tags_accept_comma_or_list_forms() -> None:
    assert normalize_pidqa_answer("['AB-2', 'AB-1']", "value") == ("ab-1", "ab-2")
    assert normalize_pidqa_answer("AB-1, AB-2", "value") == ("ab-1", "ab-2")


def test_boolean_yes_no_is_task_aware() -> None:
    assert normalize_pidqa_answer("No", "connectivity") is False
    assert normalize_pidqa_answer("yes", "connectivity") is True


def test_score_uses_value_set_semantics() -> None:
    records = [
        {"instance_id": "a", "task": "value", "source_id": "s", "answer": "['A', 'B']"},
        {"instance_id": "b", "task": "connectivity", "source_id": "s", "answer": "False"},
    ]
    predictions = [
        {"instance_id": "a", "action": "ANSWER", "answer": "B, A"},
        {"instance_id": "b", "action": "ANSWER", "answer": "No"},
    ]
    assert score_pidqa_predictions(records, predictions)["overall_accuracy"] == 1.0
