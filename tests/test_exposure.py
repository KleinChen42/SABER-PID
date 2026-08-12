from pidbench.exposure import build_same_source_cache_audit, semantic_query_signature


def test_same_source_semantic_cache_only_uses_training_answer() -> None:
    records = [
        {
            "instance_id": "train",
            "source_id": "sheet-a",
            "task": "count",
            "question": "How many class 1 symbols?",
            "fields": {"Symbol_Class": "1"},
            "answer": "2",
        },
        {
            "instance_id": "same-query",
            "source_id": "sheet-a",
            "task": "count",
            "question": "What is the count of class 1?",
            "fields": {"Symbol_Class": "1"},
            "answer": "2",
        },
        {
            "instance_id": "unseen-query",
            "source_id": "sheet-a",
            "task": "count",
            "question": "How many class 2 symbols?",
            "fields": {"Symbol_Class": "2"},
            "answer": "3",
        },
        {
            "instance_id": "unseen-source",
            "source_id": "sheet-b",
            "task": "count",
            "question": "How many class 1 symbols?",
            "fields": {"Symbol_Class": "1"},
            "answer": "7",
        },
    ]
    assignments = [
        {"instance_id": "train", "split": "train"},
        {"instance_id": "same-query", "split": "test"},
        {"instance_id": "unseen-query", "split": "test"},
        {"instance_id": "unseen-source", "split": "test"},
    ]

    predictions, summary = build_same_source_cache_audit(records, assignments)

    assert semantic_query_signature(records[0]) == semantic_query_signature(records[1])
    assert summary["same_source_test_rate"] == 2 / 3
    assert summary["unambiguous_cache_hit_rate"] == 1 / 3
    assert [row["action"] for row in predictions] == ["ANSWER", "ABSTAIN", "ABSTAIN"]
    assert predictions[0]["answer"] == "2"
    assert summary["test_answers_used_to_build_predictions"] is False
