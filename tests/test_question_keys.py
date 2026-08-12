from pidbench.question_keys import question_semantic_signature


def test_query_key_parser_handles_each_pidqa_task() -> None:
    assert question_semantic_signature("count", "How many symbols are categorized as 4?") == (
        "count",
        (("Symbol_Class", "4"),),
    )
    assert question_semantic_signature(
        "connectivity", "Are class 7 symbols linked to both class 9 and class 20?"
    ) == (
        "connectivity",
        (("Symbol_XX", "7"), ("Symbol_YY", "9"), ("Symbol_ZZ", "20")),
    )
    assert question_semantic_signature(
        "spatial_count",
        "How many symbols are directly connected to symbols of class 24, given that they belong to class 20?",
    ) == (
        "spatial_count",
        (("Symbol_XX", "20"), ("Symbol_YY", "24")),
    )
    assert question_semantic_signature(
        "value", "List class 12 symbol tags that begin with WX"
    ) == ("value", (("Prefix", "WX"), ("Symbol_Class", "12")))
