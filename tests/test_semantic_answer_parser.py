from pidbench.semantic_answer_parser import parse_semantic_answer


def test_boolean_prefix_is_semantic_but_not_format_compliant() -> None:
    parsed = parse_semantic_answer("Yes, the symbols are connected.", "connectivity")
    assert parsed.value is True
    assert parsed.parsed is True
    assert parsed.format_compliant is False
    assert parsed.parser_rule == "boolean_prefix"


def test_integer_sentence_is_semantic_but_not_format_compliant() -> None:
    parsed = parse_semantic_answer("There are 12 symbols.", "count")
    assert parsed.value == 12
    assert parsed.parsed is True
    assert parsed.format_compliant is False


def test_integer_with_multiple_values_is_not_silently_repaired() -> None:
    parsed = parse_semantic_answer("There are 2 or 3 symbols.", "count")
    assert parsed.parsed is False
    assert parsed.value is None


def test_tag_parser_handles_punctuation_and_hierarchical_tags() -> None:
    parsed = parse_semantic_answer("The tags are KL-58999, UV-00-001, and SDL 299.", "value")
    assert parsed.value == ("kl-58999", "sdl 299", "uv-00-001")
    assert parsed.parsed is True
    assert parsed.format_compliant is False


def test_tag_list_and_empty_set_are_format_compliant() -> None:
    parsed = parse_semantic_answer("KL-58999, SDL 299", "value")
    assert parsed.value == ("kl-58999", "sdl 299")
    assert parsed.format_compliant is True
    empty = parse_semantic_answer("[]", "value")
    assert empty.value == ()
    assert empty.format_compliant is True
