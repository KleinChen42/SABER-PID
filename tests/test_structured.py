from pidbench.structured import check_prediction, parse_structured_output


def test_parser_and_checker_accept_internal_graph() -> None:
    parsed = parse_structured_output(
        '{"action":"ANSWER","answer":"yes","entities":[{"local_id":"E1"},{"local_id":"E2"}],"edges":[{"source":"E1","target":"E2"}],"evidence":["E1"]}'
    )
    assert parsed["ok"]
    assert check_prediction(parsed["prediction"])["valid"]


def test_checker_rejects_unknown_edge_endpoint() -> None:
    checked = check_prediction(
        {
            "action": "ANSWER",
            "answer": "yes",
            "entities": [{"local_id": "E1"}],
            "edges": [{"source": "E1", "target": "E2"}],
        }
    )
    assert not checked["valid"]
    assert "unknown_edge_endpoint" in checked["violations"]
