from pidbench.splits import (
    assert_source_isolated,
    make_random_split,
    make_source_split,
    summarize_split,
)


def _records():
    return [
        {"instance_id": f"i-{source}-{row}", "source_id": f"source-{source}"}
        for source in range(10)
        for row in range(3)
    ]


def test_source_split_keeps_sources_together() -> None:
    assignments = make_source_split(_records(), seed=17)
    assert_source_isolated(assignments)
    assert summarize_split(assignments)["sources_across_multiple_splits"] == 0


def test_random_split_intentionally_spreads_sources() -> None:
    summary = summarize_split(make_random_split(_records(), seed=17))
    assert summary["sources_across_multiple_splits"] > 0
