from score_rineng_v8_extensions import paired_f1_difference_in_differences


def test_paired_f1_difference_in_differences() -> None:
    sources = ["a", "b", "c"]
    clean_correct = {source: (1, 0, 0) for source in sources}
    clean_shuffled = {source: (0, 0, 1) for source in sources}
    degraded_correct = {source: (0, 0, 1) for source in sources}
    degraded_shuffled = {source: (0, 0, 1) for source in sources}
    point, interval = paired_f1_difference_in_differences(
        clean_correct,
        clean_shuffled,
        degraded_correct,
        degraded_shuffled,
        reps=200,
        seed=4,
    )
    assert point == -1.0
    assert interval == [-1.0, -1.0]
