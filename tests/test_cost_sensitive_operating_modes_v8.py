import numpy as np

from build_cost_sensitive_operating_modes_v8 import (
    CostLine,
    bootstrap_optimal_probabilities,
    exact_lower_envelope,
)


def test_exact_operating_envelope_matches_set_b_switches() -> None:
    lines = [
        CostLine("intersection", "Intersection", "precision", "#1", fn=242, fp=1, tp=106),
        CostLine("ocr", "OCR joined", "low-fp", "#2", fn=189, fp=29, tp=159),
        CostLine("ocr_first", "OCR-first", "fallback", "#3", fn=173, fp=39, tp=175),
        CostLine("union", "Union", "recall", "#4", fn=103, fp=180, tp=245),
    ]
    envelope = exact_lower_envelope(lines)
    assert [row["method_id"] for row in envelope] == [
        "intersection",
        "ocr",
        "ocr_first",
        "union",
    ]
    np.testing.assert_allclose(
        [row["upper_ratio"] for row in envelope[:-1]],
        [28 / 53, 10 / 16, 141 / 70],
    )


def test_bootstrap_probabilities_are_normalized_and_deterministic() -> None:
    fn = np.asarray([[2, 1], [1, 0], [3, 2]], dtype=np.int64)
    fp = np.asarray([[0, 2], [0, 1], [1, 3]], dtype=np.int64)
    ratios = np.asarray([0.1, 1.0, 10.0])
    first, diagnostics = bootstrap_optimal_probabilities(fn, fp, ratios, reps=200, seed=7)
    second, _ = bootstrap_optimal_probabilities(fn, fp, ratios, reps=200, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=0), 1.0, rtol=0.0, atol=1e-12)
    assert diagnostics["source_count"] == 3
