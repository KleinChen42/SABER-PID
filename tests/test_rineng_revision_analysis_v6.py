from build_rineng_revision_analysis_v6 import (
    deterministic_boundary_case_gallery,
    edit_distance,
    error_pattern,
    per_source_metrics,
    percentile,
)


def test_per_source_metrics_and_patterns() -> None:
    exact = per_source_metrics({"a", "b"}, {"a", "b"})
    assert exact["f1"] == 1.0
    assert error_pattern(exact) == "exact_set"

    partial = per_source_metrics({"a", "b"}, {"a"})
    assert partial["tp"] == 1
    assert partial["fn"] == 1
    assert error_pattern(partial) == "partial_recovery_no_false_candidate"

    mixed = per_source_metrics({"a", "b"}, {"a", "c"})
    assert mixed["fp"] == 1
    assert mixed["fn"] == 1
    assert error_pattern(mixed) == "partial_recovery_with_false_candidates"


def test_empty_set_convention_is_explicit() -> None:
    both_empty = per_source_metrics(set(), set())
    assert both_empty["precision"] == 1.0
    assert both_empty["recall"] == 1.0
    assert both_empty["f1"] == 1.0
    assert both_empty["exact"] == 1

    missed = per_source_metrics({"a"}, set())
    assert missed["precision"] == 0.0
    assert missed["recall"] == 0.0
    assert missed["f1"] == 0.0


def test_percentile_matches_project_nearest_index_rule() -> None:
    assert percentile([0, 1, 2, 3, 4], 0.25) == 1.0
    assert percentile([0, 1, 2, 3, 4], 0.95) == 4.0
    assert percentile([], 0.5) is None


def test_edit_distance_for_tag_form_diagnostics() -> None:
    assert edit_distance("kl58999", "kl58999") == 0
    assert edit_distance("kl58999", "kl5899") == 1
    assert edit_distance("uv001", "uv101") == 1


def test_boundary_gallery_selects_predeclared_qwen_strata() -> None:
    patterns = {
        "source-success": "exact_set",
        "source-partial": "partial_recovery_with_false_candidates",
        "source-failure": "false_candidates_without_recovery",
    }
    per_method = {}
    for method in ("qwen", "paddleocr_geometry", "set_union", "set_intersection"):
        per_method[method] = {
            source_id: {
                "source_id": source_id,
                "instance_id": f"instance-{source_id}",
                "truth_tags": "a;b",
                "prediction_tags": "a;b" if pattern == "exact_set" else "a;c",
                "tp": 2 if pattern == "exact_set" else 1 if "partial" in pattern else 0,
                "fp": 0 if pattern == "exact_set" else 1,
                "fn": 0 if pattern == "exact_set" else 1 if "partial" in pattern else 2,
                "error_pattern": pattern,
            }
            for source_id, pattern in patterns.items()
        }
    cases = deterministic_boundary_case_gallery(per_method)
    assert [case["case"] for case in cases] == ["success", "partial", "failure"]
    assert [case["qwen_outcome_stratum"] for case in cases] == list(patterns.values())
    assert all(case["eligible_source_count"] == 1 for case in cases)
