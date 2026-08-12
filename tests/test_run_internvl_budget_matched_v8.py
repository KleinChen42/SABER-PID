from run_internvl_budget_matched_v8 import is_fatal_accelerator_error


def test_fatal_accelerator_error_classification() -> None:
    assert is_fatal_accelerator_error(
        RuntimeError("CUDA error: Invalid access of peer GPU memory over nvlink")
    )
    assert is_fatal_accelerator_error(RuntimeError("an illegal memory access was encountered"))
    assert not is_fatal_accelerator_error(ValueError("image manifest hash mismatch"))
