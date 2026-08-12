"""Single-H200 loader wrapper for the v8 Qwen quality matrix.

The underlying v7 matrix runner is unchanged.  This wrapper avoids
``device_map='auto'`` (and therefore an unnecessary Accelerate dependency)
because the frozen Qwen3-VL-8B model fits on one H200.
"""

from __future__ import annotations

import run_qwen_counterfactual_prompt_matrix_v7 as base


def load_single_gpu(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).eval().cuda()
    return model, processor


base.load_model = load_single_gpu


if __name__ == "__main__":
    raise SystemExit(base.main())
