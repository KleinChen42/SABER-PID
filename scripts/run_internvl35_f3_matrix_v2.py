"""Compatibility wrapper for InternVL3.5 on the pinned remote transformers.

The first smoke exposed an accelerate/custom-model incompatibility in
``device_map='auto'``.  This wrapper keeps the frozen F3 loop but loads the
8B model directly on the selected CUDA device and disables optional flash
attention.
"""
from __future__ import annotations

import run_internvl35_f3_matrix as base


def load_model(model_path: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=None,
    ).eval()
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


base.load_model = load_model

if __name__ == "__main__":
    raise SystemExit(base.main())
