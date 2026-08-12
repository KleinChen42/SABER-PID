"""InternVL F3 loader with a narrow transformers compatibility shim."""
from __future__ import annotations

import run_internvl35_f3_matrix as base


def load_model(model_path: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    # The vendored transformers expects the renamed public attribute while
    # the InternVL remote class exposes the older private spelling.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
            lambda self: getattr(self, "_tied_weights_keys", {}) or {}
        )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        device_map=None,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


base.load_model = load_model

if __name__ == "__main__":
    raise SystemExit(base.main())
