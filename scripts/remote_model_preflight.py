"""Check model-hub reachability and Transformers compatibility before GPU use."""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoProcessor

    result: dict[str, object] = {
        "model": args.model,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    config = AutoConfig.from_pretrained(args.model)
    result["config_model_type"] = str(config.model_type)
    try:
        processor = AutoProcessor.from_pretrained(args.model)
    except Exception as error:  # report a precise compatibility issue to the launcher
        result["processor_ok"] = False
        result["processor_error_type"] = type(error).__name__
        result["processor_error"] = str(error)
    else:
        result["processor_ok"] = True
        result["processor_class"] = type(processor).__name__
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("processor_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
