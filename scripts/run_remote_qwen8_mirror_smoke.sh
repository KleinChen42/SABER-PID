#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
# The H200 worker cannot route to huggingface.co directly but can reach this
# Hugging Face-compatible public mirror.  The model/cache remains project-local.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
cd "$ROOT"

exec "$PYTHON" scripts/run_vlm_pilot.py \
  --input data/processed/pilot48_source_train_with_images_public.jsonl \
  --output outputs/pilot/qwen3vl8b_structured_smoke.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --mode structured \
  --max-samples 1 \
  --max-image-side 1536 \
  --max-new-tokens 192
