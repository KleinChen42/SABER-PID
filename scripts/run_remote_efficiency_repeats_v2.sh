#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
INPUT=data/processed/efficiency_subset_v2_public.jsonl
OUT=outputs/telemetry/efficiency_repeats_v2.jsonl
export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p outputs/telemetry logs

run_cell() {
  local family="$1" condition="$2" side="$3" model="$4"
  echo "START family=$family condition=$condition side=$side $(date -Is)" >> logs/efficiency_repeats_v2.log
  "$PYTHON" scripts/run_efficiency_repeats_v2.py --input "$INPUT" --output "$OUT" --model "$model" --family "$family" --condition "$condition" --max-image-side "$side" --run-id f6_efficiency_v2_20260805 --image-root "$ROOT" --repeats 3 --warmup 20 --resume >> logs/efficiency_repeats_v2.log 2>&1
  echo "END family=$family condition=$condition side=$side $(date -Is)" >> logs/efficiency_repeats_v2.log
}

Q8="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
Q32="$ROOT/models/Qwen3-VL-32B-Instruct-modelscope"
IVL="$ROOT/models/InternVL3_5-8B-modelscope"
for side in 768 1536 2304 3072; do run_cell qwen "qwen8_${side}" "$side" "$Q8"; done
for side in 1536 3072; do run_cell qwen "qwen32_${side}" "$side" "$Q32"; done
for side in 768 3072; do run_cell internvl "internvl35_${side}" "$side" "$IVL"; done
