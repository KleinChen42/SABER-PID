#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
MODEL="$ROOT/models/Qwen3-VL-32B-Instruct-modelscope"
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
INPUT=data/processed/main400_remote_public.jsonl

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p "$ROOT/outputs/main" "$ROOT/logs"

for SIDE in 1536 3072; do
  OUTPUT="outputs/main/qwen3vl32b_source400_clean_${SIDE}.jsonl"
  LOG="logs/qwen3vl32b_source400_clean_${SIDE}.log"
  echo "START side=$SIDE $(date -Is)" >> "$LOG"
  "$PYTHON" scripts/run_vlm_pilot_robust.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --mode direct \
    --max-samples 0 \
    --max-image-side "$SIDE" \
    --max-new-tokens 192 \
    --resume >> "$LOG" 2>&1
  echo "END side=$SIDE $(date -Is)" >> "$LOG"
done
