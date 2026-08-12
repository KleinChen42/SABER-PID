#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
INPUT=data/processed/main400_remote_public.jsonl

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p "$ROOT/outputs/main" "$ROOT/data/processed/r4_conditions" "$ROOT/data/raw/r4_conditions" "$ROOT/logs"

for CONDITION in blur jpeg35 center_crop; do
  CONDITION_INPUT="data/processed/r4_conditions/main400_${CONDITION}_public.jsonl"
  "$PYTHON" scripts/make_extended_image_conditions.py \
    --input "$INPUT" \
    --condition "$CONDITION" \
    --image-output-root "data/raw/r4_conditions/${CONDITION}" \
    --records-output "$CONDITION_INPUT" \
    --summary "reports/generated/main400_${CONDITION}_summary.json"
  OUTPUT="outputs/main/qwen3vl8b_source400_${CONDITION}_1536.jsonl"
  LOG="logs/qwen3vl8b_source400_${CONDITION}_1536.log"
  echo "START condition=$CONDITION $(date -Is)" >> "$LOG"
  "$PYTHON" scripts/run_vlm_pilot_robust.py \
    --input "$CONDITION_INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --mode direct \
    --max-samples 0 \
    --max-image-side 1536 \
    --max-new-tokens 192 \
    --resume >> "$LOG" 2>&1
  echo "END condition=$CONDITION $(date -Is)" >> "$LOG"
done
