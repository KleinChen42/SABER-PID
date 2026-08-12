#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
INPUT_A=data/processed/main400_remote_public.jsonl
INPUT_B=data/processed/main400_hashblind_set_b_public.jsonl
OUTDIR=outputs/final_replication
LOG="$ROOT/logs/f2_qwen8_matrix.log"

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p "$OUTDIR" "$ROOT/logs"

echo "F2_START $(date -Is) run_id=f2_qwen8_20260805 gpu=$CUDA_VISIBLE_DEVICES" >> "$LOG"
for SPEC in "A p1" "A p2" "B p0" "B p1" "B p2"; do
  read -r SET_ID PROMPT_ID <<< "$SPEC"
  echo "CELL_START set=$SET_ID prompt=$PROMPT_ID $(date -Is)" >> "$LOG"
  "$PYTHON" scripts/run_vlm_f2_matrix_v2.py \
    --input-a "$INPUT_A" \
    --input-b "$INPUT_B" \
    --output-dir "$OUTDIR" \
    --model "$MODEL" \
    --image-root "$ROOT" \
    --run-id f2_qwen8_20260805 \
    --sets "$SET_ID" \
    --prompts "$PROMPT_ID" \
    --sides 768,3072 \
    --max-new-tokens 192 \
    --skip-existing >> "$LOG" 2>&1
  echo "CELL_END set=$SET_ID prompt=$PROMPT_ID $(date -Is)" >> "$LOG"
done
echo "F2_END $(date -Is)" >> "$LOG"
