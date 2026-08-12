#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
SET_ID=${1:?set id}
PROMPT_ID=${2:?prompt id}
SIDES=${3:?comma-separated sides}
LOG="$ROOT/logs/f2_cell_${SET_ID}_${PROMPT_ID}_${SIDES//,/x}.log"
export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p outputs/final_replication logs
echo "CELL_START set=$SET_ID prompt=$PROMPT_ID sides=$SIDES $(date -Is)" >> "$LOG"
"$PYTHON" scripts/run_vlm_f2_matrix_v3.py \
  --input-a data/processed/main400_remote_public.jsonl \
  --input-b data/processed/main400_hashblind_set_b_public.jsonl \
  --output-dir outputs/final_replication \
  --model "$MODEL" \
  --image-root "$ROOT" \
  --run-id f2_qwen8_20260805_parallel \
  --sets "$SET_ID" \
  --prompts "$PROMPT_ID" \
  --sides "$SIDES" \
  --max-new-tokens 192 \
  --skip-existing >> "$LOG" 2>&1
echo "CELL_END set=$SET_ID prompt=$PROMPT_ID sides=$SIDES $(date -Is)" >> "$LOG"
