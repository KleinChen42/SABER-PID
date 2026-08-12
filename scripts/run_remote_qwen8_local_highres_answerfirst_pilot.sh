#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <input-jsonl> <output-jsonl> [max-samples]" >&2
  exit 64
fi

INPUT=$1
OUTPUT=$2
MAX_SAMPLES=${3:-0}
ROOT=/home/hera/pid_reliability_benchmark
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
LOG="$ROOT/logs/$(basename "$OUTPUT").log"

mkdir -p "$ROOT/logs" "$(dirname "$ROOT/$OUTPUT")"
exec >>"$LOG" 2>&1
echo "START $(date -Is) model=$MODEL mode=structured-answer-first input=$INPUT output=$OUTPUT max_image_side=3072"

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"

exec "$PYTHON" scripts/run_vlm_pilot_answer_first.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --mode structured \
  --max-samples "$MAX_SAMPLES" \
  --max-image-side 3072 \
  --max-new-tokens 192 \
  --resume
