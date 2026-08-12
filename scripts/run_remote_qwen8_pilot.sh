#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <input-jsonl> <output-jsonl> <direct|structured> [max-samples]" >&2
  exit 64
fi

INPUT=$1
OUTPUT=$2
MODE=$3
MAX_SAMPLES=${4:-0}
ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
LOG="$ROOT/logs/$(basename "$OUTPUT").log"

mkdir -p "$ROOT/logs" "$(dirname "$ROOT/$OUTPUT")"
exec >>"$LOG" 2>&1
echo "START $(date -Is) model=Qwen/Qwen3-VL-8B-Instruct mode=$MODE input=$INPUT output=$OUTPUT"

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
cd "$ROOT"

exec "$PYTHON" scripts/run_vlm_pilot.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --mode "$MODE" \
  --max-samples "$MAX_SAMPLES" \
  --max-image-side 1536 \
  --max-new-tokens 192 \
  --resume
