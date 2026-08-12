#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hera/pid_reliability_benchmark"
PYTHON_BIN="$PROJECT_ROOT/.venv_paddleocr_v1/bin/python"
OUTPUT_FILE="$PROJECT_ROOT/outputs/editorial_revision/paddleocr_value_baseline_v1/paddleocr_value_full_image.jsonl"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT_FILE")" logs
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/run_paddleocr_value_baseline_v1.py \
  --input data/processed/main400_hashblind_set_b_remote_public.jsonl \
  --output "$OUTPUT_FILE" \
  --image-root "$PROJECT_ROOT" \
  --run-id paddleocr_value_full_image_v1_20260811 \
  --det-limit-side-len 3072 \
  --skip-existing
