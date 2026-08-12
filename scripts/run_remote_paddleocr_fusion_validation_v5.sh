#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: run_remote_paddleocr_fusion_validation_v5.sh 29|31}"
if [[ "$SEED" != "29" && "$SEED" != "31" ]]; then
  echo "seed must be 29 or 31" >&2
  exit 2
fi

PROJECT_ROOT="/home/hera/pid_reliability_benchmark"
PYTHON_BIN="$PROJECT_ROOT/.venv_paddleocr_v1/bin/python"
INPUT_FILE="$PROJECT_ROOT/data/processed/source_seed${SEED}_resolution_v1_remote_public.jsonl"
OUTPUT_FILE="$PROJECT_ROOT/outputs/positive_narrative/paddleocr_seed${SEED}_v1.jsonl"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT_FILE")" logs
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/run_paddleocr_value_baseline_v1.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --image-root "$PROJECT_ROOT" \
  --run-id "paddleocr_fusion_validation_seed${SEED}_v1_20260811" \
  --det-limit-side-len 3072 \
  --skip-existing
