#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hera/pid_reliability_benchmark"
PYTHON_BIN="/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python"
MODEL_DIR="$PROJECT_ROOT/models/InternVL3_5-8B-modelscope"
OUTPUT_DIR="$PROJECT_ROOT/outputs/editorial_revision/internvl_counterfactual_ladder_v2_tokenizerfix"
RUN_CONDITIONS="${SABER_LADDER_CONDITIONS:-correct,shuffled,text_only}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/vendor:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${SABER_LADDER_GPU:-7}"

"$PYTHON_BIN" scripts/run_internvl_counterfactual_ladder_v1.py \
  --correct-input data/processed/main400_hashblind_set_b_remote_public.jsonl \
  --shuffled-input data/processed/main400_hashblind_set_b_shuffled_v1_remote_public.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_DIR" \
  --image-root "$PROJECT_ROOT" \
  --run-id editorial_revision_internvl8_ladder_tokenizerfix_20260811 \
  --conditions "$RUN_CONDITIONS" \
  --max-num 12 \
  --max-new-tokens 512 \
  --skip-existing
