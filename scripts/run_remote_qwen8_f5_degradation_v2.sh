#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p outputs/final_degradation logs

run_cell() {
  local condition="$1" input="$2" output="$3"
  echo "START condition=$condition $(date -Is)" >> logs/f5_qwen8_1536.log
  "$PYTHON" scripts/run_vlm_degradation_matrix.py \
    --input "$input" --output "$output" --model "$MODEL" --image-root "$ROOT" \
    --condition "$condition" --run-id f5_qwen8_set_b_1536_20260805 \
    --max-image-side 1536 --max-new-tokens 192 --resume >> logs/f5_qwen8_1536.log 2>&1
  echo "END condition=$condition $(date -Is)" >> logs/f5_qwen8_1536.log
}

# The clean Set-B anchor is measured explicitly so every degradation delta is
# paired on the same hidden-answer question/source set.
run_cell clean data/processed/main400_hashblind_set_b_remote_public.jsonl outputs/final_degradation/qwen8_set_b_clean.jsonl
for condition in blur_r1 blur_r2 blur_r4 jpeg_q70 jpeg_q35 jpeg_q15 downsample_s075 downsample_s050 downsample_s025; do
  run_cell "$condition" "data/processed/final_degradation/main400_hashblind_set_b_${condition}_public.jsonl" "outputs/final_degradation/qwen8_set_b_${condition}.jsonl"
done
