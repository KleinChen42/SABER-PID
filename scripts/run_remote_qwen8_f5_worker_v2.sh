#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p outputs/final_degradation logs
for condition in "$@"; do
  input="data/processed/final_degradation/main400_hashblind_set_b_${condition}_public.jsonl"
  output="outputs/final_degradation/qwen8_set_b_${condition}.jsonl"
  echo "START worker=$$ condition=$condition $(date -Is)" >> "logs/f5_qwen8_worker_${$}.log"
  "$PYTHON" scripts/run_vlm_degradation_matrix.py --input "$input" --output "$output" --model "$MODEL" --image-root "$ROOT" --condition "$condition" --run-id f5_qwen8_set_b_1536_20260805_parallel --max-image-side 1536 --max-new-tokens 192 --resume >> "logs/f5_qwen8_worker_${$}.log" 2>&1
  echo "END worker=$$ condition=$condition $(date -Is)" >> "logs/f5_qwen8_worker_${$}.log"
done
