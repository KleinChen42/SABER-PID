#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
MODEL="$ROOT/models/InternVL3_5-8B-modelscope"
export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p outputs/final_replication logs
exec "$PYTHON" scripts/run_internvl35_f3_matrix_v4.py \
  --input-a data/processed/main400_remote_public.jsonl \
  --input-b data/processed/main400_hashblind_set_b_remote_public.jsonl \
  --output-dir outputs/final_replication \
  --model "$MODEL" \
  --image-root "$ROOT" \
  --run-id f3_internvl35_8b_20260805_v4 \
  --sets A,B --sides 768,3072 --max-new-tokens 192 --skip-existing
