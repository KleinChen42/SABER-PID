#!/usr/bin/env bash
# Resumable E4 launcher.  It is intentionally not invoked until E2/E3 finish.
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
MODEL="$ROOT/models/InternVL3_5-8B-modelscope"
INPUT=${1:-data/processed/main400_hashblind_set_b_remote_public.jsonl}
LOG="$ROOT/logs/evidence_internvl_tile_budget_v1.log"

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p outputs/evidence_strengthening/internvl_tile_budget_v1 logs

echo "START E4 $(date -Is)" | tee -a "$LOG"
"$PYTHON" scripts/run_internvl_tile_budget_v1.py \
  --input "$INPUT" \
  --output-dir outputs/evidence_strengthening/internvl_tile_budget_v1 \
  --model "$MODEL" \
  --image-root "$ROOT" \
  --run-id internvl_tile_budget_v1 \
  --set-id B \
  --low-max-num 1 \
  --high-max-num 12 \
  --max-new-tokens 192 \
  --skip-existing 2>&1 | tee -a "$LOG"
echo "END E4 $(date -Is)" | tee -a "$LOG"
