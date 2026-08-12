#!/usr/bin/env bash
# Launch one resumable answer-isolated Qwen evidence condition on H200.
#
# Usage:
#   run_remote_qwen_evidence_condition.sh INPUT CONDITION TASKS SIDES TOKENS [LEGEND] [MODE]
# Examples:
#   ... data/processed/main400_hashblind_set_b_remote_public.jsonl value_budget_v1 value 768,3072 512
#   ... data/processed/main400_hashblind_set_b_shuffled_v1_public.jsonl image_shuffle_v1 '' 768,3072 192
#   ... data/processed/main400_hashblind_set_b_remote_public.jsonl text_only_v1 connectivity,count,spatial_count,value 3072 192 - text_only
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
MODEL="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
INPUT=${1:?answer-isolated input JSONL required}
CONDITION=${2:?condition identifier required}
TASKS=${3:-}
SIDES=${4:-768,3072}
TOKENS=${5:-192}
LEGEND=${6:-}
MODE=${7:-image}
if [[ "$LEGEND" == "-" ]]; then
  LEGEND=""
fi
LOG="$ROOT/logs/evidence_${CONDITION}.log"

export PYTHONPATH="$ROOT/vendor:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
cd "$ROOT"
mkdir -p "outputs/evidence_strengthening/$CONDITION" logs

args=(
  scripts/run_qwen_evidence_matrix.py
  --input "$INPUT"
  --output-dir "outputs/evidence_strengthening/$CONDITION"
  --model "$MODEL"
  --image-root "$ROOT"
  --run-id "${CONDITION}_$(date -u +%Y%m%dT%H%M%SZ)"
  --condition-id "$CONDITION"
  --set-id B
  --prompt p0
  --tasks "$TASKS"
  --sides "$SIDES"
  --max-new-tokens "$TOKENS"
  --skip-existing
)
if [[ -n "$LEGEND" ]]; then
  args+=(--legend-image "$LEGEND")
fi
case "$MODE" in
  image)
    ;;
  text_only)
    if [[ -n "$LEGEND" ]]; then
      echo "text_only mode does not accept a legend image" >&2
      exit 2
    fi
    args+=(--text-only)
    ;;
  *)
    echo "unsupported mode: $MODE (expected image or text_only)" >&2
    exit 2
    ;;
esac

echo "START condition=$CONDITION input=$INPUT tasks=$TASKS sides=$SIDES tokens=$TOKENS mode=$MODE $(date -Is)" | tee -a "$LOG"
"$PYTHON" "${args[@]}" 2>&1 | tee -a "$LOG"
echo "END condition=$CONDITION $(date -Is)" | tee -a "$LOG"
