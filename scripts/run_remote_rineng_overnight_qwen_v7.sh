#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
GPU=${1:?GPU index required}
MODEL_LABEL=${2:?model label required}
MODEL_REL=${3:?model path relative to project root required}
OUTPUT="$ROOT/outputs/rineng_overnight_v7/$MODEL_LABEL"

if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+'; then
  echo "REFUSING_BUSY_GPU gpu=$GPU $(date -Is)" >&2
  exit 3
fi

mkdir -p "$OUTPUT" "$ROOT/logs"
rm -f "$OUTPUT/FINISHED" "$OUTPUT/COMPLETE" "$OUTPUT/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/vendor:$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

echo "START model=$MODEL_LABEL gpu=$GPU $(date -Is)"
set +e
"$PYTHON" scripts/run_qwen_counterfactual_prompt_matrix_v7.py \
  --root "$ROOT" \
  --plan data/manifests/rineng_overnight_v7_public_plan.json \
  --model "$MODEL_REL" \
  --model-label "$MODEL_LABEL" \
  --output-dir "outputs/rineng_overnight_v7/$MODEL_LABEL" \
  --run-id "${MODEL_LABEL}_overnight_v7_20260812" \
  --prompts p0,p1 \
  --conditions correct,shuffled,text_only \
  --max-image-side 3072 \
  --max-new-tokens 512 \
  --skip-existing
status=$?
set -e
touch "$OUTPUT/FINISHED"
if [[ "$status" -eq 0 ]]; then
  touch "$OUTPUT/COMPLETE"
  echo "END status=pass model=$MODEL_LABEL gpu=$GPU $(date -Is)"
else
  touch "$OUTPUT/FAILED"
  echo "END status=fail code=$status model=$MODEL_LABEL gpu=$GPU $(date -Is)"
fi
exit "$status"

