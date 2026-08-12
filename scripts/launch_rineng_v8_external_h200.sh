#!/usr/bin/env bash
set -euo pipefail

ROOT="${RINENG_ROOT:-/home/hera/pid_reliability_benchmark}"
PUBLIC="${RINENG_PUBLIC_V8_ROOT:-/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812}"
GPU_INDEX="${RINENG_EXTERNAL_GPU_INDEX:-1}"
PY_GPU="${RINENG_PY_GPU:-/home/hera/EACL_v92/.venv/bin/python}"
PY_OCR="${RINENG_PY_OCR:-$ROOT/.venv_paddleocr_v1/bin/python}"

cd "$ROOT"
mkdir -p "$PUBLIC/logs" "$PUBLIC/outputs/dexpi_external_qwen" "$PUBLIC/reports"
exec > >(tee -a "$PUBLIC/logs/dexpi_external_full.log") 2>&1

echo "DEXPI_WATCH_START $(date -u --iso-8601=seconds)"
while screen -ls 2>/dev/null | grep -q '[.]rie_v8_mainline'; do
  sleep 30
done

# The external branch is deliberately queued behind the long mainline.  It
# starts only when the selected physical GPU is actually unoccupied; a model
# retained by another user therefore delays this branch rather than causing
# memory contention.
while true; do
  used="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  util="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$used" -lt 4096 && "$util" -lt 10 ]]; then
    break
  fi
  echo "DEXPI_WAIT_GPU $(date -u --iso-8601=seconds) gpu=$GPU_INDEX memory_mib=$used util_pct=$util"
  sleep 30
done

echo "DEXPI_QWEN_START $(date -u --iso-8601=seconds) gpu=$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$ROOT/.v8_site:$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PY_GPU" scripts/run_qwen_counterfactual_quality_v8.py \
  --root "$ROOT" \
  --plan data/manifests/rineng_v8_dexpi_external_plan.json \
  --model models/Qwen3-VL-8B-Instruct-modelscope \
  --model-label qwen3vl8b \
  --output-dir "$PUBLIC/outputs/dexpi_external_qwen" \
  --run-id rineng-v8-dexpi-external-3072-cap512 \
  --prompts p0 \
  --conditions correct,shuffled,text_only \
  --max-image-side 3072 \
  --max-new-tokens 512 \
  --skip-existing
echo "DEXPI_QWEN_COMPLETE $(date -u --iso-8601=seconds)"

# OCR normally finishes while the GPU mainline is running.  If it did not,
# resume it here before scorer-only references are opened.
"$PY_OCR" scripts/run_paddleocr_external_v8.py \
  --input data/processed/rineng_v8_dexpi_external/dexpi_external_v8_correct_public.jsonl \
  --plan data/manifests/rineng_v8_dexpi_external_plan.json \
  --output "$PUBLIC/outputs/dexpi_external_ocr.jsonl" \
  --run-id rineng-v8-dexpi-paddleocr \
  --det-limit-side-len 3072 \
  --skip-existing

"$PY_GPU" scripts/score_dexpi_external_v8.py \
  --root "$ROOT" \
  --plan data/manifests/rineng_v8_dexpi_external_plan.json \
  --qwen-output-root "$PUBLIC/outputs/dexpi_external_qwen" \
  --ocr-output "$PUBLIC/outputs/dexpi_external_ocr.jsonl" \
  --output "$PUBLIC/reports/rineng_v8_dexpi_external_score.json" \
  --csv "$PUBLIC/reports/rineng_v8_dexpi_external_score.csv" \
  --bootstrap-reps 10000
echo "DEXPI_EXTERNAL_COMPLETE $(date -u --iso-8601=seconds)"
