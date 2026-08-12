#!/usr/bin/env bash
set -euo pipefail

ROOT="${RINENG_ROOT:-/home/hera/pid_reliability_benchmark}"
PUBLIC="${RINENG_PUBLIC_V8_ROOT:-/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812}"
PY_GPU="${RINENG_PY_GPU:-/home/hera/EACL_v92/.venv/bin/python}"
DATASET="${1:?usage: launch_rineng_v8_internvl_shard_h200.sh DATASET GPU_INDEX WAIT_SCREEN}"
GPU_INDEX="${2:?usage: launch_rineng_v8_internvl_shard_h200.sh DATASET GPU_INDEX WAIT_SCREEN}"
WAIT_SCREEN="${3:?usage: launch_rineng_v8_internvl_shard_h200.sh DATASET GPU_INDEX WAIT_SCREEN}"
CONDITIONS="${4:-correct,shuffled,text_only}"

case "$DATASET" in
  set_b100|seed29_strict65|seed31_strict65) ;;
  *) echo "Unsupported dataset shard: $DATASET" >&2; exit 2 ;;
esac
case "$GPU_INDEX" in
  ''|*[!0-9]*) echo "GPU index must be numeric" >&2; exit 2 ;;
esac

while screen -ls 2>/dev/null | grep -Fq ".$WAIT_SCREEN"; do
  sleep 15
done

while true; do
  used="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  util="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$used" -lt 4096 && "$util" -lt 10 ]]; then
    break
  fi
  sleep 15
done

cd "$ROOT"
mkdir -p "$PUBLIC/logs" "$PUBLIC/outputs/internvl35_8b_budget54"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$ROOT/.v8_site:$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee -a "$PUBLIC/logs/internvl_${DATASET}_gpu${GPU_INDEX}.log") 2>&1
echo "INTERNVL_SHARD_START $(date -u --iso-8601=seconds) dataset=$DATASET gpu=$GPU_INDEX"
"$PY_GPU" scripts/run_internvl_budget_matched_v8.py \
  --root "$ROOT" \
  --plan data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json \
  --model models/InternVL3_5-8B-modelscope \
  --model-label internvl35_8b_budget54 \
  --output-dir "$PUBLIC/outputs/internvl35_8b_budget54" \
  --run-id "rineng-v8-internvl-budget54-$DATASET" \
  --datasets "$DATASET" \
  --conditions "$CONDITIONS" \
  --skip-existing
echo "INTERNVL_SHARD_COMPLETE $(date -u --iso-8601=seconds) dataset=$DATASET gpu=$GPU_INDEX"
