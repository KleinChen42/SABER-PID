#!/usr/bin/env bash
set -euo pipefail

ROOT="${RINENG_ROOT:-/home/hera/pid_reliability_benchmark}"
ACTION="${1:?usage: launch_rineng_v8_h200.sh ACTION}"
PY_CPU="${RINENG_PY_CPU:-/usr/bin/python3}"
PY_GPU="${RINENG_PY_GPU:-/home/hera/EACL_v92/.venv/bin/python}"
PUBLIC_V8_ROOT="${RINENG_PUBLIC_V8_ROOT:-/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812}"
cd "$ROOT"
mkdir -p reports/logs outputs/rineng_v8 data/external/pid2graph_v8
mkdir -p "$PUBLIC_V8_ROOT/outputs"

case "$ACTION" in
  quality_prepare)
    exec "$PY_CPU" scripts/prepare_quality_robustness_v8.py --root "$ROOT"
    ;;
  quality_full)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export PYTHONPATH="$ROOT/.v8_site${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PY_GPU" scripts/run_qwen_counterfactual_quality_v8.py \
      --root "$ROOT" \
      --plan data/manifests/rineng_v8_quality_robustness_plan.json \
      --model models/Qwen3-VL-8B-Instruct-modelscope \
      --model-label qwen3vl8b \
      --output-dir "$PUBLIC_V8_ROOT/outputs/qwen3vl8b_quality" \
      --run-id rineng-v8-quality-3072-cap512 \
      --prompts p0 \
      --conditions correct,shuffled \
      --max-image-side 3072 \
      --max-new-tokens 512 \
      --skip-existing
    ;;
  quality_smoke)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    export PYTHONPATH="$ROOT/.v8_site${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PY_GPU" scripts/run_qwen_counterfactual_quality_v8.py \
      --root "$ROOT" \
      --plan data/manifests/rineng_v8_quality_robustness_plan.json \
      --model models/Qwen3-VL-8B-Instruct-modelscope \
      --model-label qwen3vl8b \
      --output-dir outputs/rineng_v8/qwen3vl8b_quality_smoke \
      --run-id rineng-v8-quality-smoke \
      --datasets set_b100__clean \
      --record-limit 1 \
      --prompts p0 \
      --conditions correct,shuffled \
      --max-image-side 3072 \
      --max-new-tokens 512 \
      --skip-existing
    ;;
  internvl_smoke)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    export PYTHONPATH="$ROOT/.v8_site${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PY_GPU" scripts/run_internvl_budget_matched_v8.py \
      --root "$ROOT" \
      --plan data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json \
      --model models/InternVL3_5-8B-modelscope \
      --model-label internvl35_8b_budget54 \
      --output-dir outputs/rineng_v8/internvl35_8b_budget54_smoke \
      --run-id rineng-v8-internvl-budget54-smoke \
      --conditions correct,shuffled,text_only \
      --record-limit 1 \
      --skip-existing
    ;;
  internvl_smoke_seed29)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    export PYTHONPATH="$ROOT/.v8_site${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PY_GPU" scripts/run_internvl_budget_matched_v8.py \
      --root "$ROOT" \
      --plan data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json \
      --model models/InternVL3_5-8B-modelscope \
      --model-label internvl35_8b_budget54 \
      --output-dir outputs/rineng_v8/internvl35_8b_budget54_smoke_seed29 \
      --run-id rineng-v8-internvl-budget54-smoke-seed29 \
      --datasets seed29_strict65 \
      --conditions correct,shuffled,text_only \
      --record-limit 1 \
      --skip-existing
    ;;
  internvl_full)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    export PYTHONPATH="$ROOT/.v8_site${PYTHONPATH:+:$PYTHONPATH}"
    for dataset in set_b100 seed29_strict65 seed31_strict65; do
      "$PY_GPU" scripts/run_internvl_budget_matched_v8.py \
        --root "$ROOT" \
        --plan data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json \
        --model models/InternVL3_5-8B-modelscope \
        --model-label internvl35_8b_budget54 \
        --output-dir "$PUBLIC_V8_ROOT/outputs/internvl35_8b_budget54" \
        --run-id "rineng-v8-internvl-budget54-$dataset" \
        --datasets "$dataset" \
        --conditions correct,shuffled,text_only \
        --skip-existing
    done
    ;;
  mainline_full)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    export PYTHONPATH="$ROOT/.v8_site${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$PUBLIC_V8_ROOT/logs"
    exec > >(tee -a "$PUBLIC_V8_ROOT/logs/mainline_full.log") 2>&1
    echo "MAINLINE_START $(date -u --iso-8601=seconds)"
    "$PY_GPU" scripts/run_qwen_counterfactual_quality_v8.py \
      --root "$ROOT" \
      --plan data/manifests/rineng_v8_quality_robustness_plan.json \
      --model models/Qwen3-VL-8B-Instruct-modelscope \
      --model-label qwen3vl8b \
      --output-dir "$PUBLIC_V8_ROOT/outputs/qwen3vl8b_quality" \
      --run-id rineng-v8-quality-3072-cap512 \
      --prompts p0 \
      --conditions correct,shuffled \
      --max-image-side 3072 \
      --max-new-tokens 512 \
      --skip-existing
    for dataset in set_b100 seed29_strict65 seed31_strict65; do
      "$PY_GPU" scripts/run_internvl_budget_matched_v8.py \
        --root "$ROOT" \
        --plan data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json \
        --model models/InternVL3_5-8B-modelscope \
        --model-label internvl35_8b_budget54 \
        --output-dir "$PUBLIC_V8_ROOT/outputs/internvl35_8b_budget54" \
        --run-id "rineng-v8-internvl-budget54-$dataset" \
        --datasets "$dataset" \
        --conditions correct,shuffled,text_only \
        --skip-existing
    done
    echo "MAINLINE_COMPLETE $(date -u --iso-8601=seconds)"
    ;;
  pid2graph_catalog)
    exec "$PY_CPU" scripts/fetch_remote_zip_subset.py \
      --url 'https://zenodo.org/records/14803338/files/PID2Graph.zip?download=1' \
      --total-size 9303633645 \
      --sparse-zip data/external/pid2graph_v8/PID2Graph_subset_sparse.zip \
      --catalog-json reports/generated/pid2graph_open100_remote_catalog_v8.json \
      --member-regex OPEN100
    ;;
  *)
    echo "Unknown ACTION: $ACTION" >&2
    exit 2
    ;;
esac
