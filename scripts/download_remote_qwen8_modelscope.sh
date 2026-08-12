#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
TARGET="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
LOG="$ROOT/logs/qwen3vl8b_modelscope_download.log"
BASE=https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master

mkdir -p "$TARGET" "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "START $(date -Is) ModelScope Qwen3-VL-8B download target=$TARGET"

aria2c \
  --continue=true \
  --file-allocation=none \
  --auto-file-renaming=false \
  --console-log-level=notice \
  --summary-interval=60 \
  --max-concurrent-downloads=4 \
  --max-connection-per-server=8 \
  --split=8 \
  --dir="$TARGET" \
  "$BASE/chat_template.json" \
  "$BASE/config.json" \
  "$BASE/merges.txt" \
  "$BASE/model.safetensors.index.json" \
  "$BASE/preprocessor_config.json" \
  "$BASE/tokenizer_config.json" \
  "$BASE/tokenizer.json" \
  "$BASE/video_preprocessor_config.json" \
  "$BASE/vocab.json" \
  "$BASE/model-00001-of-00004.safetensors" \
  "$BASE/model-00002-of-00004.safetensors" \
  "$BASE/model-00003-of-00004.safetensors" \
  "$BASE/model-00004-of-00004.safetensors"

echo "COMPLETE $(date -Is)"
