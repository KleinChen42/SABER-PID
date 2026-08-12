#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
TARGET="$ROOT/models/Qwen3-VL-8B-Instruct-modelscope"
LOG="$ROOT/logs/qwen3vl8b_modelscope_download.log"
BASE=https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master

mkdir -p "$TARGET" "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "RESTART $(date -Is) ModelScope parallel Qwen3-VL-8B download target=$TARGET"

fetch_one() {
  local file=$1
  aria2c \
    --continue=true \
    --file-allocation=none \
    --auto-file-renaming=false \
    --console-log-level=warn \
    --summary-interval=60 \
    --max-connection-per-server=8 \
    --split=8 \
    --dir="$TARGET" \
    --out="$file" \
    "$BASE/$file"
}

for file in \
  chat_template.json config.json merges.txt model.safetensors.index.json \
  preprocessor_config.json tokenizer_config.json tokenizer.json \
  video_preprocessor_config.json vocab.json; do
  fetch_one "$file"
done

for shard in \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors; do
  fetch_one "$shard" &
done
wait

echo "COMPLETE $(date -Is)"
