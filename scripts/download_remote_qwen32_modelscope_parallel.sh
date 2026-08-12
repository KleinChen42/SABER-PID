#!/usr/bin/env bash
set -euo pipefail

# Project-local download: do not modify the shared H200 environment or model
# cache.  Four simultaneous shards keep the request load comparable to the
# already validated 8B acquisition path.
ROOT=/home/hera/pid_reliability_benchmark
TARGET="$ROOT/models/Qwen3-VL-32B-Instruct-modelscope"
LOG="$ROOT/logs/qwen3vl32b_modelscope_download.log"
BASE=https://www.modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct/resolve/master

mkdir -p "$TARGET" "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "RESTART $(date -Is) ModelScope parallel Qwen3-VL-32B download target=$TARGET"

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

shard_number=0
for shard in \
  model-00001-of-00014.safetensors \
  model-00002-of-00014.safetensors \
  model-00003-of-00014.safetensors \
  model-00004-of-00014.safetensors \
  model-00005-of-00014.safetensors \
  model-00006-of-00014.safetensors \
  model-00007-of-00014.safetensors \
  model-00008-of-00014.safetensors \
  model-00009-of-00014.safetensors \
  model-00010-of-00014.safetensors \
  model-00011-of-00014.safetensors \
  model-00012-of-00014.safetensors \
  model-00013-of-00014.safetensors \
  model-00014-of-00014.safetensors; do
  fetch_one "$shard" &
  shard_number=$((shard_number + 1))
  if (( shard_number % 4 == 0 )); then
    wait
  fi
done
wait

echo "COMPLETE $(date -Is)"
