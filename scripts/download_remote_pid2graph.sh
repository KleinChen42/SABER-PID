#!/usr/bin/env bash
set -euo pipefail

# Public PID2Graph archive acquisition for the E4 external-source check.
# Extraction is intentionally a separate, inspected step because the archive
# contains several dataset families while this project uses OPEN100 only.
ROOT=/home/hera/pid_reliability_benchmark
TARGET_DIR="$ROOT/data/raw/pid2graph"
ARCHIVE="$TARGET_DIR/PID2Graph.zip"
LOG="$ROOT/logs/pid2graph_download.log"
URL='https://zenodo.org/records/14803338/files/PID2Graph.zip?download=1'

mkdir -p "$TARGET_DIR" "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "START $(date -Is) url=$URL archive=$ARCHIVE"
aria2c \
  --continue=true \
  --file-allocation=none \
  --auto-file-renaming=false \
  --console-log-level=warn \
  --summary-interval=60 \
  --max-connection-per-server=8 \
  --split=8 \
  --dir="$TARGET_DIR" \
  --out='PID2Graph.zip' \
  "$URL"
echo "COMPLETE $(date -Is)"
