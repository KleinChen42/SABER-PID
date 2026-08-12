#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
MODELS=(qwen3vl8b qwen3vl32b internvl35_8b)
LOG="$ROOT/logs/rineng_overnight_v7_score_watch.log"
export PYTHONPATH="$ROOT/vendor:$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p reports/generated logs

echo "WATCH_START $(date -Is)" >>"$LOG"
while true; do
  pending=0
  for model in "${MODELS[@]}"; do
    if [[ ! -f "outputs/rineng_overnight_v7/$model/FINISHED" ]]; then
      pending=$((pending + 1))
    fi
  done
  echo "WATCH_TICK $(date -Is) pending=$pending" >>"$LOG"
  if [[ "$pending" -eq 0 ]]; then
    break
  fi
  sleep 300
done

set +e
"$PYTHON" scripts/score_rineng_overnight_v7.py --root "$ROOT" \
  >>"$LOG" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  touch reports/generated/RINENG_OVERNIGHT_V7_SCORE_COMPLETE
  echo "WATCH_END status=pass $(date -Is)" >>"$LOG"
else
  touch reports/generated/RINENG_OVERNIGHT_V7_SCORE_FAILED
  echo "WATCH_END status=fail code=$status $(date -Is)" >>"$LOG"
fi
exit "$status"

