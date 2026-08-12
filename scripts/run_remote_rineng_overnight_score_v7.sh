#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
PYTHON=/data/hera-1105/RINENG/environments/anomalib-fadc6203c067/bin/python
LOG="$ROOT/logs/rineng_overnight_v7_score_rerun.log"
COMPLETE_MARKER="$ROOT/reports/generated/RINENG_OVERNIGHT_V7_SCORE_COMPLETE"
FAILED_MARKER="$ROOT/reports/generated/RINENG_OVERNIGHT_V7_SCORE_FAILED"
STAMP=$(date +%Y%m%dT%H%M%S%z)

export PYTHONPATH="$ROOT/vendor:$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0

cd "$ROOT"
mkdir -p logs reports/generated

if [[ -e "$FAILED_MARKER" ]]; then
  mv "$FAILED_MARKER" "${FAILED_MARKER}.previous_${STAMP}"
fi
if [[ -e "$COMPLETE_MARKER" ]]; then
  mv "$COMPLETE_MARKER" "${COMPLETE_MARKER}.previous_${STAMP}"
fi

echo "SCORE_RERUN_START $STAMP" >"$LOG"
set +e
"$PYTHON" scripts/score_rineng_overnight_v7.py \
  --root "$ROOT" \
  --bootstrap-reps 10000 \
  >>"$LOG" 2>&1
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  touch "$COMPLETE_MARKER"
  echo "SCORE_RERUN_END status=pass $(date -Is)" >>"$LOG"
else
  touch "$FAILED_MARKER"
  echo "SCORE_RERUN_END status=fail code=$status $(date -Is)" >>"$LOG"
fi

exit "$status"
