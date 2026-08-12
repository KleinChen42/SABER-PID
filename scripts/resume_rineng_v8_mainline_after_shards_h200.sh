#!/usr/bin/env bash
set -euo pipefail

MAINLINE_SHELL_PID="${1:?usage: resume_rineng_v8_mainline_after_shards_h200.sh MAINLINE_SHELL_PID}"
PUBLIC="${RINENG_PUBLIC_V8_ROOT:-/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812}"
CONTROL="$PUBLIC/control/internvl_mainline_barrier.txt"
mkdir -p "$PUBLIC/logs" "$PUBLIC/control"
exec > >(tee -a "$PUBLIC/logs/mainline_shard_resume.log") 2>&1
echo "MAINLINE_SHARD_WAIT_START $(date -u --iso-8601=seconds) pid=$MAINLINE_SHELL_PID"

while screen -ls 2>/dev/null | grep -qE '[.]rie_v8_(i29|i31|i_setb_correct|i_setb_shuffled)'; do
  sleep 30
done

printf 'GO\n' > "$CONTROL"
echo "MAINLINE_SHARD_BARRIER_RELEASED $(date -u --iso-8601=seconds) control=$CONTROL"

if ! kill -0 "$MAINLINE_SHELL_PID" 2>/dev/null; then
  echo "MAINLINE_SHELL_ABSENT $(date -u --iso-8601=seconds) pid=$MAINLINE_SHELL_PID"
  exit 3
fi

state="$(ps -o stat= -p "$MAINLINE_SHELL_PID" | tr -d ' ')"
if [[ "$state" == *T* ]]; then
  kill -CONT "$MAINLINE_SHELL_PID"
  echo "MAINLINE_SHELL_RESUMED $(date -u --iso-8601=seconds) pid=$MAINLINE_SHELL_PID"
else
  echo "MAINLINE_SHELL_RUNNING $(date -u --iso-8601=seconds) pid=$MAINLINE_SHELL_PID state=$state"
fi
