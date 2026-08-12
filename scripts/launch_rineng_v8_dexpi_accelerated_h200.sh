#!/usr/bin/env bash
set -euo pipefail

ROOT="${RINENG_ROOT:-/home/hera/pid_reliability_benchmark}"

# The two seed31 shards own GPU 6 and GPU 7 and write condition-disjoint files.
# Start the external-family branch only after both have exited, then let the
# regular launcher independently verify that GPU 7 is actually idle.
while screen -ls 2>/dev/null | grep -qE '[.]rie_v8_q31_(correct|shuffled)'; do
  sleep 15
done

export RINENG_EXTERNAL_GPU_INDEX="${RINENG_EXTERNAL_GPU_INDEX:-7}"
export RINENG_WAIT_FOR_MAINLINE=0
exec "$ROOT/scripts/launch_rineng_v8_external_h200.sh"
