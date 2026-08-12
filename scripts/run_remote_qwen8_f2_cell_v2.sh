#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/run_remote_qwen8_f2_cell.sh"
sed 's/main400_hashblind_set_b_public.jsonl/main400_hashblind_set_b_remote_public.jsonl/' "$SCRIPT" | bash
