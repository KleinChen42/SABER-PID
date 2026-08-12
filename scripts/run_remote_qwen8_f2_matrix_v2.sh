#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/run_remote_qwen8_f2_matrix.sh"
sed 's/run_vlm_f2_matrix_v2.py/run_vlm_f2_matrix_v3.py/' "$SCRIPT" | bash
