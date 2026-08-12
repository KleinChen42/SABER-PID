#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hera/pid_reliability_benchmark"
VENV_DIR="$PROJECT_ROOT/.venv_paddleocr_v1"
LOCK_FILE="$PROJECT_ROOT/reports/generated/paddleocr_environment_v1.txt"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade "pip==24.3.1"
"$VENV_DIR/bin/python" -m pip install "paddlepaddle==2.6.2" "paddleocr==2.8.1"
mkdir -p "$PROJECT_ROOT/reports/generated"
"$VENV_DIR/bin/python" -m pip freeze > "$LOCK_FILE"
"$VENV_DIR/bin/python" -c "import paddle, paddleocr; print('paddle', paddle.__version__); print('paddleocr', paddleocr.__version__)"
