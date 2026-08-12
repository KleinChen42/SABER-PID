#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/download_remote_internvl35_8b_modelscope.sh"
sed \
  -e 's/b7f9c784a27d30ddb3fc78fca353ab6ce982c2a263b702e2547302e8c1e0087a/b7f9c784a27f30ddb3fc78fca353ab6ce982c2a263b702e2547302e8c1e0087a/' \
  -e 's/8c33a67f6ea2cc5c35fd631a763d70b52e4b132e12ca331c1767c0a95326840 3225776168/8c33a67f6ea2cc5c35fd631a763d70b52e4b132e12ca331c1767c0a95326840e 3225776168/' \
  "$SCRIPT" | bash
