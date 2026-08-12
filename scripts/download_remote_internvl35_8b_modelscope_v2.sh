#!/usr/bin/env bash
set -euo pipefail

# Reuse the audited downloader while correcting the one-character checksum
# typo in the first draft.  The expected digest is the ModelScope API digest.
SCRIPT="$(cd "$(dirname "$0")" && pwd)/download_remote_internvl35_8b_modelscope.sh"
sed 's/b7f9c784a27d30ddb3fc78fca353ab6ce982c2a263b702e2547302e8c1e0087a/b7f9c784a27f30ddb3fc78fca353ab6ce982c2a263b702e2547302e8c1e0087a/' "$SCRIPT" | bash
