#!/usr/bin/env bash
set -euo pipefail

# Create a self-contained, checksum-verified H200 backup set on the shared disk.
# The local Windows workspace is uploaded separately as local_windows_state.tar.zst
# and finalized with scripts/finalize_complete_public_backup_v10.sh.

PROJECT_ROOT="${1:-/home/hera/pid_reliability_benchmark}"
ENV_ROOT="${2:-/data/hera-1105/RINENG/environments/anomalib-fadc6203c067}"
ACTIVE_ROOT="${3:-/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812}"
BACKUP_ROOT="${4:-/kwkj-k8s/hera_pid_reliability_backups}"
STAMP="${5:-$(date -u +%Y%m%dT%H%M%SZ)}"

PROJECT_ROOT="$(realpath "$PROJECT_ROOT")"
ENV_ROOT="$(realpath "$ENV_ROOT")"
ACTIVE_ROOT="$(realpath "$ACTIVE_ROOT")"
BACKUP_ROOT="$(realpath -m "$BACKUP_ROOT")"

[[ "$PROJECT_ROOT" == /home/hera/pid_reliability_benchmark ]] || {
  echo "Refusing unexpected project root: $PROJECT_ROOT" >&2; exit 2;
}
[[ "$ENV_ROOT" == /data/hera-1105/RINENG/environments/anomalib-fadc6203c067 ]] || {
  echo "Refusing unexpected environment root: $ENV_ROOT" >&2; exit 2;
}
[[ "$ACTIVE_ROOT" == /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812 ]] || {
  echo "Refusing unexpected active-results root: $ACTIVE_ROOT" >&2; exit 2;
}
[[ "$BACKUP_ROOT" == /kwkj-k8s/* ]] || {
  echo "Refusing destination outside /kwkj-k8s: $BACKUP_ROOT" >&2; exit 2;
}

DEST="$BACKUP_ROOT/saber_pid_complete_${STAMP}"
mkdir -p "$BACKUP_ROOT"
if [[ -e "$DEST" ]]; then
  [[ -d "$DEST" ]] || { echo "Existing destination is not a directory: $DEST" >&2; exit 2; }
  [[ ! -e "$DEST/REMOTE_COMPONENTS_COMPLETE" ]] || {
    echo "Remote backup is already complete: $DEST" >&2; exit 2;
  }
  if compgen -G "$DEST/project_*.tar.zst" >/dev/null \
    || compgen -G "$DEST/shared_python_environment.tar.zst" >/dev/null \
    || compgen -G "$DEST/active_public_results.tar.zst" >/dev/null; then
    echo "Refusing to overwrite completed remote archives in: $DEST" >&2
    exit 2
  fi
else
  mkdir "$DEST"
fi
LOG="$DEST/backup.log"
exec > >(tee -a "$LOG") 2>&1

echo "Backup started: $(date -u --iso-8601=seconds)"
echo "Destination: $DEST"

mkdir -p "$DEST/environment_metadata"
{
  echo "created_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "user=$(whoami)"
  echo "kernel=$(uname -srmo)"
  command -v lsb_release >/dev/null && lsb_release -a 2>/dev/null || true
} > "$DEST/environment_metadata/system.txt"

{
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
  echo
  nvidia-smi
} > "$DEST/environment_metadata/nvidia_smi.txt" 2>&1 || true
nvcc --version > "$DEST/environment_metadata/nvcc.txt" 2>&1 || true
gcc --version > "$DEST/environment_metadata/gcc.txt" 2>&1 || true
git --version > "$DEST/environment_metadata/git.txt" 2>&1 || true
df -h "$PROJECT_ROOT" "$ENV_ROOT" "$BACKUP_ROOT" > "$DEST/environment_metadata/filesystems.txt"
du -sh "$PROJECT_ROOT" "$PROJECT_ROOT/models" "$ENV_ROOT" "$ACTIVE_ROOT" \
  > "$DEST/environment_metadata/source_sizes.txt"

"$ENV_ROOT/bin/python" --version > "$DEST/environment_metadata/python_version.txt" 2>&1
if "$ENV_ROOT/bin/python" -m pip --version >/dev/null 2>&1; then
  "$ENV_ROOT/bin/python" -m pip freeze --all > "$DEST/environment_metadata/pip_freeze.txt"
  "$ENV_ROOT/bin/python" -m pip list --format=json > "$DEST/environment_metadata/pip_list.json"
else
  echo "pip module unavailable; generated from importlib.metadata" \
    > "$DEST/environment_metadata/pip_freeze.txt"
  "$ENV_ROOT/bin/python" - <<'PY' >> "$DEST/environment_metadata/pip_freeze.txt"
from importlib.metadata import distributions
for dist in sorted(distributions(), key=lambda item: (item.metadata.get("Name") or "").lower()):
    print(f"{dist.metadata.get('Name') or 'UNKNOWN'}=={dist.version}")
PY
  "$ENV_ROOT/bin/python" - <<'PY' > "$DEST/environment_metadata/pip_list.json"
import json
from importlib.metadata import distributions
rows = [
    {"name": dist.metadata.get("Name") or "UNKNOWN", "version": dist.version}
    for dist in distributions()
]
print(json.dumps(sorted(rows, key=lambda row: row["name"].lower()), indent=2))
PY
fi
cp "$ENV_ROOT/pyvenv.cfg" "$DEST/environment_metadata/pyvenv.cfg" 2>/dev/null || true
"$ENV_ROOT/bin/python" - <<'PY' > "$DEST/environment_metadata/python_runtime.json"
import json, platform, sys

payload = {
    "executable": sys.executable,
    "python": sys.version,
    "platform": platform.platform(),
}
for name in ("torch", "transformers", "accelerate", "numpy", "PIL", "paddle", "paddleocr"):
    try:
        module = __import__(name)
        payload[name] = getattr(module, "__version__", "installed-version-not-exposed")
    except Exception as exc:
        payload[name] = f"unavailable: {type(exc).__name__}: {exc}"
try:
    import torch
    payload["torch_cuda_version"] = torch.version.cuda
    payload["cuda_available"] = torch.cuda.is_available()
    payload["cudnn_version"] = torch.backends.cudnn.version()
except Exception:
    pass
print(json.dumps(payload, indent=2, sort_keys=True))
PY

find "$PROJECT_ROOT/models" -xdev -type f -printf '%P\t%s\n' | sort \
  > "$DEST/environment_metadata/model_file_inventory.tsv"
find "$PROJECT_ROOT" -xdev -type f -printf '%P\t%s\n' | sort \
  > "$DEST/environment_metadata/project_file_inventory.tsv"
find "$ENV_ROOT" -xdev -type f -printf '%P\t%s\n' | sort \
  > "$DEST/environment_metadata/environment_file_inventory.tsv"
find "$ACTIVE_ROOT" -xdev -type f -printf '%P\t%s\n' | sort \
  > "$DEST/environment_metadata/active_results_file_inventory.tsv"

finalize_archive() {
  local stem="$1"
  local archive="$DEST/${stem}.tar.zst"
  local partial="${archive}.partial"
  zstd -q -t "$partial"
  tar --zstd -tf "$partial" > "${archive}.members.txt"
  mv "$partial" "$archive"
  sha256sum "$archive" > "${archive}.sha256"
  echo "[$(date -u --iso-8601=seconds)] completed $stem: $(stat -c %s "$archive") bytes"
}

echo "[$(date -u --iso-8601=seconds)] building project_core"
tar --one-file-system -I "zstd -T0 -1" -cf "$DEST/project_core.tar.zst.partial" \
  --exclude='./models' \
  --exclude='./.cache' \
  --exclude='./.pytest_cache' \
  --exclude='./.p9_tmp' \
  --exclude='./tmp' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.partial' \
  -C "$PROJECT_ROOT" .
finalize_archive project_core

echo "[$(date -u --iso-8601=seconds)] building project_models"
tar --one-file-system -I "zstd -T0 -1" -cf "$DEST/project_models.tar.zst.partial" \
  -C "$(dirname "$PROJECT_ROOT")" "$(basename "$PROJECT_ROOT")/models"
finalize_archive project_models

echo "[$(date -u --iso-8601=seconds)] building shared_python_environment"
tar --one-file-system -I "zstd -T0 -1" -cf "$DEST/shared_python_environment.tar.zst.partial" \
  -C "$(dirname "$ENV_ROOT")" "$(basename "$ENV_ROOT")"
finalize_archive shared_python_environment

echo "[$(date -u --iso-8601=seconds)] building active_public_results"
tar --one-file-system -I "zstd -T0 -1" -cf "$DEST/active_public_results.tar.zst.partial" \
  -C "$(dirname "$ACTIVE_ROOT")" "$(basename "$ACTIVE_ROOT")"
finalize_archive active_public_results

{
  echo "backup_version=2"
  echo "status=remote_components_pass_awaiting_local_windows_state"
  echo "created_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "project_root=$PROJECT_ROOT"
  echo "environment_root=$ENV_ROOT"
  echo "active_results_root=$ACTIVE_ROOT"
  echo "destination=$DEST"
  echo "components=project_core,project_models,shared_python_environment,active_public_results,local_windows_state"
  for stem in project_core project_models shared_python_environment active_public_results; do
    archive="$DEST/${stem}.tar.zst"
    echo "${stem}_bytes=$(stat -c %s "$archive")"
    echo "${stem}_members=$(wc -l < "${archive}.members.txt")"
    echo "${stem}_sha256=$(cut -d' ' -f1 "${archive}.sha256")"
  done
  echo "remote_zstd_integrity=pass"
  echo "remote_tar_catalog_integrity=pass"
  echo "remote_completed_utc=$(date -u --iso-8601=seconds)"
} > "$DEST/BACKUP_MANIFEST.txt"

cat > "$DEST/RESTORE.md" <<EOF
# SABER-PID complete-state restore order

1. Verify every archive with \`sha256sum -c *.sha256\` and \`zstd -t ARCHIVE\`.
2. Extract \`project_core.tar.zst\` into the intended project directory.
3. Extract \`local_windows_state.tar.zst\` over a separate checkout, or use it as
   the authoritative Windows-side history/results snapshot. It contains the
   current Git repository and V10 manuscript assets.
4. Extract \`project_models.tar.zst\` beneath \`/home/hera\` to restore the three
   local model directories.
5. Extract \`shared_python_environment.tar.zst\` beneath
   \`/data/hera-1105/RINENG/environments\`. If paths change, recreate the
   environment from \`environment_metadata/pip_freeze.txt\` instead.
6. Extract \`active_public_results.tar.zst\` beneath
   \`/kwkj-k8s/hera_pid_reliability_backups\` when a standalone copy of the
   active V8 result tree is required.

No credentials or unrestricted environment-variable dump is included.
EOF

touch "$DEST/REMOTE_COMPONENTS_COMPLETE"
echo "Remote components complete: $DEST"
