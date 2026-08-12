#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-/home/hera/pid_reliability_benchmark}"
BACKUP_ROOT="${2:-/kwkj-k8s/hera_pid_reliability_backups}"
STAMP="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"

SOURCE_ROOT="$(realpath "$SOURCE_ROOT")"
BACKUP_ROOT="$(realpath -m "$BACKUP_ROOT")"
if [[ "$SOURCE_ROOT" != "/home/hera/pid_reliability_benchmark" ]]; then
  echo "Refusing unexpected source root: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ "$BACKUP_ROOT" != /kwkj-k8s/* ]]; then
  echo "Refusing backup destination outside /kwkj-k8s: $BACKUP_ROOT" >&2
  exit 2
fi

mkdir -p "$BACKUP_ROOT"
BACKUP_DIR="$BACKUP_ROOT/pid_reliability_benchmark_${STAMP}"
mkdir "$BACKUP_DIR"
ARCHIVE="$BACKUP_DIR/pid_reliability_benchmark_${STAMP}.tar.zst"
PARTIAL="$ARCHIVE.partial"
LOG="$BACKUP_DIR/backup.log"
MANIFEST="$BACKUP_DIR/BACKUP_MANIFEST.txt"

exec > >(tee -a "$LOG") 2>&1
echo "Backup started: $(date -u --iso-8601=seconds)"
echo "Source: $SOURCE_ROOT"
echo "Destination: $BACKUP_DIR"

cat > "$MANIFEST" <<EOF
backup_version=1
created_utc=$(date -u --iso-8601=seconds)
hostname=$(hostname)
source=$SOURCE_ROOT
destination=$BACKUP_DIR
compression=zstd
included=code,.git,data,outputs,reports,scoring_artifacts,paper_artifacts,manifests
excluded=models,download_caches,python_caches,test_temp,partial_files,remote_sparse_zip_cache
model_recovery_note=Model weights are excluded because they are reproducibly downloadable and occupy about 95 GB.
EOF

tar --one-file-system --zstd -cf "$PARTIAL" \
  --exclude='./models' \
  --exclude='./.cache' \
  --exclude='./.pytest_cache' \
  --exclude='./.p9_tmp' \
  --exclude='./tmp' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.partial' \
  --exclude='data/external/*/*sparse*.zip' \
  -C "$SOURCE_ROOT" .

zstd -q -t "$PARTIAL"
tar --zstd -tf "$PARTIAL" > "$BACKUP_DIR/archive_members.txt"
MEMBER_COUNT="$(wc -l < "$BACKUP_DIR/archive_members.txt")"
mv "$PARTIAL" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
ARCHIVE_BYTES="$(stat -c %s "$ARCHIVE")"

cat >> "$MANIFEST" <<EOF
archive=$(basename "$ARCHIVE")
archive_bytes=$ARCHIVE_BYTES
archive_member_count=$MEMBER_COUNT
archive_sha256=$(cut -d' ' -f1 "$ARCHIVE.sha256")
zstd_integrity=pass
tar_catalog_integrity=pass
completed_utc=$(date -u --iso-8601=seconds)
status=pass
EOF

echo "Backup complete: $ARCHIVE"
cat "$ARCHIVE.sha256"
echo "Archive bytes: $ARCHIVE_BYTES"
echo "Archive members: $MEMBER_COUNT"
