#!/usr/bin/env bash
set -euo pipefail

DEST="${1:?usage: finalize_complete_public_backup_v10.sh /kwkj-k8s/.../saber_pid_complete_STAMP}"
DEST="$(realpath "$DEST")"
[[ "$DEST" == /kwkj-k8s/hera_pid_reliability_backups/saber_pid_complete_* ]] || {
  echo "Refusing unexpected backup directory: $DEST" >&2; exit 2;
}
[[ -f "$DEST/REMOTE_COMPONENTS_COMPLETE" ]] || {
  echo "Remote component marker is missing" >&2; exit 1;
}
LOCAL="$DEST/local_windows_state.tar.zst"
[[ -f "$LOCAL" ]] || { echo "Local Windows archive is missing" >&2; exit 1; }
[[ ! -e "$DEST/COMPLETE" ]] || { echo "Backup is already finalized" >&2; exit 1; }

zstd -q -t "$LOCAL"
tar --zstd -tf "$LOCAL" > "$LOCAL.members.txt"
sha256sum "$LOCAL" > "$LOCAL.sha256"

cat >> "$DEST/BACKUP_MANIFEST.txt" <<EOF
local_windows_state_bytes=$(stat -c %s "$LOCAL")
local_windows_state_members=$(wc -l < "$LOCAL.members.txt")
local_windows_state_sha256=$(cut -d' ' -f1 "$LOCAL.sha256")
local_zstd_integrity=pass
local_tar_catalog_integrity=pass
finalized_utc=$(date -u --iso-8601=seconds)
status=pass
EOF

for archive in "$DEST"/*.tar.zst; do
  cat "${archive}.sha256"
done > "$DEST/ALL_ARCHIVES.sha256"
sha256sum -c "$DEST/ALL_ARCHIVES.sha256" > "$DEST/FINAL_INTEGRITY_CHECK.txt"

total_bytes=0
total_members=0
{
  echo "backup_version=2"
  echo "status=pass"
  echo "finalized_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "destination=$DEST"
  echo "component_count=5"
  for archive in "$DEST"/*.tar.zst; do
    stem="$(basename "$archive" .tar.zst)"
    bytes="$(stat -c %s "$archive")"
    members="$(wc -l < "${archive}.members.txt")"
    digest="$(cut -d' ' -f1 "${archive}.sha256")"
    total_bytes=$((total_bytes + bytes))
    total_members=$((total_members + members))
    echo "${stem}_bytes=$bytes"
    echo "${stem}_members=$members"
    echo "${stem}_sha256=$digest"
  done
  echo "total_archive_bytes=$total_bytes"
  echo "total_archive_members=$total_members"
  echo "zstd_integrity=pass"
  echo "tar_catalog_integrity=pass"
  echo "aggregate_sha256_check=pass"
} > "$DEST/BACKUP_MANIFEST_FINAL.txt"

find "$DEST" -maxdepth 2 -type f -printf '%P\t%s\n' | sort \
  > "$DEST/BACKUP_FILE_TREE.tsv"
touch "$DEST/COMPLETE"
echo "Complete backup finalized: $DEST"
cat "$DEST/ALL_ARCHIVES.sha256"
