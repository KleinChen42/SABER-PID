#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-/home/hera/pid_reliability_benchmark}"
DEST_ROOT="${2:-/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/snapshots}"
LABEL="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"

SOURCE_ROOT="$(realpath "$SOURCE_ROOT")"
DEST_ROOT="$(realpath -m "$DEST_ROOT")"
if [[ "$SOURCE_ROOT" != "/home/hera/pid_reliability_benchmark" ]]; then
  echo "Unexpected source root: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ "$DEST_ROOT" != /kwkj-k8s/* ]]; then
  echo "Destination must remain under /kwkj-k8s: $DEST_ROOT" >&2
  exit 2
fi

mkdir -p "$DEST_ROOT"
ARCHIVE="$DEST_ROOT/rineng_v8_${LABEL}.tar.zst"
PARTIAL="$ARCHIVE.partial"
MANIFEST="$DEST_ROOT/rineng_v8_${LABEL}.manifest.txt"

INCLUDES=(
  data/manifests/rineng_v8_quality_robustness_plan.json
  data/manifests/rineng_v8_quality_images.json
  data/manifests/rineng_v8_internvl_budget_matched_plan.json
  data/manifests/rineng_v8_internvl_budget_matched_plan_r2.json
  data/manifests/rineng_v8_internvl_budget_matched_plan_r3.json
  data/manifests/rineng_v8_dexpi_external_plan.json
  data/raw/rineng_v8_quality
  data/processed/rineng_v8_quality
  data/processed/rineng_v8_dexpi_external
  data/answer_store/rineng_v8_dexpi_external_hidden.jsonl
  reports/logs
  reports/generated/rineng_cost_sensitive_operating_modes_v8.json
  reports/generated/rineng_cost_sensitive_operating_modes_v8.csv
  reports/generated/rineng_cost_sensitive_decision_rule_v8.csv
  reports/generated/rineng_v8_dexpi_external_audit.json
  reports/generated/pid2graph_open100_remote_catalog_v8.json
  reports/generated/pid2graph_open100_complete_materialized_v8.json
  reports/generated/rineng_v8_pid2graph_open100_audit.json
  reports/RINENG_V8_PUBLIC_BACKUP_STATUS.md
  reports/RINENG_V8_H200_MAINTENANCE_RESUME.md
  paper/manuscript.tex
  paper/supplementary.tex
  paper/figure_manifest.md
  paper/figure_captions.md
  paper/figures/figure_4_cost_sensitive_operating_modes_v8.pdf
  paper/figures/figure_4_cost_sensitive_operating_modes_v8.png
  scripts/build_cost_sensitive_operating_modes_v8.py
  scripts/build_rineng_v8_extension_figures.py
  scripts/audit_pid2graph_open100_v8.py
  scripts/prepare_quality_robustness_v8.py
  scripts/prepare_dexpi_external_v8.py
  scripts/run_internvl_budget_matched_v8.py
  scripts/run_paddleocr_external_v8.py
  scripts/run_qwen_counterfactual_quality_v8.py
  scripts/run_qwen_counterfactual_prompt_matrix_v7.py
  scripts/score_dexpi_external_v8.py
  scripts/score_rineng_v8_extensions.py
  scripts/fetch_remote_zip_subset.py
  scripts/launch_rineng_v8_h200.sh
  scripts/launch_rineng_v8_external_h200.sh
  scripts/snapshot_rineng_v8_active.sh
  licenses/DEXPI_TRAINING_TEST_CASES_LICENSE.txt
  tests/test_cost_sensitive_operating_modes_v8.py
  tests/test_audit_pid2graph_open100_v8.py
  tests/test_build_rineng_v8_extension_figures.py
  tests/test_prepare_dexpi_external_v8.py
  tests/test_rineng_v8_extension_preparation.py
  tests/test_score_dexpi_external_v8.py
  tests/test_score_rineng_v8_extensions.py
)

EXISTING=()
for path in "${INCLUDES[@]}"; do
  if [[ -e "$SOURCE_ROOT/$path" ]]; then
    EXISTING+=("$path")
  fi
done
if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "No v8 artifacts found" >&2
  exit 1
fi

tar --zstd -cf "$PARTIAL" -C "$SOURCE_ROOT" "${EXISTING[@]}"
zstd -q -t "$PARTIAL"
tar --zstd -tf "$PARTIAL" > "$ARCHIVE.members.txt"
mv "$PARTIAL" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
cat > "$MANIFEST" <<EOF
status=pass
created_utc=$(date -u --iso-8601=seconds)
source=$SOURCE_ROOT
archive=$ARCHIVE
archive_bytes=$(stat -c %s "$ARCHIVE")
archive_member_count=$(wc -l < "$ARCHIVE.members.txt")
archive_sha256=$(cut -d' ' -f1 "$ARCHIVE.sha256")
zstd_integrity=pass
tar_catalog_integrity=pass
EOF
cat "$MANIFEST"
