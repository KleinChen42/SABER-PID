# RINENG V8 public-disk backup status

Status: **PASS**

Backup date: 2026-08-12

Public root: `/kwkj-k8s/hera_pid_reliability_backups`

This record freezes the recovery state prepared before maintenance of the H200
server. Model weights and disposable download caches are intentionally excluded;
code, Git metadata, data manifests, scorer-only references, raw outputs, reports,
paper artifacts, and experiment plans are included. Formal V8 outputs are also
written directly to `/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812`,
so completed cells do not depend on the node-local filesystem.

## Verified archives

| Role | Archive | Bytes | Members | SHA-256 | Checks |
|---|---|---:|---:|---|---|
| Full project recovery point | `/kwkj-k8s/hera_pid_reliability_backups/pid_reliability_benchmark_20260812T064200Z/pid_reliability_benchmark_20260812T064200Z.tar.zst` | 1,781,606,107 | 19,088 | `c7217352208621b13b4c3db6dc9afebd36f19d586d21258cf03d2ebc7b3bfe59` | zstd pass; tar catalogue pass |
| V8 preparation snapshot | `/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/snapshots/rineng_v8_preparation.tar.zst` | 1,938,262,815 | 746 | `006f833adc9b5a178f79e445ab81b086c02e4cf17639f6b06e7e331cfb8438cb` | zstd pass; tar catalogue pass |
| V8 post-DEXPI freeze | `/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/snapshots/rineng_v8_post_dexpi_freeze.tar.zst` | 1,938,251,798 | 758 | `ff1fdc899b6e8ad1e74ace8975b260d30fbb007b9f2b2e29a9a21ad7bccf152a` | zstd pass; tar catalogue pass |
| V8 post-PID2Graph/code freeze | `/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/snapshots/rineng_v8_post_pid2graph_codefreeze.tar.zst` | 1,938,984,958 | 784 | `bb0600d80f879e7a0102bd7d726f53cfefe8cb496e0a3e0dc93213a7cdcac779` | zstd pass; tar catalogue pass |

The latest code-only recovery overlay is
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r2.tar`
(1,325,568 bytes; 58 members; SHA-256
`86dff10fd666512197b45438153e185c32e0aa2bfbc7ac614b3321204918398a`).
It is applied after archive restoration and contains the post-snapshot scorer,
independent validator, table/release builders, tests, manuscript files, and the
fresh-process InternVL CUDA-recovery enhancement.

The full backup was created from `/home/hera/pid_reliability_benchmark` on host
`hd03-gpu2-0002` and completed at `2026-08-12T06:44:03Z`. Approximately 95 GB
of reproducibly downloadable model weights were excluded. Their identifiers,
runtime settings, frozen predictions, and scoring provenance remain in the
project artifacts.

## Recovery and verification

On a Linux host with access to the public disk:

```bash
sha256sum -c pid_reliability_benchmark_20260812T064200Z.tar.zst.sha256
zstd -t pid_reliability_benchmark_20260812T064200Z.tar.zst
tar --use-compress-program=unzstd -tf pid_reliability_benchmark_20260812T064200Z.tar.zst >/dev/null
mkdir -p /chosen/recovery/parent
tar --use-compress-program=unzstd -xf pid_reliability_benchmark_20260812T064200Z.tar.zst -C /chosen/recovery/parent
```

Restore only into an explicitly selected empty recovery parent. The V8 snapshots
use the same verification and extraction commands.

## Local audit evidence

The copied server-side manifests and logs are retained under
`reports/logs/rineng_v8_backup/`. They are evidence of archive construction and
integrity checking; the SHA-256 values above remain the authoritative content
identifiers.
