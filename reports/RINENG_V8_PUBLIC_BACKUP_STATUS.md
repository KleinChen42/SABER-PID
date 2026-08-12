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

The maintenance-time acceleration delta is frozen separately as
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r3.tar`
(15,872 bytes; 4 members; SHA-256
`039c84e666ace88779d57590aabef8d3e064090ec25c4c5db1d9477a98911df9`).
Apply it after r2. It adds the condition-disjoint seed31/DEXPI launch path and
the corresponding recovery-document updates; its remote hash and tar catalogue
were independently checked.

The latest script delta superseding r3 is
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r4.tar`
(9,728 bytes; 3 members; SHA-256
`55fffb948a24b2f7078dd2354dddc951ebdfe63b6ba8b619bd9d0a0042724f9b`).
It contains the optional DEXPI mainline-wait bypass plus the DEXPI and
dataset-disjoint InternVL acceleration launchers. Its remote hash and three-file
catalogue passed. Apply r4 after r2; r3 is retained only as an earlier recovery
point.

The latest maintenance-execution delta is
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r5.tar`
(78,336 bytes; 9 members; SHA-256
`c37e14d7c1f70d98c1f97cfc7a849ccf9d2014178fd940fe9d47619b36d9a666`).
Its remote hash, byte size, and nine-file catalogue passed. It adds the
mainline shard barrier, the four-GPU condition-disjoint InternVL route, and
final-submission validation updates. Apply r5 after the newer Git bundle (or
after r2+r4 when restoring from filesystem archives).

The latest manuscript/code/result-note overlay is
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r6.tar`
(395,264 bytes; 25 members; SHA-256
`c28c28f37e90383750f54680e4b12169d5c7b3be45b6650cb1fb721647a1e29d`).
Its remote hash, size, and catalogue passed. It includes the 230-word abstract,
quality/DEXPI results narrative and tables, inspected DEXPI figure, final-
submission validator, and r5 shard controls. Apply r6 last when restoring the
latest in-progress paper state.

The subsequent generated-table/validation overlay is
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r7.tar`
(1,164,288 bytes; 34 members; SHA-256
`63b2b5c53f8b1d9a798ec8e5a3a9c49e6e36d43bfedd47dd0ca45980ccaf6b55`).
Its local and remote hashes, byte size, and catalogue count passed. It adds the
score-derived source-disjoint quality table, the stronger V8 submission gate,
their tests, and the latest manuscript/result files. Apply r7 after r6.

Version-control recovery is independently available as
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_preparation_2ed25a4.bundle`
(12,211,669 bytes; SHA-256
`95b107d83a8ee703c9a588e4b06a87d7723e449e63855c802006b268cdace6bb`).
Local `git bundle verify` passed; the bundle records complete history with
`master` and `HEAD` at commit `2ed25a4cf75b346e26e37b0467ed6fc905e882cb`.

The maintenance-safe launch delta is also preserved in the newer complete
bundle
`/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_maintenance_f706fd8.bundle`
(12,190,420 bytes; SHA-256
`b8bd57ae1563dd9f4bf1b03d439e49474ed29a78bb3f86d5c10146e62a66278b`).
Local `git bundle verify` passed and remote size/hash matched; `master` and
`HEAD` are at commit `f706fd8ceb028555eaaafce802ed5be28cd29b48`.

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
