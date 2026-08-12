# RINENG V8 H200 maintenance resume runbook

This runbook resumes only the authorized RINENG V8 experiments. It must not
terminate or appropriate another user's process. Formal outputs are stored on
the public disk and all inference runners use `--skip-existing`, so a partial
cell is resumed by instance ID and a completed cell is not repeated.

## Recovery roots

- Node project: `/home/hera/pid_reliability_benchmark`
- Public experiment root:
  `/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812`
- Full recovery archive:
  `/kwkj-k8s/hera_pid_reliability_backups/pid_reliability_benchmark_20260812T064200Z/pid_reliability_benchmark_20260812T064200Z.tar.zst`
- Latest V8 snapshots:
  `/kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/snapshots`

Verify SHA-256 and `zstd -t` before restoring an archive. Restore into an
explicitly selected empty parent; never extract over an unrelated project.
After restoring the latest snapshot, verify and apply the code-only overlay:

```bash
echo '86dff10fd666512197b45438153e185c32e0aa2bfbc7ac614b3321204918398a  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r2.tar' | sha256sum -c -
tar -tf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r2.tar | wc -l
tar -xf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r2.tar -C /home/hera/pid_reliability_benchmark
```

The catalogue count must be 58 before extraction.

Then verify and apply the small maintenance-time acceleration delta:

```bash
echo '039c84e666ace88779d57590aabef8d3e064090ec25c4c5db1d9477a98911df9  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r3.tar' | sha256sum -c -
tar -tf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r3.tar | wc -l
tar -xf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r3.tar -C /home/hera/pid_reliability_benchmark
```

The r3 catalogue count must be 4. A later script-only delta supersedes it:

```bash
echo '55fffb948a24b2f7078dd2354dddc951ebdfe63b6ba8b619bd9d0a0042724f9b  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r4.tar' | sha256sum -c -
tar -tf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r4.tar | wc -l
tar -xf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r4.tar -C /home/hera/pid_reliability_benchmark
```

The r4 catalogue count must be 3. Apply r4 after r2; r3 is retained only as an
earlier recovery point and need not be applied when r4 is used.

Finally apply the latest execution/validation delta:

```bash
echo 'c37e14d7c1f70d98c1f97cfc7a849ccf9d2014178fd940fe9d47619b36d9a666  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r5.tar' | sha256sum -c -
tar -tf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r5.tar | wc -l
tar -xf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_overlay_20260812_r5.tar -C /home/hera/pid_reliability_benchmark
```

The r5 catalogue count must be 9.

For the latest in-progress paper and validation state, apply r6 last:

```bash
echo 'c28c28f37e90383750f54680e4b12169d5c7b3be45b6650cb1fb721647a1e29d  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r6.tar' | sha256sum -c -
tar -tf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r6.tar | wc -l
tar -xf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r6.tar -C /home/hera/pid_reliability_benchmark
```

The r6 catalogue count must be 25.

Apply the final pre-maintenance generated-table and validation delta after r6:

```bash
echo '63b2b5c53f8b1d9a798ec8e5a3a9c49e6e36d43bfedd47dd0ca45980ccaf6b55  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r7.tar' | sha256sum -c -
tar -tf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r7.tar | wc -l
tar -xf /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_code_paper_overlay_20260812_r7.tar -C /home/hera/pid_reliability_benchmark
```

The r7 catalogue count must be 34.

If Git metadata must be reconstructed independently of the filesystem
archive, verify the public bundle hash and clone it into an empty target:

```bash
echo '95b107d83a8ee703c9a588e4b06a87d7723e449e63855c802006b268cdace6bb  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_preparation_2ed25a4.bundle' | sha256sum -c -
git clone /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_preparation_2ed25a4.bundle /explicit/empty/recovery/target
```

Prefer the newer complete bundle when the maintenance-time launchers are
needed:

```bash
echo 'b8bd57ae1563dd9f4bf1b03d439e49474ed29a78bb3f86d5c10146e62a66278b  /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_maintenance_f706fd8.bundle' | sha256sum -c -
git clone /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/recovery_docs/rineng_v8_maintenance_f706fd8.bundle /explicit/empty/recovery/target
```

This bundle records complete history at
`f706fd8ceb028555eaaafce802ed5be28cd29b48`.

## Read-only state check

```bash
screen -ls | grep rie_v8 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -n 30 /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/logs/mainline_full.log
tail -n 30 /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/logs/dexpi_external_full.log
cat /kwkj-k8s/hera_pid_reliability_backups/active_v8_20260812/control/internvl_mainline_barrier.txt 2>/dev/null || true
```

The configured physical GPU is GPU 1. Resume only if it is still the selected
project GPU and is actually idle. If it is occupied after maintenance, wait;
do not kill the occupying process.

## Mainline resume

If `rie_v8_mainline` is absent and `MAINLINE_COMPLETE` is not present in the
public log:

```bash
cd /home/hera/pid_reliability_benchmark
screen -dmS rie_v8_mainline bash -lc \
  'CUDA_VISIBLE_DEVICES=1 ./scripts/launch_rineng_v8_h200.sh mainline_full'
```

This resumes, in order:

1. Qwen3-VL-8B quality robustness at 3072-side/512 tokens;
2. InternVL3.5-8B at the closest safe 54-tile/512-token budget on all three
   pairwise source-disjoint subsets.

During the maintenance-window acceleration, the original mainline may report
`waiting_for_condition_disjoint_shards` while the public control file contains
`WAIT`. This is intentional. Keep it at `WAIT` while any of
`rie_v8_i29`, `rie_v8_i31`, `rie_v8_i_setb_correct`, or
`rie_v8_i_setb_shuffled` is active or incomplete. The
`rie_v8_mainline_resume` coordinator changes it to `GO` only after all four
screens exit. After an outage, validate every shard's expected row count and
status before writing `GO`; otherwise relaunch the missing shard with the r5
launch script and `--skip-existing` behavior.

## External-family queue resume

If `rie_v8_dexpi_queue` is absent and `DEXPI_EXTERNAL_COMPLETE` is not present:

```bash
cd /home/hera/pid_reliability_benchmark
screen -dmS rie_v8_dexpi_queue bash -lc \
  './scripts/launch_rineng_v8_external_h200.sh'
```

The queue waits for the mainline and for physical GPU 1 to become idle, then
runs DEXPI Qwen correct/shuffled/no-image, resumes OCR if necessary, and runs
the CPU scorer. The DEXPI OCR file already completed before maintenance and is
stored under the public output root.

## Completion evidence

Do not infer completion from a missing screen alone. Require the corresponding
public log marker, the complete expected JSONL row counts, zero non-`ok` rows,
zero answer-use flags, the frozen plan hash in every row, and a passing
independent local rescore before manuscript integration.
