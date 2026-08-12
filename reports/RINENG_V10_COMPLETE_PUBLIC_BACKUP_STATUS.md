# RINENG V10 complete public-disk backup status

Status: **PASS**  
Completed UTC: `2026-08-12T17:40:46Z`  
H200 host: `hd03-gpu2-0002`  
Destination: `/kwkj-k8s/hera_pid_reliability_backups/saber_pid_complete_20260812T170627Z`

## Scope

The backup is split into five independently recoverable archives:

| Component | Bytes | Members | SHA-256 |
|---|---:|---:|---|
| H200 active public results | 7,870,282,848 | 972 | `a9b2abc91a455686ecdd87a0e172fe2825dc30b6eca1b36172edbe4aaf058aa4` |
| Current Windows workspace, Git history, V10 manuscript and results | 2,045,073,096 | 5,336 | `146e122679ef105cf2c909496191a3cd8a1d93d07ddae3777f483aee6d0f918f` |
| H200 project core, data, outputs, reports and project-local environments | 3,886,939,957 | 22,981 | `caf198581af884d3ba89b6e09c7904b2a78eb42aa0960b5856954145f40f79a0` |
| Three local model directories: Qwen3-VL-8B, Qwen3-VL-32B and InternVL3.5-8B | 79,306,317,356 | 62 | `234a714056df9a64d9959128f14e36c0d3fc27760dbd222ec346d273f5d616ac` |
| Shared H200 Python environment | 3,714,534,984 | 37,559 | `2d7899afb236adc9563ab58b6161507310b9f26e3f7e4e39759ec0f3b9868397` |
| **Total** | **96,823,148,241** | **66,910** | Aggregate check: **PASS** |

## Validation

- Every component passed `zstd -t`.
- Every component passed a full `tar --zstd -tf` catalog read.
- Each component has its own `.sha256` and `.members.txt` file.
- `ALL_ARCHIVES.sha256` was rechecked after all five archives were finalized;
  all five results are `OK` in `FINAL_INTEGRITY_CHECK.txt`.
- `BACKUP_MANIFEST_FINAL.txt` reports `status=pass`.
- The zero-byte `COMPLETE` marker was created only after aggregate validation.
- Key-member checks confirmed the V10 manuscript/supplement PDFs, Git HEAD and
  working-tree patch, experiment outputs/reports, all three model families,
  the Python executable and PyTorch installation, and the active V8 result tree.

## Environment record

`environment_metadata/` includes the host/kernel, GPU and driver inventory,
CUDA/NVCC, GCC, disk layout, Python runtime, installed-distribution inventory,
model/project/environment/result file inventories, and source sizes. No secret
environment-variable dump or credentials are included.

The captured shared environment reports Python 3.10.12, PyTorch 2.13.0+cu126,
CUDA availability, CUDA 12.6, and cuDNN 9.1.2. Project-local `.v8_site`, `.venv`,
and PaddleOCR environment content are retained in `project_core.tar.zst`.

## Recovery

Use `RESTORE.md` in the backup directory. Final backup and finalizer scripts are
also copied under `recovery_tools/`. Verify `ALL_ARCHIVES.sha256` before any
extraction.

The Windows archive omits only transient pytest/pip temporary directories and
duplicate extracted release directories that Windows ACLs marked inaccessible;
their formal ZIP/bundle counterparts are included. It retains the formal data,
outputs, reports, release archives, Git repository, source, paper, figures, and
compiled V10 PDFs.
