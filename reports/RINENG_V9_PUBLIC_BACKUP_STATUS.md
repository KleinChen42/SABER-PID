# RINENG V9 public-disk backup status

Status: **PASS**

Backup date: 2026-08-12

Host: `hd03-gpu2-0002`

Public path:
`/kwkj-k8s/hera_pid_reliability_backups/active_v9_20260812/submission/saber_pid_rineng_v9_public_release.zip`

## Verified object

| Artifact | Bytes | SHA-256 | Remote verification |
|---|---:|---|---|
| SABER-PID RINENG V9 public reproducibility release | 8,200,491 | `39f61f3b122e9d60633f6c681fcd7594767609c69e5aa801576954b2dead3493` | byte-preserved upload; remote `sha256sum` matched local release manifest |

The archive contains both final PDFs, editable manuscript and supplementary
sources, deterministic figures and tables, all-page visual-validation records,
the V9 artifact inventory, 91 immutable raw-prediction files with 26,940 rows,
scorer-only references, frozen manifests, analysis code, and relevant tests.
It includes all 54 raw V7 counterfactual cells, closing the older V8 release's
three-cell-only raw V7 subset.

OpenSSH's default SFTP mode was rejected by the JumpServer after transfer; the
verified upload used legacy SCP mode (`scp -O`). Subsequent optional sidecar
uploads were refused by the JumpServer, but those sidecars are already members
of the hash-verified release archive, so recoverability is unaffected.

## Recovery check

On a host with public-disk access:

```bash
sha256sum /kwkj-k8s/hera_pid_reliability_backups/active_v9_20260812/submission/saber_pid_rineng_v9_public_release.zip
unzip -t /kwkj-k8s/hera_pid_reliability_backups/active_v9_20260812/submission/saber_pid_rineng_v9_public_release.zip
```

The expected SHA-256 is the value in the table above. Restore into an explicit
empty directory; do not overwrite an unrelated workspace.
