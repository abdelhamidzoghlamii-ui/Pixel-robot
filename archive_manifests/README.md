# Off-phone archive on Google Drive (private)

The remote is `gdrive:pixel-robot-archive` (rclone remote `gdrive`, owner's private Drive; **not** public).
This folder holds only the manifests. No tar and no archived content is in this repository.

| Drive folder | Uploaded (UTC) | Content | Manifest here |
|---|---|---|---|
| `P1/`, `P3/` | 2026-10-26 13:06–17:08 | 29 `.tar` (P23D sweep: reviews, cleans, therm/llama evidence, laya/von/sel-candidates trees) | `UPLOADED.txt` (bytes, SHA-256, MD5 per tar); `FAILED_attempt1.txt`, `FAILED.txt` = first-attempt failures, all later in `UPLOADED.txt` |
| — | 2026-10-26 12:43 | not uploaded: venv package lists and pip cache size recorded before removal | `ext_venv-relate-export.txt`, `yolo_bench_venv.txt`, `pip_cache.txt` |
| `single_copies_20261116T041820Z/` | 2026-11-16 04:18 | 12 `.tar.gz`: files that existed only on the phone (CLEANUP-A) | `single_copies_20261116T041820Z.txt` (index), `single_copies_20261116T041820Z/` (`<name>.MANIFEST.sha256` per tar, `SHA256SUMS.txt` for the tars) |
| `strays_20261116T082915Z/` | 2026-11-16 08:29 | 1 `.tar.gz`: thermal logs, two MIND-B4 smokes, MIND-B2 stdout, #118 deletion lists, `test_oneshot.sh` (CLEANUP-B) | `strays_20261116T082915Z/` (`.MANIFEST.sha256`, `.tar.gz.sha256`) |

Verification on 2026-11-16:
- the 29 P1/P3 tars: `rclone lsl` + `rclone md5sum` equal `UPLOADED.txt` (name, bytes, MD5);
- the single_copies and strays folders: `rclone check` 0 differences, sizes equal local;
- one tar from each was downloaded back: tar SHA-256 OK and every manifest line OK.

`single_copies/dryrun1_batterystats_history.tar.gz` must never be extracted into this public repository
(`benchmark/heat_control/dryrun1/ARTIFACTS.md`).

## Restore and verify

```bash
D=gdrive:pixel-robot-archive/single_copies_20261116T041820Z   # or .../strays_20261116T082915Z
mkdir -p ~/restore && cd ~/restore
rclone copy "$D" .                                             # or one file: rclone copy "$D/<name>.tar.gz" .
sha256sum -c SHA256SUMS.txt                                    # strays: sha256sum -c strays_*.tar.gz.sha256
mkdir x && tar -xzf <name>.tar.gz -C x                         # paths are relative to / (termux-home/..., tmp/...)
cd x && sha256sum -c MANIFEST.sha256                           # every file must print OK
```

For a P1/P3 tar: `rclone copy gdrive:pixel-robot-archive/P1/<name>.tar .`, then compare `sha256sum <name>.tar`
with its `UPLOADED.txt` line before `tar -xf`.
