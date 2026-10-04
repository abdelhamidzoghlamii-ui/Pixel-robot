# Owner run index

M2 (relsgg-vits16, our export) is the owner-selected benchmark candidate; not deployed.

| File | Bytes | SHA-256 | Fixed label |
|---|---:|---|---|
| [owner_dry_m1.json](runs/owner_dry_m1.json) | 7545 | `46f31122b0f5190a7c0332ca075c7ffa288e8f380778e33602f03eaf4b5c5c15` | DRY RUN, NOT VALID TIMING. |
| [owner_dry_m2.json](runs/owner_dry_m2.json) | 7538 | `88d38481f4f1bb69fd650b376714c3c6269bae9d371336eee90d9c0dd86766e6` | DRY RUN, NOT VALID TIMING. |
| [owner_speed_m1.json](runs/owner_speed_m1.json) | 4278 | `215622ba931ff478d9905195ff1872098e6403c0ff424891409cd1be60caf3b5` | NOT VALID — INCOMPLETE. CoresLost during idle (cores 6-7 lost; Termux not in front); no measurement. |
| [owner_speed_m1_b.json](runs/owner_speed_m1_b.json) | 373263 | `e2532d51db4850cf66af57e6104215ea2296c62128b2c428e7d79c37befc6bc6` | VALID. M1 relsgg-vits16plus, MID cpus 4-5, 2 threads, 36/36 calls, 1288/1301 ms median/P95, VmHWM 446 MiB, no caps, no warm start. Power samples fell just before calls: watts NOT VALID as absolute. |
| [owner_speed_m2.json](runs/owner_speed_m2.json) | 373726 | `e005aff846d17bfcddbc01d73d10e273883b593ac726deb1df4dcc67299cdfb1` | VALID. M2 relsgg-vits16 (our export), same setup, 36/36 calls, 1128/1142 ms median/P95, VmHWM 419 MiB, no caps, no warm start; started cooler (29.2 vs 32.2 C). Power samples fell just before calls: watts NOT VALID as absolute. |

All owner runs used the frozen R1 candidate (manifest SHA-256 `9dd6c64793b4ba733d013d8ff79d98cfc5256a0b6226ff03f1be6af1de2c2ab9`); later code changes do not apply to them.

## Speed1 failed owner session

Separate from the desk2 R1 runs above. Owner-reported base: `0f7b5f4d93998e0e3ae2bf5ea126332a8b2e962c`. Archived files are byte-identical to Downloads originals.

| File | Bytes | SHA-256 | Fixed label |
|---|---:|---|---|
| [owner_session_speed1.json](speed1/runs/owner_session_speed1.json) | 2780 | `6b3166705f8a96aeb4d854eeb5a3311b7f33031d4843940c8ea2283dc6505fb8` | NOT VALID — SESSION INCOMPLETE. Preflight PermissionError reading root-shell affinity; no idle, no block ran; no measurement. |
| [owner_session_speed1.stdout](speed1/runs/owner_session_speed1.stdout) | 1080 | `2589f43983a33024f05dfe8df32c697f22a768c223e4eacde0d7f95f5a9497d9` | NOT VALID — SESSION INCOMPLETE. Preflight PermissionError reading root-shell affinity; no idle, no block ran; no measurement. |
| [owner_session_speed1.stderr](speed1/runs/owner_session_speed1.stderr) | 872 | `b94905603b2fe8f4af39e18f3eb43f6b036a7b446018cd6c0b17e9df0d16b73e` | NOT VALID — SESSION INCOMPLETE. Preflight PermissionError reading root-shell affinity; no idle, no block ran; no measurement. |
