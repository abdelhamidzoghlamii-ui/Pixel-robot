# MIND-B4 run index

Archived 2026-11-09 by `cp -p` from `~/robot_runs/mind_b4/20261104T181853Z/` and
`~/robot_runs/mind_b4/oneshot_console_20261104T181840Z.log` (runs/); every file `cmp` equal (`diff -r`); the run's
`artifacts.sha256` passes inside the archive. Reviews and smokes added 2026-11-09 (sections below), same method.
SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md).
Code of the run: report code `57d0f1f1d851` = `b4.py` `57d0f1f1…`; before archiving, `b4.py`, `test_b4.py`
(`99096e57…`), `README.md` (`f845058c…`) and `oneshot.sh` (`666423b0…`) matched review round 3's
`frozen_candidate.sha256`. Base `df2bee5`. Owner run from native Termux with `oneshot.sh` (no agent resident).

## Run

| Run | Content | Result | Labels |
|---|---|---|---|
| `runs/20261104T181853Z` | one timed session, 2026-11-04 18:18:54–19:22:46 UTC; blocks I, P, P0, PT, V70, V140, Y, PY, each with its own skin gate (launch-start skin 30.24 °C + 1.0 °C; waits 0–537 s) | **all 8 blocks VALID**, 1 attempt each, session end `all blocks done` | single run; timings VALID (no agent resident: agent check empty before and after every block and at session end; watcher 0 hits, 0 vanished threads in every block); energy INDICATIVE; P vs PT NOT COMPARABLE |

Console log of the run: `runs/oneshot_console_20261104T181840Z.log`.

| Block | Work | Median (p95) | Heat Δ skin / energy (INDICATIVE) | Overrun |
|---|---|---|---|---|
| I | idle, screen on, 3.01 min | – | −0.87 °C / 6 mAh | 0.003 s |
| P | 47 picker decisions (B2 V3), 9 restores | wall 3.792 s (5.079); gen 11.5 tok/s; restore 0.013 s | +7.58 °C / 74 mAh | 0.546 s |
| P0 | item 1 after slot erase, then after restore | 13.042 s (prompt_n 816) / 2.874 s (prompt_n 85) | +2.94 °C / 10 mAh | – |
| PT | P + 5 s root sampler (39 samples, cores 0–3 nice 10) | wall 4.115 s (5.352); gen 11.05 tok/s | +6.23 °C / 70 mAh | 5.181 s |
| V70 | 31 vision calls, 70 image tokens | wall 5.95 s (6.842); prompt 4517 ms, prompt_n 149 | +6.97 °C / 70 mAh | 2.66 s |
| V140 | 17 vision calls, 140 image tokens | wall 11.44 s (13.149); prompt 9930 ms, prompt_n 240 | +6.69 °C / 72 mAh | 9.973 s |
| Y | RobotCam + YOLO, 101 cycles, no server | period 1.676 s (2.062); detect 320: 143 ms, 640: 526 ms | +2.74 °C / 32 mAh | 0.001 s |
| PY | P phase (45 decisions) then Y phase (101 cycles), server loaded | decision 4.001 s (5.046); period 1.677 s (2.067) | P +5.64 °C / 68 mAh; Y −1.37 °C / 34 mAh | 1.012 s / 0.263 s |

All six server blocks: exit 0 after SIGTERM, no SIGKILL. Camera stops: none; capture shown stopped 0.37 s after
`stop_camera` in Y and PY; detector threads on cores 4–5 at start and end. Full numbers: `runs/20261104T181853Z/report.txt`.

## Notes

- **P vs PT NOT COMPARABLE:** start skin 30.386 vs 32.089 °C (1.7 °C apart, limit 1.0). Descriptive only: PT/P median
  decision wall 1.085.
- **Energy INDICATIVE** (charge counter). Battery 81 % at the first reading → 64 % at the last.
- **Stale RobotCam process:** pid 22173 (left from the 2026-11-04 round-3 smoke's Y block; phone not rebooted) was
  present in blocks I, P and P0: Pss 7,235 kB in I, 2,477 → 2,386 kB in P, 2,128 kB in P0; gone from PT on. Y and PY
  use their own RobotCam (pid 11362). `stop_camera` stops capture, not the Android process.

## Smokes (`smokes/`, SMOKE - NOT A MEASUREMENT)

Owner native smokes (`oneshot.sh --smoke`, 20 s blocks, gate before the first block only), copied by `cp -p` from
`~/robot_runs/mind_b4/` with their console logs; every file `cmp` equal (`diff -r`); each smoke's `artifacts.sha256`
passes inside the archive. The two earlier round-1 smokes (`20261103T180210Z_smoke`, `20261103T180735Z_smoke`, no
console logs) are not archived; round 1's packet holds their evidence (`reviews/mind_b4_20261103T181242Z/`).

| Smoke | Console log | Code | Outcome |
|---|---|---|---|
| S1 `20261103T192552Z_smoke` | `oneshot_console_20261103T192540Z.log` | `ade152e337cc` | no block: launch-start skin None (root script trailing `;`) → TypeError; fixed by owner change A |
| S2 `20261103T205510Z_smoke` | `oneshot_console_20261103T205457Z.log` | `6ca61f17447b` | I and P ran; deferred before P0 (skin 28.66 > 27.40 + 1.0 °C after 486 s; smoke then gated every block); owner change B |
| S3 `20261103T215645Z_smoke` | `oneshot_console_20261103T215633Z.log` | `4ee1d851d086` | all 8 blocks, problems [] (round-2 candidate) |
| S4 `20261104T143106Z_smoke` | `oneshot_console_20261104T143053Z.log` | `57d0f1f1d851` | all 8 blocks, problems [], watcher 0 hits / 0 vanished (round-3 candidate = timed-run code) |

## Reviews (`reviews/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh sessions, source-only static reviews (no tools; #144 R3),
copied by `cp -p` from `/termux-home/review/`; every file `cmp` equal (`diff -r`). Each review's record:
`final_review.md`, `review_identity.txt`, `invocation.txt`, `packet.txt`, `snap_pre.txt` = `snap_post.txt`.

| Folder | Content | Verdict |
|---|---|---|
| `mind_b4_20261103T181242Z/` | round 1 (task text `task.txt`, round-1 smokes' evidence) | CHANGES REQUESTED |
| `mind_b4_round1_files_20261103T191014Z/` | the round-1 candidate files (base of round 2's diff) | – |
| `mind_b4_r2_20261103T221847Z/` | round 2 (owner decisions round 1, A and B; S1–S3), 25/25 citations exact | CHANGES REQUESTED (2 × P1, 2 × P2) |
| `mind_b4_round2_files_20261104T122238Z/` | the round-2 candidate files (base of round 3's diff) | – |
| `mind_b4_r3_prep_20261104T134334Z/` | round-3 offline test output (47/47) and frozen candidate hashes | – |
| `mind_b4_r3_20261104T152923Z/` | round 3 (owner decisions on round 2 and vanished threads; S4), 10/10 citations exact | **APPROVED**, one P3 |

Round 3's P3: report restore statistics omit a restore-only row that crosses the deadline; none occurred in the timed
run (every restore was inside a decision row). No code change after round 3. The reviewers' `empty_cwd/` folders are
empty and not tracked by Git.
