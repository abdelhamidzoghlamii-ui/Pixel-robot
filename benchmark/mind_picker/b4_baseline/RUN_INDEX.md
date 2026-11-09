# MIND-B4 run index

Archived 2026-11-09 by `cp -p` from `~/robot_runs/mind_b4/20261104T181853Z/` and
`~/robot_runs/mind_b4/oneshot_console_20261104T181840Z.log` (runs/); every file `cmp` equal (`diff -r`); the run's
`artifacts.sha256` passes inside the archive. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md).
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
- Not archived here: the native smokes (`~/robot_runs/mind_b4/*_smoke`, NOT A MEASUREMENT) and the review folders.

## Reviews (not archived in this folder)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh sessions, source-only static reviews (no tools; #144 R3):
round 1 `/termux-home/review/mind_b4_20261103T181242Z/` CHANGES REQUESTED; round 2
`/termux-home/review/mind_b4_r2_20261103T221847Z/` CHANGES REQUESTED; round 3
`/termux-home/review/mind_b4_r3_20261104T152923Z/` **APPROVED** with one P3 (report restore statistics omit a
restore-only row that crosses the deadline; none occurred in this run: every restore was inside a decision row).
Round-3 citations verified exact (10/10). No code change after round 3.
