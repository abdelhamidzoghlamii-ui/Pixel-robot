# MIND-B5 run index

Archived 2026-11-15 by `cp -p` from `~/robot_runs/mind_b5/` (runs/, smokes/, with their console logs) and
`/termux-home/review/` (reviews/); every file `cmp` equal (`diff -r`); each run's `artifacts.sha256` passes inside the
archive. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md). Code archived here = review round 3's frozen
candidate (`b5.py` `c9dd0630…`, `test_b5.py` `9aec6c15…`, `README.md` `2a0000da…`, `oneshot.sh` `14d1f738…`), base
`72eac6c`. Timed run of this code: `runs/20261115T172733Z` (added 2026-11-15, same method).

## Runs (`runs/`)

| Run | Content | Result | Labels |
|---|---|---|---|
| **`runs/20261115T172733Z`** | owner, native Termux, `oneshot.sh` (no agent resident), code `c9dd0630cd99` (= round-3 approved candidate, base `edbbc35`); session 1 2026-11-15 17:27–18:28 UTC: arm A, then deferred before B (skin 26.78 °C not ≤ launch-start 24.85 + 1.0 °C after 1,804 s); session 2 (`--resume`) 19:39–20:10 UTC: arm B | **both arms VALID**, one attempt each; session 2 end `all arms done` | timings VALID (agent check empty before and after each arm and at both session ends; watcher 0 hits in 62 scans per arm; overrun 0.078 / 0.026 s); no affinity drift, no re-pin changed a thread, cpuset always `/top-app`; A vs B **descriptive only** (B started at battery 82 % vs A 99 %, after a separate cool-down); energy INDICATIVE |
| T1 `runs/20261115T002824Z` | owner, native Termux, `oneshot.sh` without `--smoke`, 2026-11-15 00:28–00:58 UTC, session 1; code `7ce57beee409` (round-2 code BEFORE the re-pin change, not reviewed) | arm A ran its 30 min; arm B not run; session end `arm A INVALID` | **INVALID**: server affinity drift (all 15 llama-server threads on cpus 0–7 in all 62 watcher scans from t = 0.1 s; detector threads also found on 0–7 at the first camera start and near 930 s, re-pinned by `main.py` at camera starts); unreviewed code. Kept as evidence (owner decision 2026-11-15: not resumed); not a measurement |

Console logs: `runs/oneshot_console_20261115T172720Z.log` (session 1), `runs/oneshot_console_20261115T193937Z.log`
(session 2), `runs/oneshot_console_20261115T002811Z.log` (T1). T1 motivated the owner's re-pin decision (2026-11-15).

| `20261115T172733Z` | A FIXED (30 s marks) | B ADAPTIVE (skin < 36.0 °C) |
|---|---|---|
| Decisions served | 41 (ticks: 41 decided, 19 paused) | 46 (41 scripted + 5 real) |
| Scripted offered / served / dropped / unserved | 180 / 41 / 138 / 1 | 180 / 41 / 138 / 1 |
| Real person events offered / served / dropped | 0 / 0 / 0 | 29 / 5 / 24 |
| Event→decision latency median (p95) | 4.12 s (5.28) | 4.46 s (10.60) |
| Decision wall median (p95) | 4.12 s (5.28) | 4.18 s (5.34) |
| Heat pauses / resumes; first pause | 10 / 9 (last pause open at the end); 616 s | 4 / 4; 561 s |
| Pause time; ACTIVE share | 538.5 s of 1,800 s; 0.701 | 234.6 s of 1,800 s; 0.870 |
| Skin start / max (trace) / end | 26.91 / 37.45 / 35.77 °C | 26.88 / 38.25 / 36.79 °C |
| Minimum CPU caps at readings (kHz) | policy0 1,401,000; policy4 2,348,000; policy6 2,850,000 | same |
| Camera cycle period median (p95) | 1.691 s (2.125) | 1.680 s (2.099) |
| Detect 320 / 640 median | 152.4 / 568.4 ms | 146.8 / 549.2 ms |
| MemAvailable minimum | 1,559,732 kB | 1,809,272 kB |
| Battery start → end; energy INDICATIVE | 99 → 88 %; 372 mAh | 82 → 68 %; 432 mAh |
| Agreement with the B2 run of record | 41/41 | 40/41 (id 39 differs: one of B2's three TEST flip items, 6, 36, 39) |

All numbers from `runs/20261115T172733Z/report.txt` / `arms.jsonl`; server exit 0, no SIGKILL, in both arms.

## Smokes (`smokes/`, SMOKE - NOT A MEASUREMENT)

Owner native smokes (`oneshot.sh --smoke`), gate before the first arm only, smoke-only lowered heat thresholds.

| Smoke | Console log | Code | Outcome |
|---|---|---|---|
| S1 `20261114T201817Z_smoke` | `oneshot_console_20261114T201803Z.log` | `7ad479ec0070` | 120 s arms, thresholds launch-start skin + 1.5 / + 0.5 °C; both arms 1 pause, **0 resumes** (owner decision: per-arm thresholds, 180 s arms) |
| S2 `20261114T211511Z_smoke` | `oneshot_console_20261114T211459Z.log` | `42c4212a1d92` | round-1 candidate; 180 s arms, arm-start skin + 1.0 / pause − 0.5 °C; A 2 pauses / 1 resume, B 1 / 1; problems []; evidence for review round 1 |
| S3 `20261115T073643Z_smoke` | `oneshot_console_20261115T073630Z.log` | `e0eb377635e1` | round-2 candidate with re-pin; A 2 pauses / 1 resume, B 1 / 1; problems []; watcher 0 hits, no drift, every re-pin ok with no thread changed, cpuset always `/top-app`; evidence for review round 2 |

`oneshot_console_20261115T073143Z.log`: an attempt refused before any run (battery `Charging`). No real person event
and no cancelled decision occurred in any smoke; the drift re-pin path is untested on the phone.

## Reviews (`reviews/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh sessions, source-only static reviews (no tools; #144 R3). Each
review's record: `final_review.md`, `review_identity.txt`, `invocation.txt`, `packet.txt`, `snap_pre.txt` =
`snap_post.txt`. The reviewers' `empty_cwd/` folders are empty and not tracked by Git.

| Folder | Content | Verdict |
|---|---|---|
| `mind_b5_prep_20261109T220110Z/` | the Coder's B4-findings checklist (owner change C) | – |
| `mind_b5_r1_20261114T220056Z/` | round 1 (task, Coder choices, S2), 18/18 citations exact | CHANGES REQUESTED (5 × P1, 2 × P2, 1 × P3) |
| `mind_b5_round1_files_20261114T233711Z/` | the round-1 frozen files, round-2 candidate hashes before the re-pin change, test log | – |
| `mind_b5_r2_20261115T160557Z/` | round 2 (owner decisions on round 1 and the re-pin change; S3; T1 as evidence), 6/6 citations exact | CHANGES REQUESTED (1 × P1, 1 × P2) |
| `mind_b5_r3_20261115T171427Z/` | round 3 (owner decision on round 2; no new smoke by owner decision), 9/9 citations exact | **APPROVED**, no findings |
