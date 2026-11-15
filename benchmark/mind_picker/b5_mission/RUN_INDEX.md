# MIND-B5 run index

Archived 2026-11-15 by `cp -p` from `~/robot_runs/mind_b5/` (runs/, smokes/, with their console logs) and
`/termux-home/review/` (reviews/); every file `cmp` equal (`diff -r`); each run's `artifacts.sha256` passes inside the
archive. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md). Code archived here = review round 3's frozen
candidate (`b5.py` `c9dd0630…`, `test_b5.py` `9aec6c15…`, `README.md` `2a0000da…`, `oneshot.sh` `14d1f738…`), base
`72eac6c`. No timed run of this code exists yet.

## Runs (`runs/`)

| Run | Content | Result | Labels |
|---|---|---|---|
| T1 `runs/20261115T002824Z` | owner, native Termux, `oneshot.sh` without `--smoke`, 2026-11-15 00:28–00:58 UTC, session 1; code `7ce57beee409` (round-2 code BEFORE the re-pin change, not reviewed) | arm A ran its 30 min; arm B not run; session end `arm A INVALID` | **INVALID**: server affinity drift (all 15 llama-server threads on cpus 0–7 in all 62 watcher scans from t = 0.1 s; detector threads also found on 0–7 at the first camera start and near 930 s, re-pinned by `main.py` at camera starts); unreviewed code. Kept as evidence (owner decision 2026-11-15: not resumed); not a measurement |

Console log: `runs/oneshot_console_20261115T002811Z.log`. T1 motivated the owner's re-pin decision (2026-11-15).

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
