# MIND-BV run index

Archived 2026-11-03 by `cp -p` from `/termux-home/mind_bv_key/20261103T100550Z/` (key/),
`~/robot_runs/mind_bv/20261103T100914Z/` and `/termux-home/mind_bv_stdout_20261103T100907Z.txt` (runs/) and
`/termux-home/review/mind_bv_20261103T105258Z/` (reviews/); every file equal (`diff -r`, `cmp`); the run's
`artifacts.sha256` passes inside the archive. Before archiving, `bv.py` (`987bb525…`, = run.json `code_sha256`),
`test_bv.py` (`7e3d13fa…`) and `README.md` (`9354cb34…`) matched the review's `frozen_candidate.sha256`. SHA-256 of
every archived file: [ARTIFACTS.md](ARTIFACTS.md). Base `c7db848`. Run from proot Debian with Termux python, Claude
Code Coder session resident, nothing else started.

## Run

| Run | Content | Result | Labels |
|---|---|---|---|
| `runs/20261103T100914Z` | launch A (b11716 + Q8_0 projector, 70 image tokens): V70, VY70; launch B (140 tokens): V140; Y from M2-Q1 results.json | **FAIL**; exit 0 / 0 after SIGTERM, no SIGKILL | accuracy VALID (183/183 rows, one PID per launch, 0 errors, frame and key hashes unchanged); timings NOT A MEASUREMENT |

Console log of the run: `runs/mind_bv_stdout_20261103T100907Z.txt`.

| Method | Room (61) | Person (61) | Distance (40) | All three (61) | UNPARSED room/person/distance |
|---|---|---|---|---|---|
| Y (YOLO + rules) | 44 (0.721) | 61 (1.000) | 18 (0.450), 4 UNAVAILABLE | 29 | 0/0/0 |
| V70 | 31 (0.508) | 61 (1.000) | 16 (0.400) | 9 | 0/0/0 |
| V140 | 40 (0.656) | 61 (1.000) | 19 (0.475) | 17 | 0/0/0 |
| VY70 | 39 (0.639) | 61 (1.000) | 19 (0.475) | 11 | 0/0/0 |

PASS rule (fixed before the first call): the best vision method by room accuracy has room >= 0.80 and person >= Y's.
Best is V140 with 0.656 → **FAIL**. Gemma said "close" for most person photos and never "far" (0/10 far photos in
every method); on photos without a person it rarely answered distance "none" (V70 0/21, V140 3/21, VY70 2/21);
"unknown" was its main room error; hallway was named once (V140, 1 of 4).

Detector behind Y (M2-Q1 `manifest.json` + `detect_person.py`, same SHA-256 `2420d70b…` as in that run):
`yolo11s_640.onnx` (`dc9fe7c5…`), `detect(frame, 640)` = 640×640 plain resize, confidence > 0.35, per-class NMS
IoU 0.45.

## Key (`key/20261103T100550Z/bv_key.json`, SHA-256 `456f633b…`)

Held outside the repo until scoring finished (pinned in `bv.py`, refused inside the repo); archived on the owner's
decision. Caveats:
- Labels are owner-confirmed draft labels, not an independent annotation.
- Several distances are estimated, not measured.
- The photos are mostly not taken at robot height.
- The `roomsig_*` photos (ids 24–38) were chosen for objects that favour Y's room rules.
- Owner decisions before the first call: 40 person photos (the task text said 44); photos with one detection keep
  only `classes` in results.json, so Y's distance there is UNAVAILABLE (scored wrong: ids 10, 12, 43, 44) and VY70
  lists them as `<class> (position unknown, distance unknown)`.

## Review (`reviews/mind_bv_20261103T105258Z/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session (thread `01a1a10b…`, model and effort confirmed in the
session log), source-only static review (no tools; #144 R3), 2026-11-03 10:54:22–10:55:52Z, exit 0, empty stderr,
0 command executions, `snap_pre.txt` = `snap_post.txt`: **CHANGES REQUESTED**, one P2 (validity checks the row count
per launch, not exactly one row per launch/method/id; `bv.py:242`, `:221`, `:229`) and one P3 (carried-over B0
interrupt/cleanup limitation, README.md:22–23); citations verified exact (393 and 23 lines). The reviewer found no
discrepancy in the scores and confirmed FAIL. Review of record: `final_review.md`, `review_identity.txt`.

Owner decision 2026-11-03: review accepted as review of record with a task-specific exception: the P2 is recorded,
not fixed (it did not affect this run; the reviewer confirmed the expected sequences); no code change after review.
The Coder's two open choices are accepted as recorded: parsing strips `*` and one trailing `.`; on a room-accuracy
tie, PASS if any tied vision method meets both conditions. The files are committed with `git add -f`
(`.gitignore:44` `benchmark/gemma_vision/*`, unchanged). `empty_cwd/` (the reviewer's empty working directory) is not
tracked by Git.
