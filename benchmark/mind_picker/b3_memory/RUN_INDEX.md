# MIND-B3 run index

Archived 2026-11-02 by `cp -p` from `/termux-home/mind_b3_set/20261102T193300Z/` (set),
`~/robot_runs/mind_b3/20261102T193348Z/` and `/termux-home/mind_b3_stdout_20261102T193340Z.txt` (runs/),
`/termux-home/review/mind_b3_render_20261102T193310Z/` and `/termux-home/review/mind_b3_20261102T201750Z/` (reviews/);
every file `cmp` equal (`diff -r`); the run's `artifacts.sha256` passes inside the archive. Before archiving, `b3.py`
(`bcead2d6…`), `test_b3.py` (`576116ce…`) and `README.md` (`14ba424e…`) matched the review's
`frozen_candidate.sha256`. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md). B2 runner `c5f0b2b4…`, B1
`26b10c65…`, B0 `cb42caaf…`, base `fe8d2de`. Run from proot Debian with Termux python, Claude Code Coder session
resident, nothing else started.

## Set (`set/b3_set.json`)

20 owner-written bases × 6 conditions = 120 rows, SHA-256 `15654298…a0cf4217ab`, written 2026-11-02T19:33:00Z before
any Gemma call, outside the repo. Owner decision 2026-11-02: scoring is finished, the set leaves custody; it becomes
DEV for a later memory test.

## Run

| Run | Content | Result | Labels |
|---|---|---|---|
| `runs/20261102T193348Z` | one launch (setup_q4 flags, ctx 2048, b1609 tree); B2 V3 picker, 6 conditions × FWD/REV, 20 items each | overall **FAIL** | accuracy VALID (240/240 rows, one PID 23484, exit 0, no SIGKILL, set hash unchanged); timings NOT A MEASUREMENT |

Console log of the run: `runs/mind_b3_stdout_20261102T193340Z.txt`.

| Condition | Acceptable FWD / REV | Change vs NONE | Flips | Bar |
|---|---|---|---|---|
| NONE | 14 / 14 | 0 / 0 | 1 | none |
| RELEVANT | 7 / 6 | −7 / −8 | 5 | ≥ 16: FAIL / FAIL |
| MIXED | 6 / 5 | −8 / −9 | 2 | ≥ 15: FAIL / FAIL |
| IRRELEVANT | 12 / 15 | −2 / +1 | 5 | ≥ 12: PASS / PASS |
| STALE | 9 / 12 | −5 / −2 | 6 | ≥ 12: FAIL / PASS |
| MISLEADING | 14 / 14 | 0 / 0 | 2 | ≥ 12: PASS / PASS |

Strict equals acceptable in every block. Memory-citing reasons under NONE: 1 [14] each order (misapplied Rule 5);
IRRELEVANT 0. Relevant notes made the picker worse, not better.

## Review (`reviews/mind_b3_20261102T201750Z/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session (thread `01a19dea…`), source-only static review (no
tools; #144 R3), 2026-11-02 20:19–20:20, exit 0, `snap_pre.txt` = `snap_post.txt`: **APPROVED WITH NOTES**, one P3
(carried-over B0 interrupt/cleanup limitation, `b3.py:398`, `399`); citations verified exact (405 lines). Review of
record: `final_review.md`, `review_identity.txt`. `mind_b3_render_20261102T193310Z/` holds the offline render of MIXED
base 8 made before the run (equal to the copy in the review folder). No code change after review. `empty_cwd/` (the
reviewer's empty working directory) is not tracked by Git; `server.log` is added with `git add -f`.

Owner decision 2026-11-02: memory stays out of the picker for now (to be improved later).
