# MIND-B1 run index

Archived 2026-11-02 by `cp -p` from `/termux-home/mind_b1_set/` (set), `~/robot_runs/mind_b1/` (run) and
`/termux-home/review/mind_b1/` (review); every file `cmp` equal; the run's `artifacts.sha256` passes inside the
archive. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md). Code `26b10c65…` (= committed `b1.py`), B0
runner `cb42caaf…`, `picker.py` `b152947b…`, B0 set `dddb288a…`, base `d0c9cd5`. Run from proot Debian with Termux
python, Claude Code Coder session resident, nothing else started.

## Set (`set/b1_set.json`)

40 owner-labelled items, SHA-256 `b1abba5daa3cb87c30a225af40d3e19e6f0feaf82ba95af48f8b48c4a23461eb`, recorded
2026-11-02T13:10:33Z before any Gemma call and kept outside the repo until scoring was finished (WORKFLOW test-set
custody). Owner decision 2026-11-02: scoring is finished, the set leaves custody and becomes a DEV set; it is no
longer an independent test set.

## Run

| Run | Content | Result | Labels |
|---|---|---|---|
| `runs/20261102T131422Z` | one launch (setup_q4 flags, ctx 2048), blocks A-FWD, A-REV, B-FWD, B-REV, 40 items each | A FAIL, B FAIL | accuracy VALID (160/160 rows, one PID, exit 0, set hash unchanged); timings NOT A MEASUREMENT |

| Arm × order | Strict | Acceptable | Copy rate | Near-ties |
|---|---|---|---|---|
| A-FWD (old, `picker.Gemma`) | 0.350 (14/40) | 0.500 (20/40) | 6/32 | 20, 24 |
| A-REV | 0.350 (14/40) | 0.525 (21/40) | 6/32 | 10, 23 |
| B-FWD (new, B0 layout) | 0.475 (19/40) | 0.700 (28/40) | 6/32 | 31 |
| B-REV | 0.500 (20/40) | 0.700 (28/40) | 7/32 | 1, 27 |

- Order flips: A 19/40, B 6/40. PASS rule (acceptable ≥ 0.80 and strict ≥ 0.65 in both orders, flips ≤ 4): both
  arms fail every component.
- B is better than A on every metric. Recall of B: visit 0/7 in both orders, look 2/7; it picks greet for a far or
  calling person (items 5, 20, 25) and explore on entering a room (13, 28).
- B's cache: cache_n 308–328 after the first B request (that one follows the uncached A blocks: cache_n 1).

## Review (`reviews/mind_b1/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session, source-only static review (no tools; the read-only
sandbox cannot execute in proot, #144 R3): `packet.txt`, `final_review.md`, `review_events.jsonl`,
`review_identity.txt`. Verdict CHANGES REQUESTED: one P1 — `inside_repo()` (`b1.py:60–61`) does not resolve
symlinks, so an outside symlink to a set inside the repo would pass the custody check; the run used the real path
and is not affected. Metrics, PASS rule, rendering and the set's 40 items confirmed. Citation verified exact. Owner
decision 2026-11-02: accepted as review of record with a task-specific exception; the P1 is recorded, not fixed here,
and will be fixed in the next runner (MIND-B2); no code change after review. `empty_cwd/` (the reviewer's empty
working directory) is not tracked by Git. `server.log` (`*.log`, .gitignore line 13) is added with `git add -f`;
.gitignore is unchanged.
