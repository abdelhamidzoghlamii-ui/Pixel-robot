# MIND-B2 run index

Archived 2026-11-02 by `cp -p` from `/termux-home/mind_b2_test/` (set), `~/robot_runs/mind_b2/` (run) and
`/termux-home/review/mind_b2/` (review, all subfolders); every file `cmp` equal; the run's `artifacts.sha256` passes
inside the archive. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md). Code `c5f0b2b4…` (= committed
`b2.py`), B1 runner `26b10c65…`, B0 runner `cb42caaf…`, DEV `b1abba5d…`, base `4d0deb8`. Run from proot Debian with
Termux python, Claude Code Coder session resident, nothing else started.

## Set (`set/b2_test.json`)

40 owner-labelled TEST items, SHA-256 `d5faed98fb424ce8ca959ae58fe6f0ca1cd16d85bb7ed30f7cff8f74cd9f7b1f`, written
2026-11-02T14:29:08Z before any Gemma call, kept outside the repo and opened by the runner only after `selection.json`
was written and its hash logged (WORKFLOW test-set custody). Owner decision 2026-11-02: scoring is finished, TEST
leaves custody; it is no longer an independent test set.

## Run

| Run | Content | Result | Labels |
|---|---|---|---|
| `runs/20261102T143251Z` | one launch (setup_q4 flags, ctx 2048); DEV blocks V1/V2/V3 × FWD/REV, selection, TEST V3-FWD, V3-REV; 40 items each | V3 selected; TEST PASS | accuracy VALID (320/320 rows, one PID, exit 0, both set hashes unchanged); timings NOT A MEASUREMENT |

| DEV | Strict FWD / REV | Acceptable FWD / REV | Flips |
|---|---|---|---|
| V1 rules | 0.675 / 0.675 | 0.775 / 0.825 | 5 |
| V2 rules + examples | 0.750 / 0.750 | 0.825 / 0.850 | 3 |
| V3 rules + examples + reason | 0.775 / 0.725 | 0.875 / 0.850 | 4 |

- Selection (fixed rule): V1 excluded (flips 5 > 4); V3 mean acceptable 0.8625 > V2 0.8375 → **V3**, no FLIPS flag;
  `selection.json` SHA-256 `684addfa…`.
- TEST, V3 (official): strict 0.900 / 0.900, acceptable 0.900 / 0.900 (36/40 each order), flips 3 [6, 36, 39] →
  **PASS** (acceptable ≥ 0.80 and strict ≥ 0.65 in both orders, flips ≤ 4). Wrong in both orders: 2 ("Leave me
  alone" → wait), 7 (memory: dislikes interruptions → approach), 36 ("Tell me about yourself" → explore/look).
- TEST partly resembles DEV items and the worked examples (e.g. "Nothing new happened."): the result holds for this
  benchmark only; V3's advantage over V1/V2 is shown on DEV only.

## Review (`reviews/mind_b2/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session, source-only static review (no tools; the read-only
sandbox cannot execute in proot, #144 R3). Round 1 (thread `01a19cc3…`, 2026-11-02 14:56): **APPROVED WITH NOTES**,
two P3 (selection `rule` text keeps a docstring fragment, `b2.py:255`; the carried-over B0 interrupt/cleanup
limitation, `b2.py:470`, `482`); citations verified exact (488 lines). Review of record: `final_review.md`,
`review_identity.txt`, `snap_pre.txt` = `snap_post.txt`.

Owner decision 2026-11-02, task-specific exception: the Coder (after a context clear) reused this folder and
overwrote round 1's packet, events, invocation and pre-call evidence files; the top-level `packet.txt`,
`review_*.txt|jsonl`, `invocation.txt`, `base.txt`, `git_pre.txt`, `frozen_candidate.sha256`, `task.txt`,
`test_output.txt`, `run_artifacts_check.txt`, `build_packet.py`, `item3_v3.txt` and `test_compare.*` belong to round 2.
Round 1's packet and events are recovered from the Codex session log in `round1_recovered/` (model `gpt-6.1-sol`,
no tool calls, final message identical to `final_review.md`). Round 2 (2026-11-02 15:25) failed: "Selected model is
at capacity", exit 1, no review (`round2_failed/`); no new round. TEST item 3 was rendered for V3 after the run
(`item3_v3_render.txt`, 14:55), not before; recorded, no effect (texts frozen, hashes match). The round-1 packet
claimed TEST was "made read-only"; false (the file is `-rw-r--r--`), caught by the reviewer, recorded. No code change
after review. `empty_cwd/` (the reviewer's empty working directory) is not tracked by Git. `server.log` (`*.log`,
.gitignore line 13) is added with `git add -f`; .gitignore is unchanged.
