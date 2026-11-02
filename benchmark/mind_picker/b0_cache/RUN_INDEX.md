# MIND-B0 run index

Archived 2026-11-02 by `cp -p` from `~/robot_runs/mind_b0/` (runs) and `/termux-home/review/mind_b0/` (reviews);
every file `cmp` equal; each run's `artifacts.sha256` passes inside the archive. SHA-256 of every archived file:
[ARTIFACTS.md](ARTIFACTS.md). Frozen set `b0_set.json` SHA-256 `dddb288a…` in both runs. Base `fc6cf1a`.

| Run | Code | Status | Labels |
|---|---|---|---|
| `runs/20261102T100659Z` | `aaa9476a…` (review round 1 candidate) | SUPERSEDED | counters and probabilities VALID; timings NOT A MEASUREMENT |
| `runs/20261102T110006Z` | `cb42caaf…` (review round 2 candidate = committed code) | RUN OF RECORD | counters and probabilities VALID; timings NOT A MEASUREMENT |

Both runs: 13/13 requests, 0 errors, one server PID, exit code 0, no SIGKILL; C1, C2, C4, C5 PASS, C3 INFO. Prompt-ids
hash, counts, LCP, prompt_n, cache_n, generated ids, replies and probabilities are identical in all 13 requests of
both runs (`reviews/mind_b0/r2_run_comparison.txt`): counters and probabilities are reproduced by both runs. Timings
(prompt_ms, wall) were taken with the Claude Code Coder session resident, without thermal or core control: NOT A
MEASUREMENT.

Results (run of record): picker on a growing transcript re-processed 36 / 52 / 44 tokens (R3 / R5 / R8), an identical
repeat 1 (R2); the shared chat 15 (R4) and the next picker reused all 370 tokens of T(1) (R5); after the separate chat
the picker rebuilt 428 of 434 tokens (R7). Cached vs fresh: same top action 5/5, max |dp| 0.016179 (<= 0.02).
Observation: the shared chat replied `greet` (the action catalogue in the shared system turn steers the chat reply).

## Reviews (`reviews/mind_b0/`)

Reviewer Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session each round, source-only static review (no tools;
Codex's read-only sandbox cannot execute in proot, #144 R3), so neither round is a pass by execution.

| Round | Records | Verdict |
|---|---|---|
| R1 | `packet.txt`, `final_review.md`, `review_events.jsonl` | CHANGES REQUESTED (P2: exceptions after the /completion POST skip the report and manifest) |
| R2 | `packet_r2.txt`, `final_review_r2.md`, `review_events_r2.jsonl` | CHANGES REQUESTED (P2: an interrupt or a server-stop error can skip report.txt and artifacts.sha256) |

Owner decision 2026-11-02: R2 accepted as review of record with a task-specific exception: its open P2 is a known
limitation, not fixed (completed rows stay in requests.jsonl); no round 3; no code or README change after the
review. `round1_files/` are the round-1 candidate files reconstructed from `packet.txt` and verified against
`frozen_candidate.sha256`. `empty_cwd/` and `empty_cwd_r2/` (the reviewer's empty working directories) are empty and
therefore not tracked by Git. The two `server.log` files are added with `git add -f` (global `*.log` ignore rule).
