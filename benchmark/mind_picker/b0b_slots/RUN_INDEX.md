# MIND-B0b run index

Archived 2026-11-02 by `cp -p` from `~/robot_runs/mind_b0b/` (runs) and `/termux-home/review/mind_b0b/` (review);
every file `cmp` equal; each run's `artifacts.sha256` passes inside the archive. SHA-256 of every archived file:
[ARTIFACTS.md](ARTIFACTS.md). Code `933f4abf…` (= committed `b0b.py`), B0 runner `cb42caaf…`, set `dddb288a…`, base
`cac5348`. Runs from proot Debian with Termux python, Claude Code Coder session resident, nothing else started.

| Run | Launch | Criteria | Labels |
|---|---|---|---|
| `runs/20261102T123725Z_L0` | B0 reference (1 slot, ctx 2048) | K0, K4, K5 PASS | counters and probabilities VALID; RAM INDICATIVE; timings NOT A MEASUREMENT |
| `runs/20261102T123830Z_L1` | + `--slot-save-path` (save/restore slot 0 around R6) | K1, K4, K5 PASS | counters, probabilities and slot file size VALID; RAM INDICATIVE; timings (incl. save/restore) NOT A MEASUREMENT |
| `runs/20261102T123930Z_L2` | `--parallel 2`, ctx 4096 (2048 per slot) | K2, K3, K4, K5 PASS | counters and probabilities VALID; RAM INDICATIVE; timings NOT A MEASUREMENT |

Results:
- L0 reproduces the B0 run of record in all 13 rows (K0).
- L1: after restore, R7 re-processes 41 of 434 tokens (cache_n 393; B0/L0: 428). The slot file is 7,790,524 bytes
  for 422 tokens. Only the picker is kept warm; the separate chat R6 still runs cold.
- L2: R7 on slot 0 re-processes 41 tokens after the chat on slot 1 (K2), R9 reuses 73 of 90 tokens (K3), R10
  re-processes 1. The shared chat R4 on slot 1 runs cold (390 tokens).
- K4: same top action 5/5 in every launch; max |dp| 0.016179 (L0), 0.019413 (L1, L2: warm R7 greet 0.6139 vs fresh
  0.5945), close to the 0.02 limit.
- RAM, L2 − L0 (INDICATIVE, kB): Pss +286,132 after load / +468,008 at the end; VmHWM +308,144; MemAvailable
  +138,272 / +144,492 (higher in L2: background noise dominates).

## Review (`reviews/mind_b0b/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session, source-only static review (no tools; the read-only
sandbox cannot execute in proot, #144 R3): `packet.txt`, `final_review.md`, `review_events.jsonl`,
`review_identity.txt`. Verdict APPROVED WITH NOTES, no P1/P2; three citations verified exact. Owner decision
2026-11-02: accepted; the P3 (the L0 test checks recorded rows, not the request body, for `id_slot`) is recorded, not
fixed; no code change after review. `empty_cwd/` (the reviewer's empty working directory) is not tracked by Git.
`slots/picker.bin` (`*.bin`, .gitignore line 4) and the three `server.log` (`*.log`, line 13) are added with
`git add -f`; .gitignore is unchanged.
