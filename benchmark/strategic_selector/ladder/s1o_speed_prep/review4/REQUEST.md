# Re-review: unattended-run changes to the s1o ladder runner (Pixel Robot, research only)
ROLE: Reviewer. PONYTAIL: off. Independent correctness/safety review. Do NOT call any tool or run any command,
do not edit, create, stage, commit or push anything, do not launch another reviewer. All files are inlined below;
answer from this text only.

## Context
The previous candidate was APPROVED. The human ran it unattended via oneshot.sh (native Termux; runs
run_s1o_speed.sh inside proot Debian as a bash coproc, answering "COLD LOAD NEXT" lines by dropping caches and
writing a newline to the coproc's stdin). Findings from that run: (a) no cold prompt appeared, because
make_cold only prompts when sys.stdin.isatty() and the coproc's stdin is a pipe, so both main blocks silently
took the weights-cold fallback (posix_fadvise DONTNEED); (b) the run aborted with "cores 4-7 not all allowed
after t3" after b2351 t3 (Termux left the top-app cpuset); (c) the +2 degC gate cost ~6 min per block.

## Requested changes (human)
1. Cold-load handshake when env LADDER_COLD_HANDSHAKE=1: create /termux-home/ladder/.drop_request, wait up to
   60 s for .drop_done, read "ok" or "failed", delete both, label the load full-cold or weights-cold as now.
   Without the env var keep the current prompt.
2. Thermal gate: z9 <= idle + 4 degC.
3. Losing cores 4-7: don't abort. Pause, recheck every 10 s, and when they're back discard and redo the
   interrupted block.
4. --resume: skip blocks already completed with valid data; b2351 t3 in the last run is invalid and must rerun.
5. b1609: main block only (the baseline); drop its t3/t4.
6. Print a partial report from what the last run completed (done with a scratch script partial.py using the
   runner's own load_done/analyse/report; inlined).

## Implementation notes
- Rows are now buffered per block and written to decisions.jsonl only after the block completes with cores held
  (checked before, after the cold step, after every decision, and after the block); each completed block's
  metadata is appended to blocks.jsonl. --resume keeps blocks listed in blocks.jsonl with a full row set. A run
  directory from before blocks.jsonl existed (the last real run) has no completion marker: a block counts there if
  all its rows are present and a later block started (the old runner started the next block only after the
  post-block core check passed). Such legacy blocks have no load/RAM metadata (reported as n/a/None). On resume the
  original decisions.jsonl/blocks.jsonl are renamed *.before_resume_<UTC> and rewritten with only the kept blocks.
- run.json (cases path, sha256, levels) is written; --resume refuses a mismatch when it exists.
- The native-Termux watcher for the handshake is not part of this change (oneshot.sh is the human's file). The
  intended watcher writes .drop_done atomically (write .tmp, mv).

## Checks run
test_output.txt (unit checks incl. handshake ok/failed/timeout, CoresLost, block_plan, load_done legacy and new
format, report for a main-only legacy model). e2e_*.txt: 5-case end-to-end run with simulated core loss mid-block
(cores faked via a flag file because Termux is on cores 0-5 now), a fake watcher answering "failed" (no root in
proot), SIGTERM during a t3 block, then --resume. partial_report.txt: partial report of the last real run.

## Please report
Whether each of 1-5 is correctly implemented, defects (file:line, concrete failure scenario), anything that would
invalidate the benchmark or lose/mix data across resume, and whether the legacy-completion rule is sound.
End with: APPROVE / APPROVE WITH NOTES / CHANGES REQUESTED.
