# Review request: #110 rubric fixes, blind-sheet generation, conversation speed script (Pixel Robot, research only)
ROLE: Reviewer. PONYTAIL: off. Independent correctness/safety review. Do NOT call any tool or run any command, do not
edit, create, stage, commit or push anything, do not run motors, and do not launch another reviewer. All material is
inlined below; answer from this text only.

## Task (human)
Compare conversation quality and speed of gemma_e2b_q4km (robot default), gemma_e2b_q40, gemma_e4b_q4km,
qwen35_4b (think off), qwen35_2b (think off), all on the rebuilt b1609-dotprod binary with server_manager.py's flags.
1. Fix the three rubric defects STATUS lists for the DECISIONS #110 benchmark, and nothing else in it:
   reject_violation must cover find_person and any other action type (not a hardcoded subset); 'refusal' must be
   defined explicitly before running; runs must record the grader's source_sha256.
2. Quality, every bucket, every model: auto-graded bucket C scored with the fixed rubric; conversation (A) and Q&A (B):
   a blind sheet, one section per prompt, answers shuffled and labelled A-E, key in a separate file.
3. Speed per model (cold and cached load, time to first token, prompt tok/s, generation tok/s, peak RSS, thermal
   start/end) as /termux-home/ladder/run_conversation.sh using the ladder runner's thermal gate, core guard, cold
   handshake and --resume. The human launches it with: RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh
4. Quick check: 2 prompts per model.

## Design notes
- runs/bench.py is archived evidence (its SHA-256 1e1b506c... is recorded in ARTIFACTS.md and DECISIONS #110), so it
  is left byte-identical; the fixed harness is a copy, benchmark/llm_objective_setting/bench.py. The diff is
  runs/bench.py -> bench.py. Refusal definition chosen: structural, identical to aggregate.py's post-hoc corrected
  rule ([] or exactly one say; message wording not auto-graded). The regrade reproduces #110's all-run Gemma figures
  (E2B 32/39 exact, 3 violations; E4B 30/39, 3) - see test_bench.py.
- bench.py's extractor still accepts an array nested inside a bare object (known, documented in the archive README,
  not one of the three defects) - so score_c.py reports three parse regimes like #110: harness, aggregate strict,
  aggregate lenient.
- conv_quality.py runs bench.do_run unchanged, overriding at runtime only: the model table, binary, server flags
  (server_manager's, pinned to cores 4-7), output dir, stop (only its own server; bench's pkill would hit others),
  and zone9 readings (from the root logger's thermal.log; bench's own su read fails in proot).
- oneshot.sh (human's native-Termux launcher, outside the repo) hardcoded run_s1o_speed.sh; RUN_SCRIPT support was
  added so the requested launch command works (default unchanged).
- ladder.py is unchanged (approved earlier); the functions conv_speed.py uses are inlined as context.

## Checks run
tests.txt: test_bench.py, test_aggregate.py (existing), test_ladder.py (existing). Quick check: 2 prompts (A1, C8)
per model through conv_quality (bench.do_run) and conv_speed (both blocks, live thermal log; core guard faked because
Termux was on cores 0-5; timings do not count), plus score_c.py and make_blind_sheet.py on the quick results.

## Please report
Correctness of the three fixes and that nothing else in the benchmark changed; whether the blind sheet is truly blind
and the key separate; correctness of the speed measurements (TTFT, tok/s, load time, VmHWM, thermal, cold/cached),
the gate/guard/handshake/--resume integration and process cleanup; anything invalidating the comparison between
models. End with APPROVE / APPROVE WITH NOTES / CHANGES REQUESTED.
