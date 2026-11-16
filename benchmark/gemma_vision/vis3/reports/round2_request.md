REVIEWER PROFILE
ROLE: Reviewer (independent, read-only)
PLATFORM: Codex CLI (codex exec), inside this isolated copy
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off

Switch Ponytail OFF for this independent correctness and safety review (round 2). READ-ONLY: do not create, modify or
delete any file anywhere (no .pyc, no temp files in this copy). Do not run servers, benchmarks, git mutations, motors,
or another reviewer. You may read /data/data/com.termux/files/home/llama.cpp-b1609-dotprod and llama.cpp-vis source read-only.

TASK VIS-3 (Coder: Claude Code claude-opus-5-5, medium, Ponytail lite). Objective: determine whether the vision-capable
llama.cpp build (~/llama.cpp-vis, b11716) can match the robot build (~/llama.cpp-b1609-dotprod, b1609) in startup, prompt
processing, generation and MTP. Scope A: diagnose build/config differences (claims in review_inputs/DIAGNOSIS.md).
Scope B: standalone non-deployed harness benchmark/gemma_vision/vis3/: OLD b1609 vs NEW b11716, same GGUF, no projector,
no MTP (O/N), optional MTP on both (OM/NM); ctx 2048, 4 threads, identical affinity, prompts, sampling, cache settings,
output budgets; cold and warm loading separately, readiness, first-response latency, prompt/gen speed, PSS, MemAvailable,
power, skin, Android status, CPU frequencies/caps, exit; short blocks, raw evidence, blocks runnable individually; keep
thermal safeguards; offline tests + preflight. Owner runs timed benchmarks later; the Coder ran none.
Restrictions: no change to server_manager.py, chat.py, robot navigation, deployed binaries, model files, VIS-1/VIS-2 runners;
no commits/pushes; no weakening of thermal protections.

Base commit: review_inputs/base_commit.txt (685a44b). VIS-1/VIS-2/VIS-3 files are git-ignored (benchmark/*), so the
candidate is four NEW files: review_inputs/candidate.diff (vis3_run.py, test_vis3.py, oneshot.sh, README.md);
oneshot.sh vs vis2/oneshot.sh: review_inputs/oneshot_vis2_to_vis3.diff. Reused unchanged: vis2/vis2_run.py, vis1/vis1_run.py.
Checks run by Coder (outputs in review_inputs/): test_vis3 17/17 ok; test_vis2 ok; test_vis1 has ONE PRE-EXISTING error
(test_wrapper_contract expects a line no longer in vis1/oneshot.sh; untouched by this task); dry_run.txt; Debian
--preflight refused with "agent found" (expected, agent resident); with the agent check stubbed it printed PREFLIGHT OK and
refused a set LD_LIBRARY_PATH. You may rerun: cd benchmark/gemma_vision/vis3 && TMPDIR=/tmp python3 -B test_vis3.py
(only TemporaryDirectory writes under /tmp, nothing in this copy).

Review for correctness and safety: fairness of the OLD/NEW comparison (anything that differs besides the binary), cold/warm
logic, thermal gate (strengthened, never weakened vs VIS-2), stop/cleanup paths and owned-process handling, validity
labelling, summary/comparability logic and pooling, preflight, oneshot adaptation, test adequacy, and the DIAGNOSIS claims.
Output: state each reviewed file's line count; for every file:line you cite, quote the exact line. Findings ranked
P0-P3 with concrete failure scenarios. End with exactly one verdict line: APPROVED, APPROVED WITH NOTES, or CHANGES REQUESTED.

ROUND 2. Round 1 (fresh session, same profile) returned CHANGES REQUESTED with five P2 findings (verbatim:
review_inputs/round1_last_message.txt). Changes since round 1: review_inputs/round1_to_round2.diff.
1 env overrides: check_env() refuses LLAMA_*, GGML_*, OMP_*, KMP_*, LD_LIBRARY_PATH, LD_PRELOAD in the Debian preflight and
  in the runner before any server; manifest 'commands' added to the pooling keys; tests test_env_overrides_refused and the
  command-mismatch case in test_summarize_pools_folders. Explicit -b/-ub/-fa flags were NOT added so the command stays
  identical to server_manager setup_q4 (test_commands_match_deployed_flags).
2 warm gate: gate(bridge, idle) runs again before the warm launch (no cache drop); summary reports per-phase start skin
  and cold/warm start_skin_spread.
3 mapped libs: a failed check raises immediately (test_run_block_rejects_foreign_libraries).
4 sampler race: detach() clears target.json and waits (<=5 s) for a monitor sample begun after the clear before
  own_stop; result recorded as phase['sampler_detached'] (test_detach_waits_for_a_later_sample).
5 preflight checks find_spec('requests') in Debian (test_preflight_requires_requests).
README updated; DIAGNOSIS.md item 2 wording corrected per the round-1 note.
Verify the fixes and look for regressions they introduce, then review the whole candidate again.
