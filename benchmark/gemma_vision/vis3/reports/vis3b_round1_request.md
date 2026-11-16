REVIEWER PROFILE
ROLE: Reviewer (independent, read-only)
PLATFORM: Codex CLI (codex exec), inside this isolated copy
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off

Switch Ponytail OFF for this independent correctness and safety review (VIS-3B round 1). READ-ONLY: do not create, modify
or delete any file anywhere (no .pyc, no temp files in this copy). Do not run servers, benchmarks, git mutations, motors,
or another reviewer. You may read /data/data/com.termux/files/home/llama.cpp-b1609-dotprod and llama.cpp-vis read-only.

TASK VIS-3B (Coder: Claude Code claude-opus-5-5, medium, Ponytail lite), owner change of the approved VIS-3 design:
- Replace the multi-phase benchmark with individually selectable tests: cold model loading, warm model loading,
  first-response latency, text generation speed, prompt processing speed, MTP performance, Q8 projector/vision performance.
- Each test accepts a specific configuration, executes only the requested measurement, saves results independently, and
  exits; no automatic next block or mandatory multi-test sequence.
- Separate old and new Gemma measurements; do not repeat a completed test unnecessarily.
- Remove the artificial 32 C gate and long cooling wait; preserve every genuine thermal, battery, CPU-affinity, agent,
  abort and memory safety check.
- Record temperature, CPU frequency/capping, power and memory for every measurement; cross-run comparisons must explicitly
  report whether operating conditions are sufficiently matched.
- Keep it small, reuse VIS-3/VIS-2 code, preserve previous runs. No timed inference by the Coder, no deployed robot code
  changes (server_manager.py, chat.py, navigation, binaries, models), VIS-1/VIS-2 runners unchanged. No commits/pushes.

Context: VIS-3 (multi-phase) was APPROVED WITH NOTES in its round 2 (review_inputs/approved_round2_review.txt). The owner
then ran it: runs/smoke_20261022T153003Z (SMOKE, worked) and runs/20261022T153641Z (gate missed at idle skin 32.95 C).
Those folders must stay untouched and must not count as completed VIS-3B tests.
Base commit: review_inputs/base_commit.txt. Files are git-ignored (benchmark/*): candidate = four files in
benchmark/gemma_vision/vis3/ (review_inputs/candidate.diff); change from the approved VIS-3: review_inputs/approved_to_vis3b.diff;
oneshot.sh vs vis2/oneshot.sh: review_inputs/oneshot_vis2_to_vis3.diff.
Checks run by the Coder (outputs in review_inputs/): test_vis3 17/17 ok; test_vis2 ok; test_vis1 ONE PRE-EXISTING error
(test_wrapper_contract, unrelated); dry_run.txt for all seven tests; real Debian --preflight with only the agent check
stubbed printed PREFLIGHT OK for gen/new and vision/new and refused vision/old. You may rerun:
cd benchmark/gemma_vision/vis3 && TMPDIR=/tmp python3 -B test_vis3.py

Review for correctness and safety: whether each test measures only what it claims (cold vs warm vs first-response
semantics, priming, warm-up, measurement intervals for frequency/power/caps), old/new fairness (anything besides the binary
differing), that no genuine safety check was weakened relative to VIS-2/VIS-3 (only the 32 C gate, the idle+1.5 gate and the
5-minute idle are intentionally removed), stop/cleanup and owned-process handling, labelling, the repeat refusal
(fingerprint, false refusals/false acceptances), the condition-matching report, preflight (native and Debian), oneshot
argument handling and folder naming, and test adequacy.
Output: state each reviewed file's line count; for every file:line you cite, quote the exact line. Findings ranked
P0-P3 with concrete failure scenarios. End with exactly one verdict line: APPROVED, APPROVED WITH NOTES, or CHANGES REQUESTED.
