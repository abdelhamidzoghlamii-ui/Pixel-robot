# Re-review: fixes to the s1o speed-variant ladder runner (Pixel Robot, research only)
ROLE: Reviewer. PONYTAIL: off. Independent correctness/safety review. Do NOT call any tool or run any command,
do not edit, create, stage, commit or push anything, do not launch another reviewer. All files are inlined below;
answer from this text only.

## Context
The previous candidate (s1o speed variants + --thermal-log gate) was reviewed APPROVE WITH NOTES with 4 findings.
The human approved applying all four, with this clarification of finding 1: the intended rule is "start when
z9 <= idle + 2 degC" (the earlier "within 2 degC" wording was an error). Requested changes:
1. Gate only on "too hot": start when z9 <= idle + 2 degC.
2. Thermal gate first, then the cache-drop prompt, so the block starts right after the drop.
3. Timeout-based kill in Worker.close (WNOHANG loop, then killpg).
4. Unlink the temp log in the test.
Wording in ladder.py help/comments and run_s1o_speed.sh was updated to match 1. adapters.py is unchanged since the
last review.

## Base
ladder/ is not a git repo. Diffs are `diff -u <previously reviewed file> <candidate>`. Full candidate ladder.py,
test_ladder.py and run_s1o_speed.sh are inlined for context. Worker processes are started with
start_new_session=True (worker pid == process group id; llama-server is in the same group).

## Checks run
test_output.txt: python3 test_ladder.py (includes new checks: cooler-than-idle starts at once; 2.1 degC over idle
waits, 2.0 passes; Worker.close on a child that ignores EOF returns None within 5 s and reaps it).
quick_run4: 5 ladder cases x 2 orders + t3/t4 blocks on s1o_b2351_qwen08b with a synthetic, continuously refreshed
thermal log (timings don't count; core check disabled by a scratch wrapper because Termux was on cores 0-5).
wait4 maxrss was recorded for every block (normal close path).

## Please report
Whether each of 1-4 is correctly implemented, any new defect the changes introduce (file:line, concrete failure
scenario), and whether anything invalidates the benchmark. End with: APPROVE / APPROVE WITH NOTES / CHANGES REQUESTED.
