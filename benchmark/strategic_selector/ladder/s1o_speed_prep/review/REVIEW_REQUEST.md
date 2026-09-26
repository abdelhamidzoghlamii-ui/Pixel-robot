# Review request: s1o speed variants for the ladder benchmark (Pixel Robot, research only)
ROLE: Reviewer. PONYTAIL: off. Independent correctness/safety review. Do NOT edit, create, stage, commit or push
any file, do not run motors, main.py, llama-server or the benchmark, and do not launch another reviewer.
Read only the files in this directory (an isolated frozen copy). Return one final review.

## Task given to the Coder
Speed up s1o (Gemma 4 E2B Q4_K_M on llama.cpp build 1609, letter scoring: 77% correct on the ladder, 7.6 s median,
~94 ms/token prompt eval). Build ladder adapters for: s1o_b1609 (current), s1o_b2351, Gemma 4 E2B Q4_0 on the faster
build, Qwen3.5-2B and 0.8B on the faster build, threads 3 and 4, batch/ubatch so the whole prompt is one batch; same
letter-scoring logic, no prompt cache; no llama.cpp rebuild. Runner: --thermal-log PATH reads the last line of the root
thermal log (z9/z10/z11 millidegrees) at start and end of each block and prints them; a block starts only when z9 is
within 2 degC of the idle reading taken at run start. Quick 5-case check per variant (timings don't count).
run_s1o_speed.sh runs all variants on the full ladder, cold loads included, with the thermal gate.

## Base
v3/ and ladder/ are not under git (v3/ is untracked in the robot repo; ladder/ is not a repo). orig/ holds the
pre-change copies; *.diff are `diff -u orig candidate`. candidate/ has the full changed files plus ladder_worker.py
(unchanged, context) and the new run_s1o_speed.sh. candidate.sha256 pins them.

## Design choices to check
- adapters.py: S1O.LLAMA / GGUF read S1O_LLAMA_BIN / S1O_GGUF env (defaults unchanged). Existing S1O_SERVER_ARGS is
  appended to the server command. Scoring code untouched.
- ladder.py VARIANTS: env per variant; MODELS/WEIGHTS extended so each variant reuses the s1o adapter/venv.
- b2351 args: --flash-attn off (b2351 CPU FA segfaults on >=64-token Gemma prompts; see INVENTORY.md), --cache-ram 0,
  --ctx-checkpoints 0 (no host prompt cache / SWA checkpoints; requests already send cache_prompt=false),
  -b/-ub 512 pinned (defaults; longest ladder prompt is 173 tokens). No separate batch variant: it would be identical.
- Threads 3/4 come from the existing ladder --threads 3,4 (main block at 4 threads, cached t3/t4 blocks), for every variant.
- Thermal gate: after the cold-load prompt, before the worker spawns; refuses a log whose last line is >30 s old.
- quick.py (check only) no-ops check_cores because Termux was on cores 0-5 during the check; timings from it don't count.
  Quick runs used a synthetic thermal log refreshed every 5 s.

## Checks run
test_output.txt (python3 test_ladder.py), quick_run2/3 stdout+stderr (5 cases x 2 orders + t3/t4, every variant).
b1609 vs b2351 (FA off) main-block choices: 10/10 identical, max top-probability difference 0.036.

## Please report
Findings ranked by severity with file:line, concrete failure scenario, and whether it invalidates the benchmark
(accuracy or timing comparability across variants, thermal gate correctness, cold-load handling, process cleanup).
End with a verdict: APPROVE / APPROVE WITH NOTES / CHANGES REQUESTED.
