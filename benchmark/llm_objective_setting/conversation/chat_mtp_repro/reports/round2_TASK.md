# VIS-3C / Experiment 1 — independent review request (frozen candidate)

EXECUTION PROFILE
ROLE: Reviewer
PLATFORM: AGY / Antigravity CLI, fresh independent session
MODEL: gemini-3.8-flash-medium
EFFORT: N/A (medium is part of the AGY model identifier)
PONYTAIL: off
WHY: Independent review of benchmark fidelity, safety and measurement methodology.
FALLBACK: Codex CLI, gpt-6.1-sol, medium, Ponytail off — owner approval required.

Ponytail is OFF: do not load or apply any Ponytail skill or command. This is a correctness, fidelity and safety review.

## Task given to the Coder (summary of the owner's authorization)

Prepare the smallest safe, independently runnable benchmark that faithfully reproduces the historical Oct 14 MTP
benchmark (run `benchmark/llm_objective_setting/conversation/chat_mtp/runs/chat_mtp_20261014T224541Z`, DECISIONS #137:
Gemma 4 E2B Q4_0 on llama.cpp b1609-dotprod, MTP OFF 13.37 tok/s, ON 19.18 tok/s, acceptance 39.0 %, 13 prompts,
ctx 4096, temperature 0.7, n_predict 512, cores 4-7). Requirements:
1. Reconstruct and freeze the historical server command, model selection, MTP flags, prompt template, payload
   construction and timing interpretation from archived evidence (not by importing today's chat.py, which changed).
2. Check archived source hashes and current model/drafter hashes; identify what cannot be reproduced exactly.
3. Use the original b1609-dotprod binary without changing robot deployment.
4. Preserve the 13-prompt workload, sampling settings, output limits and token-speed aggregation.
5. Preserve sparse monitoring comparable to Oct 14; do NOT reuse VIS-3B's ~0.45 s root-monitoring loop.
6. Safety preflight: native Termux, unplugged battery, adequate charge, thermal safety, CPU affinity, no coding agent
   during measurement, clean server termination.
7. Separate, independently launchable MTP-OFF and MTP-ON measurements; one launch runs only its config; isolated run
   folders; enough metadata for comparability.
8. A report-only command comparing completed OFF/ON folders (thermal conditions, generated tokens, draft acceptance,
   server timings).
9. No arbitrary five-minute wait or unnecessary automatic repetition; thermal safety mandatory.
10. Offline tests with mocks/fixtures only; no model inference.
11. Do not modify or overwrite historical archived runs.
12. Document exactly where the new benchmark differs from the historical run.
Not authorized: timed inference, llama-server, motors, modifying production server_manager.py / chat.py / llama.cpp /
weights, commits, pushes, canonical docs, changing VIS-3B.

Background: an independently reviewed investigation (`_review_context/VIS3B_MTP_INVESTIGATION.md`, review verdict
`_review_context/VIS3B_MTP_REVIEW_ROUND2.json`: APPROVED WITH NOTES) found the drafter functioning in VIS-3B (11.10
tok/s) and identified dense root telemetry as a LIKELY contributor; causation unproven.

## Base and candidate

Base commit: HEAD = origin/main = 685a44be63aa88c30e09386ce8e5e196ebf11abc. The candidate is four NEW untracked
files only, no modified tracked file:
`benchmark/llm_objective_setting/conversation/chat_mtp_repro/{README.md,oneshot.sh,repro.py,test_repro.py}`.
SHA-256 in `_review_context/frozen_candidate.sha256`; full diff in `_review_context/candidate.diff`.

Surrounding source (present in this copy): `benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py`
(imported; its `run_block` is reused), `.../chat_mtp/oneshot.sh` (the historical launcher the new one is derived
from), `.../chat_mtp/test_chat_mtp_runner.py`, `benchmark/coresidency/coresidency.py`, `benchmark/power_map/power_map.py`,
`benchmark/llm_objective_setting/bench.py`, the archived run folder `.../chat_mtp/runs/chat_mtp_20261014T224541Z/`
(`run.json`, `prompts.jsonl`, `blocks.jsonl`, `report.txt`, server logs) and `.../chat_mtp/runs/ARTIFACTS.md`.
Historical sources at commit 2091f42 (the code the Oct 14 run used) are extracted to
`_review_context/chat_2091f42.py`, `server_manager_2091f42.py`, `chat_mtp_2091f42.py`.
Current on-phone hashes of model, drafter and b1609 binaries (Coder, read-only sha256sum): `_review_context/current_hashes.txt`.
Coder's offline check output: `_review_context/checks.txt` (12 new tests OK; existing 35 chat_mtp tests OK).
Workflow rules: `docs/WORKFLOW.md`, `docs/CLAUDE.md`, `docs/REVIEWER.md`.

## Rules for this review (mandatory)

- Read-only. Do not create, edit, move or delete any file anywhere. Do not run git. Do not start llama-server, any
  benchmark, `oneshot.sh`, `repro.py --config`, su, or anything touching hardware. Do not invoke another agent.
- The only commands you may run are these two, from the copy's root, exactly as written (both write nothing):
  1. `/data/data/com.termux/files/usr/bin/python3 -B -c "import sys; [compile(open(f).read(), f, 'exec') for f in sys.argv[1:]]" benchmark/llm_objective_setting/conversation/chat_mtp_repro/repro.py benchmark/llm_objective_setting/conversation/chat_mtp_repro/test_repro.py`
  2. `cd benchmark/llm_objective_setting/conversation/chat_mtp_repro && /data/data/com.termux/files/usr/bin/python3 -B -W ignore -m unittest -v test_repro`
  Running them is optional; reading the code is not.
- Examine the actual code, not only this summary. Verify in particular: the frozen command and payload against the
  archived `run.json` and the 2091f42 sources; that `chat_mtp.run_block` really uses the replaced `command`/`payload`;
  the monitoring cadence actually used; every safety preflight and the server stop paths; that one launch cannot run
  the other config; the token-speed aggregation; the report's validity and comparability logic; the launcher diff
  against the historical `oneshot.sh`; and that README's list of differences is complete and accurate.
- For every file you cite, state its line count, and quote the exact code line for every file:line you cite.
- Line numbers must be the line numbers in the cited source file itself (as your file viewer shows them), never
  positions inside `_review_context/candidate.diff`. Every citation is checked against the copy; a line number
  past the end of a file or not matching the quoted line invalidates the whole review.
- End with exactly one verdict line: `APPROVED`, `APPROVED WITH NOTES`, or `CHANGES REQUESTED`, preceded by numbered
  findings each labelled BLOCKING, MINOR or NIT.
