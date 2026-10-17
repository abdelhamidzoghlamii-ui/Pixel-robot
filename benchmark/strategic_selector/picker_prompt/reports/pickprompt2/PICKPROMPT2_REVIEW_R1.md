I approve this stage. Nothing blocks it, but one P2 finding (F1) must be disclosed in the Coder report. The result, **NO VARIANT QUALIFIES**, holds up. Every one of the seven selectable variants has at least three R1 violations from parsed, eligible outputs. Selection, the TEST gate and the summary all recompute correctly from the raw per-item rows.

## Checks performed

**Commands run (only the allowed ones):**
- Both unit suites:
  - `test_harness2.py`: 14 tests OK, none skipped, so the saved-results audit actually ran against `picker_prompt2_results/`.
  - `test_harness.py`: 4 tests OK, 1 skipped (its saved-results test, because `picker_prompt_results/` is not in this copy).
- `sha256sum` on 29 files: the 6 source files, the 3 copied inputs in both places (6 files), 10 result row files, 2 PICKPROMPT-1 harness snapshots, `resume_interrupted.py`, and the 4 B2/B3 row files.
- `wc -l` on every file I read.

**Frozen set and labels**
- `set.jsonl`, `set_sha256.txt` and `freeze.json` hash the same in `picker_prompt2_results/` and `picker_prompt1_inputs/`. The set hash is `19773ea0…`.
- 48 TEST rows. Each DEV confusion row in `summary.json` sums to 18, so DEV is 72 and balanced.
- `summarize()` asserts that each row's label and rule equal `label()` of the original item. Model output is never used as ground truth.

**Source pinning**
- All six source hashes match `freeze2.json` and the first event in `events.jsonl`.
- `frozen_manifest.json` matches `pre_call_manifest.json`, and its hashes and line counts (92, 367, 317) match the files.
- `candidate.diff` (794 lines) adds only the three new files, and I compared it against them by reading both.
- `harness.py` hashes to `44e30a1b…`, which equals `picker_prompt1_inputs/scored_harness_resume.py`.
- `scored_harness_initial.py` hashes to `e0488d2f…`, which equals `harness_sha256` in the PICKPROMPT-1 `freeze.json`.
- Production timeout is unchanged:

`picker.py:35`
```python
TIMEOUT_S = 10                  # per HTTP request
```
- The desk transport uses its own 120 s timeout:

`benchmark/strategic_selector/picker_prompt/harness2.py:90`
```python
        with urllib.request.urlopen(request, timeout=120) as response:
```

**Letter scoring (A variants)**
- A variants call the inherited `Gemma.decide` unchanged (n_predict 1, n_probs 1000, temperature 0, cache_prompt false). The scoring body is not modified.
- A0 uses the V0 prompt; A2, A4 and A5 use V2.
- Calibration ran on the E4B server. All calibration states are `N/A` content-free prompts with the rules block. There are separate baselines for each eligible list, and A5 has its own reversed baselines (`LOOK,EXPLORE` and `LOOK,GREET,APPROACH,EXPLORE`).
- I recomputed A4 for item S030 by hand from its raw probabilities and the baselines: GREET 0.6294 and LOOK 0.3598, matching the saved row.
- A5 is diagnostic only. It changed 14 choices compared with A4.

**JSON variants (B1/B2/B3)**
- The JSON request matches the spec: `grammar` field, n_predict 96, temperature 0, cache_prompt false. B1 uses V1 plus the exact instruction; B2 uses V2.
- `parse_json` enforces the exact six keys in order, boolean types, no duplicates, and an eligible action. Invalid outputs count as wrong and are never retried. An HTTP error propagates and stops the run.
- The R5 fact truth is computed from the item, and it is equivalent to the full R5 condition:

`benchmark/strategic_selector/picker_prompt/harness2.py:34`
```python
                nothing_new_for_a_while=h.label(s)[1] == 'R5')
```
- B3 maps only Gemma's five booleans into `label()`. I checked S030 by hand: Gemma reported a visible person and `mission_allows_people` false, so `label()` gives EXPLORE, matching the saved row. B3 is never selectable.

**Selection**
- Selection uses exactly the seven fixed candidates and the required order of criteria:

`benchmark/strategic_selector/picker_prompt/harness2.py:173`
```python
    eligible = [v for v in CANDIDATES if scores[v]['r1_violations'] == 0 and min(scores[v]['recall'].values()) >= .6
```
- R1 violations per candidate: A0 4, A2 5, A4 7, B1-E2B 4, B2-E2B 5, B1-E4B 3, B2-E4B 4.
- I recomputed two macro recalls by hand from the confusion matrices: B2-E2B is 39/72 and B1-E4B is 15/72.
- No `*_TEST.jsonl` exists and there is no `test_started` event. `selection_frozen` comes before `finished`.

**Servers**
- The server command (argv) is identical to `setup_e4b_q4` and `setup_q4`: ctx 2048 and no MTP drafter. The unit test checks this, and I also compared it against `server_manager.py`.
- `servers.jsonl` records the E4B DEV and E2B DEV_RECOVERY servers, both with exit 0 and no server left running. MemAvailable before and after each start is recorded as information only. The summary marks all timings as NOT measurements.

## Findings

**F1 — P2, must be disclosed; does not change the outcome.** The grammar's whitespace rule has no length limit:

`benchmark/strategic_selector/picker_prompt/harness2.py:58`
```python
            'ws ::= [ \\t\\n\\r]*\naction ::= ' + actions + '\n')
```
- All 25 invalid outputs are this one failure:
  - 21 in B1-E4B: after the fifth boolean, the model emits `\n  ` repeatedly instead of the comma.
  - 4 in B1-E2B: the same, but after `"person_visible":`.
  - All 25 rows have `generated_tokens` of 96, so they ran into n_predict. B2 has no invalid outputs.
- So the "JSON validity" numbers for B1 (0.708 for E4B, 0.944 for E2B) mostly reflect this harness grammar, not models that cannot produce JSON.
- B1-E2B fails the 0.95 validity threshold only because of these four rows.
- The B1 fact accuracy and B3 numbers have the same distortion, because invalid rows count as wrong.
- The outcome still stands: R1 violations count only parsed, eligible APPROACH/GREET choices, so these failures cannot hide or remove a violation.
- No re-scoring is allowed after model calls, so this should be stated in the report. Any future JSON stage should bound the whitespace.

**F2 — Low.** Rows save the prompt but not the request body or grammar text. The grammar can be rebuilt exactly from the pinned source, and the runaway outputs show the grammar was enforced. I could not check whether the deployed build supports the `grammar` field by reading the server source.

**F3 — Low.** `summarize()` does not check that B3 rows are copies of B1 rows. I spot-checked S030, which matches, and both files' hashes were recorded in `session_recovery.json` before recovery.

**F4 — Info.** An invalid output on an R1 item is not counted as an R1 violation. That is reasonable, but README2 does not say so.

**F5 — Info, interrupted DEV attempt.**
- That attempt stopped with exit 143 after 72 A0 rows and 19 A2 rows. It has no B files and no TEST.
- `test_saved_results` confirmed that its retained `harness2.py` differs from the final one only in `metrics()` and the imports. Its source hashes match its own `freeze2.json`. Its server exited -9 with no server left running.
- The Coder could see those A0/A2 DEV results before the fix. DEV is not held out and no prompt changed.
- The first 20 A0 choices are identical between the two attempts, so the results reproduce at temperature 0.
- The exact-fraction tie-break was never used, because nothing qualified.

**F6 — Limitation, session recovery.**
- The E2B server log ends with `Received second interrupt, terminating immediately.` after five completed B2 requests, with a sixth in flight. Whether that sixth request produced output is UNKNOWN, so this is not an uninterrupted run.
- The original E2B DEV server has no cleanup record.
- The recovery script's hash matches its record (`7d50f926…`). It continues with `dev[5:]` through the frozen `json_rows`, using a separate DEV_RECOVERY log, and runs the original selection and TEST gate.
- All nine untouched result files hash as recorded. B2-E2B's first rows are, in order, S030, S077, S011, S004, S105, which matches the DEV order.
- Rows 1–5 and rows 6–72 came from two different server processes. At temperature 0 with cache_prompt false this should be equivalent, but it is a difference.
- The cause of the interruption (the Coder hitting its five-hour usage limit, then the phone shutting down on low battery) is the owner's account. No power-log evidence supports it.

## What I could not establish independently

- The hash of the five-row B2-E2B prefix. The allowed commands cannot hash part of a file.
- The PICKPROMPT-1 AST-equality claim in `p1_source_history.txt`. I only confirmed the two snapshot hashes.
- That nothing was left running when the session was interrupted.
- That the live tree is byte-identical to the base. There is no `.git` in the copy, so I rely on the Coder's pre-call and post-run scope checks and on the matching source hashes.
- That no camera, motors, M2 or root were used.
- The model identities, beyond the paths in the server command.

## Files read

| File | Lines |
|---|---|
| `docs/HANDOFF.md` | 68 |
| `docs/WORKFLOW.md` | 303 |
| `docs/REVIEWER.md` | 15 |
| `TASK.txt` | 44 |
| `benchmark/strategic_selector/picker_prompt/harness.py` | 374 |
| `benchmark/strategic_selector/picker_prompt/harness2.py` | 367 |
| `benchmark/strategic_selector/picker_prompt/test_harness2.py` | 317 |
| `benchmark/strategic_selector/picker_prompt/README2.md` | 92 |
| `benchmark/strategic_selector/picker_prompt/test_harness.py` | 153 |
| `benchmark/strategic_selector/picker_prompt/README.md` | 103 |
| `picker.py` | 257 |
| `server_manager.py` | 141 |
| `candidate.diff` | 794 |
| `frozen_manifest.json` | 20 |
| `picker_prompt2_results/events.jsonl` | 4 |
| `picker_prompt2_results/freeze2.json` | 28 |
| `picker_prompt2_results/servers.jsonl` | 2 |
| `picker_prompt2_results/calibration.json` | 58 |
| `picker_prompt2_results/summary.json` | 509 |
| `picker_prompt2_results/B1-E4B_DEV.jsonl` | 72 |
| `picker_prompt2_results/A4_DEV.jsonl` | 72 |
| `picker_prompt2_results/B3-E4B_DEV.jsonl` | 72 |
| `picker_prompt2_results/E2B_DEV.server.log` | 557 |
| `pickprompt2_evidence/resume_interrupted.py` | 51 |
| `pickprompt2_evidence/session_recovery.json` | 19 |
| `pickprompt2_evidence/validate_recovery.py` | 25 |
| `pickprompt2_evidence/recovery_validation.json` | 11 |
| `pickprompt2_evidence/interrupted_attempt.json` | 12 |
| `pickprompt2_evidence/owner_interruption_account.json` | 5 |
| `pickprompt2_evidence/p1_source_history.txt` | 1 |
| `pickprompt2_evidence/desk.stdout.txt` | 12 |
| `pickprompt2_evidence/desk.stderr.txt` | 0 |
| `pickprompt2_evidence/recovery.stdout.txt` | 578 |
| `pickprompt2_evidence/recovery.stderr.txt` | 0 |
| `pickprompt2_evidence/CONTINUE.txt` | 16 |
| `pickprompt2_evidence/pre_call_manifest.json` | 20 |
| `pickprompt2_evidence/pre_run_harness2_tests.txt` | 5 |
| `pickprompt2_evidence/post_run_harness2_tests.txt` | 5 |
| `pickprompt2_evidence/run_path.txt` | 1 |
| `interrupted_dev_attempt/events.jsonl` | 1 |
| `interrupted_dev_attempt/servers.jsonl` | 1 |
| `interrupted_dev_attempt/freeze2.json` | 28 |

Partial reads: `picker_prompt2_results/B1-E4B_DEV.jsonl` lines 1–3, `picker_prompt2_results/A4_DEV.jsonl` lines 1–2, `picker_prompt2_results/B3-E4B_DEV.jsonl` line 1, `picker_prompt2_results/E2B_DEV.server.log` lines 520–557, and `pickprompt2_evidence/recovery.stdout.txt` lines 1–75. Everything else was read in full.

**Searched only:** `interrupted_dev_attempt/attempt_source/harness2.py`, `interrupted_dev_attempt/A0_DEV.jsonl`, `picker_prompt2_results/A0_DEV.jsonl`, `picker_prompt2_results/B1-E2B_DEV.jsonl`, `picker_prompt2_results/B2-E2B_DEV.jsonl`, `picker_prompt2_results/B3-E2B_DEV.jsonl`, `picker_prompt2_results/E2B_DEV_RECOVERY.server.log`, `picker_prompt2_results/E4B_DEV.server.log`, `picker_prompt2_results/set.jsonl`, `picker_prompt1_inputs/freeze.json`, `picker_prompt1_inputs/summary.json`.

**Hashed only:** `picker_prompt2_results/set_sha256.txt`, `picker_prompt2_results/freeze.json`, `picker_prompt2_results/A2_DEV.jsonl`, `picker_prompt2_results/A5_DEV.jsonl`, `picker_prompt2_results/B2-E4B_DEV.jsonl`, `picker_prompt1_inputs/set.jsonl`, `picker_prompt1_inputs/set_sha256.txt`, `picker_prompt1_inputs/scored_harness_initial.py`, `picker_prompt1_inputs/scored_harness_resume.py`.

I did not read `verified_line_counts.txt`.

VERDICT: APPROVED
