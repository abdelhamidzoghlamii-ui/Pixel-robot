# PICKPROMPT-2: E4B letters and facts-then-action JSON

Desk accuracy only; no root, camera, motors, M2 runtime, downloads or production
changes. Imports the unchanged PICKPROMPT-1 harness for states, options and
`label()`. The prior balanced synthetic set is copied byte-for-byte from
`~/robot_runs/picker_prompt_20261016T160132Z`; its SHA-256 is
`19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44`.
There are 72 DEV items and 48 TEST items. These are policy cases, not an estimate
of real-world robot accuracy. All timings are NOT measurements (agent resident).

The runner owns one supervised server at a time and refuses an existing server.
Its argv matches `setup_e4b_q4` for E4B and `setup_q4` for E2B: deployed b1609
dotprod, ctx 2048, threads/batch 4, parallel 1, swa-full, cache-ram 0, localhost,
no drafter. No taskset is added to these setup commands. MemAvailable and load
seconds are informational. Each model is stopped between phases and at exit.
HTTP timeout is 120 seconds in this harness; production remains at 10 seconds.

## Fixed variants

- A0: E4B, PICKPROMPT-1 V0 state/options and letter scoring.
- A2: E4B, V2 state/options.
- A4: E4B, V2 with contextual calibration. Baselines are computed on E4B,
  only from content-free prompts, separately for person/no-person option lists.
- A5: diagnostic A4 with reversed options and reversed-list calibration;
  report the number of choices changed from A4. Not selectable.
- B1-E2B and B1-E4B: V1 state (three time facts), then exactly
  `First fill in the facts, then choose the action. Answer only with JSON.`
  wrapped as one Gemma user turn followed by the model turn prefix.
- B2-E2B and B2-E4B: same, with the V2 Robot rules block in the state.
- B3-E2B and B3-E4B: diagnostic reuse of B1's outputs. `label()` chooses the
  action from Gemma's five booleans, without original item truths or another
  model call. Not selectable.

Letter scoring calls the inherited `Gemma.decide`: n_predict 1, n_probs 1000,
post_sampling_probs false, temperature 0, cache_prompt false. The static
tokenizer changes request count, not assembled prompt tokens or scoring body.

JSON uses the `/completion` **grammar** field, supported by the deployed build
(server-task/common sampling grammar handling). Both `json_schema` and grammar
support exist in its source; explicit GBNF guarantees the requested fixed key
order. Body: temperature 0, n_predict 96, cache_prompt false, stream false.
Exactly these keys are generated, with `action` last:

```json
{"person_visible": true, "person_close": false, "greeted_in_last_5_min": false,
 "mission_allows_people": true, "nothing_new_for_a_while": false, "action": "APPROACH"}
```

The five facts are booleans; action is EXPLORE/APPROACH/GREET/LOOK. On no-person
items its grammar allows only EXPLORE/LOOK. Field truths come from the item:
visible person; any person close/very close; greeting age strictly below 300 s;
mission different from MS3; and the **full R5 condition** (no person, new-room
age at least 60 s or 30 s for MS2, last look never or at least 60 s ago).
`nothing_new_for_a_while` is therefore false on person-present items. Invalid
JSON, wrong types/order, duplicate/extra/missing keys or an ineligible action
count as wrong, including in the per-field accuracy denominators; never retried.
An HTTP/server failure stops the run, preserves partial evidence and stops the
server; it is not converted into a successful or retried inference.

## Selection and artifacts

Selectable: A0, A2, A4, B1-E2B, B2-E2B, B1-E4B, B2-E4B. Require R1 violations
zero, every mode recall >=0.6, and JSON validity >=0.95 for B. Highest DEV macro
recall wins; ties use fewer median **per-item prompt+generated tokens**, then
variant name. Invalid outputs remain in recall denominators and get an INVALID
confusion column. One winner gets all 48 TEST items once; none qualifying yields
NO VARIANT QUALIFIES and no TEST file. No prompt tuning or repeat invocation
after a model call: an output folder must not already exist.

`freeze2.json` pins the original three copied artifacts, all three new source
files and all three unchanged PICKPROMPT-1 files before any HTTP call.
`events.jsonl` records first-call, selection and TEST ordering. Per-item JSONL
records raw/parsed output, validity, choice/label/rule, token counts and wall ms;
letters also record raw/calibrated eligible probabilities. JSON rows include
the exact prompt. `summary.json` recomputes DEV metrics, fact accuracy, A5 order
changes and selection from the saved JSONL, checking the labels, JSON parsing,
choices and calibration. `calibration.json` retains content-free prompts and
baselines. `servers.jsonl` and server logs record memory, exact argv, load and
cleanup. Source/input hashes are checked again after scoring.

Run from the repository (proot is supported; no root calls):

```bash
python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness2.py
python3 -B benchmark/strategic_selector/picker_prompt/harness2.py /termux-home/robot_runs/picker_prompt2_<UTC>
```

Use the installed native Termux Python if Debian's Python lacks existing project
dependencies. The optional saved-results test reads `picker_prompt2_results/` in
an isolated review copy; it skips when the external artifacts are absent.
