# PICKPROMPT-2 Coder report

Coder: Codex CLI / gpt-6.1-sol / medium / Ponytail lite. Independent Reviewer: Claude Code CLI / claude-opus-5-5 / medium / Ponytail off. Research-only desk accuracy; no production integration.

## Outcome

**NO VARIANT QUALIFIES. TEST remains untouched.**

Material limitation (independent review F1/P2): all **25 invalid B1 outputs** (E4B 21, E2B 4) spent their 96-token budget repeating whitespace allowed by the grammar. B1 validity/fact/B3 results are therefore affected by this harness constraint; they do not isolate general model JSON ability. No grammar changes or re-scoring were made. The no-qualifier conclusion remains: all seven selectable variants have 3–7 R1 violations in valid parsed outputs. A future bounded-whitespace experiment would require a new task.

Accuracy uses temperature 0. All wall-ms and load seconds are **NOT MEASUREMENTS**: agent resident, no thermal/cold-load controls.

## Base and frozen input

Stage 1 was separately approved for commit and push. Stage 2 base HEAD = origin/main = live remote `8a840407b0fade86c90ed981080aa27224da3c64`; clean status and empty stash at the base check. Prior set SHA-256 `19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44`; no prior *_TEST.jsonl.

Run folder: `/termux-home/robot_runs/picker_prompt2_20261017T002036Z`. Copied set.jsonl, set_sha256.txt and freeze.json checked byte-for-byte before scoring and again after. Set: 120 policy items, 72 DEV/48 TEST, original balanced labels unchanged and verified using harness.label(). Synthetic cases are not real-world accuracy estimates.

```json
{
  "utc": "2026-10-17T00:20:37.294985+00:00",
  "set_sha256": "19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44",
  "source": "/termux-home/robot_runs/picker_prompt_20261016T160132Z",
  "copied_sha256": {
    "set.jsonl": "19773ea00d376d67e78641844241b6a8a310c0c8a7b1a048113f9c435a76ae44",
    "set_sha256.txt": "b90810f3a0f0832f0c834068ffce25a8cf6585d96db743c968e94341a76fbfd9",
    "freeze.json": "811599708c7ab8e5ca2470d2816eeba483c729dd2a7771d21b4bcbc811e6770a"
  },
  "code_sha256": {
    "harness2.py": "c6b5d3883ca617c0dff224bfd5912981daf7c06dfb8f322a29436a2dfd3c5bd7",
    "test_harness2.py": "e2652bd44730e10734086960da905f1035aa5694225de51197f57d55373ca7dc",
    "README2.md": "6660e2d5197d1d0a0e253fa689e95adda1ee117c1943e48504bc4a6dced878bd",
    "harness.py": "44e30a1bb12dfad3720f2577e1b7ecd6abdae95d04808d3aec16c32a3e2e11cb",
    "test_harness.py": "821a5f8df4290a9eed2af58c59bdcd66866f46300d475506ac91357e47e5f084",
    "README.md": "30752d54df8e8b68cc63be605cae3e1afc8fb86588777939faa38d9022d9ed1e"
  },
  "json_constraint_field": "grammar",
  "candidates": [
    "A0",
    "A2",
    "A4",
    "B1-E2B",
    "B2-E2B",
    "B1-E4B",
    "B2-E4B"
  ]
}
```

PICKPROMPT-1 freeze.json harness SHA identifies the original pre-recovery source e0488d2f..., not final committed source. Initial snapshot matches that hash; resumed snapshot matches current committed harness.py SHA 44e30a1b.... ASTs of label/state/options/suffix/StaticGemma are identical across initial, resumed and current source. This is prior PICKPROMPT-1 history, not a Stage2 change. Stage2 freeze2.json pins the final committed source.

## Model configs and fixed prompts

E4B setup_e4b_q4 and E2B setup_q4, matching argv tested against server_manager.start_setup: b1609 dotprod, ctx2048, threads/batch4, parallel1, swa-full, cache-ram0, localhost8080, no drafter, no additional taskset. One supervised server at a time; stopped between models, TEST phase if any, and at the end. No root, camera, motors, M2 runtime or downloads.

A0 = V0. A2 = V2. A4 = V2 plus E4B content-free calibration for each eligible list. A5 is reversed A4, with reversed-list baselines, diagnostic only. Letter completion body inherited unchanged: n_predict1, n_probs1000, post_sampling_probs false, temperature0, cache_prompt false.

B1 = V1 state plus exact instruction “First fill in the facts, then choose the action. Answer only with JSON.” B2 also includes the unchanged Robot rules block (V2 state). One Gemma user/model wrapper, no extra reasoning prose. Output uses the supported **grammar** field on /completion. Explicit GBNF enforces six fixed-order fields, boolean types and action last; no-person enum is EXPLORE|LOOK. n_predict96, temperature0, cache_prompt false, stream false.

The deployed source confirms /completion grammar sampling (tools/server/server-common.cpp (request parsing) and common/sampling.cpp (grammar sampler)), with json_schema support also present. Explicit grammar was chosen to enforce key order. Prompt-token timing field prompt_n equals full evaluated prompt here because cache_prompt=false (server-context.cpp disables prefix reuse in this branch); letter rows independently record assembled prompt length. HTTP timeout120 only in DeskHTTP; picker.TIMEOUT_S remains10.

Invalid/malformed JSON, duplicate/wrong/extra/missing keys, wrong types/order or ineligible actions count as wrong, never retried. Fact accuracy denominators include invalid items. Fact truths are derived from the frozen item: person visible, close/very-close person, greeting<300s, mission!=MS3, and **full R5** (no person, new-room age>=60s or MS2>=30s, last look never or>=60s). B3 derives action only from B1 Gemma booleans using label(), not original truths, with no extra inference.

## DEV results

| Variant | Accuracy | Macro recall | EXPLORE | APPROACH | GREET | LOOK | R1 violations | JSON valid | Median prompt tokens | Median generated tokens | Median prompt+generated | Median wall ms (NOT measurement) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.3889 | 0.3889 | 0.3333 | 0.5556 | 0.3333 | 0.3333 | 4 | N/A | 151.0 | 1.0 | 152.0 | 10493.1 |
| A2 | 0.5139 | 0.5139 | 0.2778 | 0.6111 | 0.9444 | 0.2222 | 5 | N/A | 359.5 | 1.0 | 360.5 | 23987.6 |
| A4 | 0.6111 | 0.6111 | 0.0556 | 0.6667 | 1.0000 | 0.7222 | 7 | N/A | 359.5 | 1.0 | 360.5 | 18401.4 |
| B1-E2B | 0.4028 | 0.4028 | 0.3333 | 0.2222 | 0.2778 | 0.7778 | 4 | 0.9444 | 161.0 | 83.0 | 246.0 | 18999.0 |
| B2-E2B | 0.5417 | 0.5417 | 0.2778 | 0.2778 | 0.8889 | 0.7222 | 5 | 1.0000 | 342.0 | 76.0 | 420.0 | 18688.1 |
| B1-E4B | 0.2083 | 0.2083 | 0.1667 | 0.0000 | 0.1667 | 0.5000 | 3 | 0.7083 | 161.0 | 83.0 | 248.0 | 34373.3 |
| B2-E4B | 0.3333 | 0.3333 | 0.3333 | 0.0000 | 0.7222 | 0.2778 | 4 | 1.0000 | 342.0 | 79.0 | 421.5 | 47432.5 |
| A5 | 0.5139 | 0.5139 | 0.1111 | 0.6111 | 1.0000 | 0.3333 | 7 | N/A | 359.5 | 1.0 | 360.5 | 17112.7 |
| B3-E2B | 0.4861 | 0.4861 | 0.3889 | 0.2778 | 0.8333 | 0.4444 | 5 | 0.9444 | 161.0 | 83.0 | 246.0 | 18999.0 |
| B3-E4B | 0.3750 | 0.3750 | 0.5000 | 0.0556 | 0.2222 | 0.7222 | 0 | 0.7083 | 161.0 | 83.0 | 248.0 | 34373.3 |

B3-E2B/B3-E4B and A5 are diagnostics and never selectable. B3 reuses B1 generation costs; it does not run another model request.

## Fact accuracy (invalid counts wrong)

| Variant | Visible | Close | Greeted <300s | Mission allows people | Full R5 condition |
|---|---:|---:|---:|---:|---:|
| B1-E2B | 0.7500 | 0.7361 | 0.8750 | 0.8056 | 0.5278 |
| B2-E2B | 0.9167 | 0.8750 | 0.6944 | 0.8194 | 0.6389 |
| B1-E4B | 0.5000 | 0.6111 | 0.5694 | 0.5000 | 0.5000 |
| B2-E4B | 1.0000 | 0.9722 | 0.7778 | 0.7222 | 0.8611 |

## A5, B3, selection and TEST

A5 changed **14/72** choices relative to A4.

Fixed selection: A0/A2/A4/B1-E2B/B2-E2B/B1-E4B/B2-E4B only; R1=0, every mode recall>=.6, JSON validity>=.95 for B; highest macro, then fewer median per-item(prompt+generated) tokens, then name. Summary is recomputed from JSONL and raw outputs/calibration, not hand-entered.

No qualifier, so no TEST file and no test_started event.

## B1/B2 exact examples

One person and one no-person DEV example for each JSON variant and each model. No prompt was changed after any model call.

### B1-E4B, S030, person=True, valid=True

Prompt:
```text
<start_of_turn>user
Mission: explore and map the rooms
Scene: potted plant (left, close), bed (center, very close), person (center, very close)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, LEFT, RIGHT
Rooms seen: none
M2 relations:
- none above threshold.
Person greeted: 301 s ago
New room found: 29 s ago
Last look-around: 180 s ago
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": true,
  "mission_allows_people": false,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}

```
Parsed output:
```json
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": true,
  "mission_allows_people": false,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}
```

### B1-E4B, S105, person=False, valid=True

Prompt:
```text
<start_of_turn>user
Mission: find a person and say hello
Scene: tv (center, close), potted plant (left, close), bottle (left, medium), clock (right, very close), bowl (center, far)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, RIGHT, HOLD
Rooms seen: bedroom, living room
M2 relations:
- potted plant [1] near tv [0].
Person greeted: never
New room found: 600 s ago
Last look-around: 61 s ago
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": true,
  "action": "LOOK"
}

```
Parsed output:
```json
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": true,
  "action": "LOOK"
}
```

### B2-E4B, S030, person=True, valid=True

Prompt:
```text
<start_of_turn>user
Mission: explore and map the rooms
Scene: potted plant (left, close), bed (center, very close), person (center, very close)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, LEFT, RIGHT
Rooms seen: none
M2 relations:
- none above threshold.
Person greeted: 301 s ago
New room found: 29 s ago
Last look-around: 180 s ago
Robot rules (first matching rule wins):
If the mission is to map quietly and not disturb people and a person is visible, keep exploring.
If a person is visible, you have not greeted them in the last 300 seconds, and they are close or very close, stop and greet them.
If a person is visible, you have not greeted them in the last 300 seconds, and they are far or medium distance, go toward them.
If a person is visible and you have greeted them in the last 300 seconds, keep exploring.
If no person is visible, no new room has been found for at least 60 seconds (30 seconds for the mission to find a person and say hello), and no look-around has happened in the last 60 seconds, look around in place.
Otherwise, keep exploring.
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}

```
Parsed output:
```json
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}
```

### B2-E4B, S105, person=False, valid=True

Prompt:
```text
<start_of_turn>user
Mission: find a person and say hello
Scene: tv (center, close), potted plant (left, close), bottle (left, medium), clock (right, very close), bowl (center, far)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, RIGHT, HOLD
Rooms seen: bedroom, living room
M2 relations:
- potted plant [1] near tv [0].
Person greeted: never
New room found: 600 s ago
Last look-around: 61 s ago
Robot rules (first matching rule wins):
If the mission is to map quietly and not disturb people and a person is visible, keep exploring.
If a person is visible, you have not greeted them in the last 300 seconds, and they are close or very close, stop and greet them.
If a person is visible, you have not greeted them in the last 300 seconds, and they are far or medium distance, go toward them.
If a person is visible and you have greeted them in the last 300 seconds, keep exploring.
If no person is visible, no new room has been found for at least 60 seconds (30 seconds for the mission to find a person and say hello), and no look-around has happened in the last 60 seconds, look around in place.
Otherwise, keep exploring.
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{"person_visible": false,"person_close": false,"greeted_in_last_5_min": true,"mission_allows_people": true,"nothing_new_for_a_while": true,"action": "LOOK"}
```
Parsed output:
```json
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": true,
  "mission_allows_people": true,
  "nothing_new_for_a_while": true,
  "action": "LOOK"
}
```

### B1-E2B, S030, person=True, valid=True

Prompt:
```text
<start_of_turn>user
Mission: explore and map the rooms
Scene: potted plant (left, close), bed (center, very close), person (center, very close)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, LEFT, RIGHT
Rooms seen: none
M2 relations:
- none above threshold.
Person greeted: 301 s ago
New room found: 29 s ago
Last look-around: 180 s ago
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}

```
Parsed output:
```json
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}
```

### B1-E2B, S105, person=False, valid=True

Prompt:
```text
<start_of_turn>user
Mission: find a person and say hello
Scene: tv (center, close), potted plant (left, close), bottle (left, medium), clock (right, very close), bowl (center, far)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, RIGHT, HOLD
Rooms seen: bedroom, living room
M2 relations:
- potted plant [1] near tv [0].
Person greeted: never
New room found: 600 s ago
Last look-around: 61 s ago
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": true,
  "action": "LOOK"
}

```
Parsed output:
```json
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": true,
  "action": "LOOK"
}
```

### B2-E2B, S030, person=True, valid=True

Prompt:
```text
<start_of_turn>user
Mission: explore and map the rooms
Scene: potted plant (left, close), bed (center, very close), person (center, very close)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, LEFT, RIGHT
Rooms seen: none
M2 relations:
- none above threshold.
Person greeted: 301 s ago
New room found: 29 s ago
Last look-around: 180 s ago
Robot rules (first matching rule wins):
If the mission is to map quietly and not disturb people and a person is visible, keep exploring.
If a person is visible, you have not greeted them in the last 300 seconds, and they are close or very close, stop and greet them.
If a person is visible, you have not greeted them in the last 300 seconds, and they are far or medium distance, go toward them.
If a person is visible and you have greeted them in the last 300 seconds, keep exploring.
If no person is visible, no new room has been found for at least 60 seconds (30 seconds for the mission to find a person and say hello), and no look-around has happened in the last 60 seconds, look around in place.
Otherwise, keep exploring.
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "GREET"
}

```
Parsed output:
```json
{
  "person_visible": true,
  "person_close": true,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "GREET"
}
```

### B2-E2B, S105, person=False, valid=True

Prompt:
```text
<start_of_turn>user
Mission: find a person and say hello
Scene: tv (center, close), potted plant (left, close), bottle (left, medium), clock (right, very close), bowl (center, far)
Ultrasonic distance ahead: no reading
Current action: look around in place (for 0 s)
Last moves: RIGHT, RIGHT, HOLD
Rooms seen: bedroom, living room
M2 relations:
- potted plant [1] near tv [0].
Person greeted: never
New room found: 600 s ago
Last look-around: 61 s ago
Robot rules (first matching rule wins):
If the mission is to map quietly and not disturb people and a person is visible, keep exploring.
If a person is visible, you have not greeted them in the last 300 seconds, and they are close or very close, stop and greet them.
If a person is visible, you have not greeted them in the last 300 seconds, and they are far or medium distance, go toward them.
If a person is visible and you have greeted them in the last 300 seconds, keep exploring.
If no person is visible, no new room has been found for at least 60 seconds (30 seconds for the mission to find a person and say hello), and no look-around has happened in the last 60 seconds, look around in place.
Otherwise, keep exploring.
First fill in the facts, then choose the action. Answer only with JSON.<end_of_turn>
<start_of_turn>model

```
Raw output:
```text
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}

```
Parsed output:
```json
{
  "person_visible": false,
  "person_close": false,
  "greeted_in_last_5_min": false,
  "mission_allows_people": true,
  "nothing_new_for_a_while": false,
  "action": "LOOK"
}
```

## Memory and server cleanup

| Model | Phase | Load s (informational) | MemAvailable before | MemAvailable after | Exit | Servers left |
|---|---|---:|---|---|---:|---|
| E4B | DEV | 8.79 | MemAvailable:    4522820 kB | MemAvailable:    2308768 kB | 0 | NONE |
| E2B | DEV_RECOVERY | 7.19 | MemAvailable:    2819564 kB | MemAvailable:    2035692 kB | 0 | NONE |

The interrupted original E2B DEV start has MemAvailable after-load **2,894,304 kB** in desk.stdout.txt. Its before-load value and load seconds are UNKNOWN because the interruption prevented the in-memory record from reaching servers.jsonl. The table above contains the two completed lifecycle records only; the missing original record is not evidence of a normal shutdown.

Full argv and logs remain in the run folder. MemAvailable is informational, not a comparative memory-cost measurement.

## Checks and scope

Pre-call tests: new harness14 OK (one optional saved-results skip), unchanged harness4 OK (one optional prior-results skip). No production or existing-test file changed; only harness2.py, test_harness2.py and README2.md are candidates. Original three PICKPROMPT-1 files byte-identical to base. Source/input hashes were frozen before first HTTP call and checked after scoring. All outputs, including invalid JSON, remain for independent review.

## Independent reviews

PICKPROMPT-1 safety flags/model/effort retained; stream-json + verbose captures actual tools for auditing. Allowed Bash: only exact stage unit tests plus wc -l <file>, sha256sum <file>. No fallback reviewer without owner decision. Full command/request, raw response and diagnostics retained separately in /termux-home/review/pickprompt2/.

### Round 1

{
  "valid": true,
  "errors": [],
  "model": "claude-opus-5-5",
  "shell_commands": [
    "wc -l picker.py",
    "wc -l server_manager.py",
    "sha256sum benchmark/strategic_selector/picker_prompt/harness2.py",
    "sha256sum benchmark/strategic_selector/picker_prompt/test_harness2.py",
    "sha256sum benchmark/strategic_selector/picker_prompt/README2.md",
    "sha256sum benchmark/strategic_selector/picker_prompt/harness.py",
    "sha256sum benchmark/strategic_selector/picker_prompt/test_harness.py",
    "sha256sum benchmark/strategic_selector/picker_prompt/README.md",
    "sha256sum picker_prompt2_results/set.jsonl",
    "sha256sum picker_prompt1_inputs/set.jsonl",
    "sha256sum picker_prompt2_results/set_sha256.txt",
    "sha256sum picker_prompt1_inputs/set_sha256.txt",
    "sha256sum picker_prompt2_results/freeze.json",
    "sha256sum picker_prompt1_inputs/freeze.json",
    "python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness2.py",
    "python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py",
    "wc -l candidate.diff",
    "wc -l picker_prompt2_results/B1-E4B_DEV.jsonl",
    "wc -l picker_prompt2_results/B2-E2B_DEV.jsonl",
    "sha256sum pickprompt2_evidence/resume_interrupted.py",
    "sha256sum picker_prompt2_results/A0_DEV.jsonl",
    "sha256sum picker_prompt2_results/A2_DEV.jsonl",
    "sha256sum picker_prompt2_results/A4_DEV.jsonl",
    "sha256sum picker_prompt2_results/A5_DEV.jsonl",
    "sha256sum picker_prompt2_results/B1-E4B_DEV.jsonl",
    "sha256sum picker_prompt2_results/B3-E4B_DEV.jsonl",
    "sha256sum picker_prompt2_results/B2-E4B_DEV.jsonl",
    "sha256sum picker_prompt2_results/B1-E2B_DEV.jsonl",
    "sha256sum picker_prompt2_results/B3-E2B_DEV.jsonl",
    "wc -l pickprompt2_evidence/recovery.stdout.txt",
    "sha256sum picker_prompt1_inputs/scored_harness_resume.py",
    "sha256sum picker_prompt1_inputs/scored_harness_initial.py",
    "wc -l docs/HANDOFF.md",
    "wc -l docs/WORKFLOW.md",
    "wc -l docs/REVIEWER.md",
    "wc -l TASK.txt",
    "wc -l benchmark/strategic_selector/picker_prompt/harness.py",
    "wc -l benchmark/strategic_selector/picker_prompt/harness2.py",
    "wc -l benchmark/strategic_selector/picker_prompt/test_harness2.py",
    "wc -l benchmark/strategic_selector/picker_prompt/README2.md",
    "wc -l benchmark/strategic_selector/picker_prompt/test_harness.py",
    "wc -l benchmark/strategic_selector/picker_prompt/README.md",
    "wc -l frozen_manifest.json",
    "wc -l picker_prompt2_results/events.jsonl",
    "wc -l picker_prompt2_results/freeze2.json",
    "wc -l picker_prompt2_results/servers.jsonl",
    "wc -l picker_prompt2_results/calibration.json",
    "wc -l picker_prompt2_results/summary.json",
    "wc -l pickprompt2_evidence/resume_interrupted.py",
    "wc -l pickprompt2_evidence/session_recovery.json",
    "wc -l pickprompt2_evidence/validate_recovery.py",
    "wc -l pickprompt2_evidence/recovery_validation.json",
    "wc -l pickprompt2_evidence/interrupted_attempt.json",
    "wc -l pickprompt2_evidence/owner_interruption_account.json",
    "wc -l pickprompt2_evidence/p1_source_history.txt",
    "wc -l picker_prompt2_results/A4_DEV.jsonl",
    "wc -l picker_prompt2_results/B3-E4B_DEV.jsonl",
    "wc -l interrupted_dev_attempt/events.jsonl",
    "wc -l interrupted_dev_attempt/servers.jsonl",
    "wc -l interrupted_dev_attempt/freeze2.json",
    "wc -l picker_prompt2_results/E2B_DEV.server.log",
    "wc -l pickprompt2_evidence/desk.stdout.txt",
    "wc -l pickprompt2_evidence/desk.stderr.txt",
    "wc -l pickprompt2_evidence/recovery.stderr.txt",
    "wc -l pickprompt2_evidence/CONTINUE.txt",
    "wc -l pickprompt2_evidence/pre_call_manifest.json",
    "wc -l pickprompt2_evidence/post_run_harness2_tests.txt",
    "wc -l pickprompt2_evidence/pre_run_harness2_tests.txt",
    "wc -l pickprompt2_evidence/run_path.txt"
  ],
  "read_files": [
    "TASK.txt",
    "benchmark/strategic_selector/picker_prompt/README.md",
    "benchmark/strategic_selector/picker_prompt/README2.md",
    "benchmark/strategic_selector/picker_prompt/harness.py",
    "benchmark/strategic_selector/picker_prompt/harness2.py",
    "benchmark/strategic_selector/picker_prompt/test_harness.py",
    "benchmark/strategic_selector/picker_prompt/test_harness2.py",
    "candidate.diff",
    "docs/HANDOFF.md",
    "docs/REVIEWER.md",
    "docs/WORKFLOW.md",
    "frozen_manifest.json",
    "interrupted_dev_attempt/events.jsonl",
    "interrupted_dev_attempt/freeze2.json",
    "interrupted_dev_attempt/servers.jsonl",
    "picker.py",
    "picker_prompt2_results/A4_DEV.jsonl",
    "picker_prompt2_results/B1-E4B_DEV.jsonl",
    "picker_prompt2_results/B3-E4B_DEV.jsonl",
    "picker_prompt2_results/E2B_DEV.server.log",
    "picker_prompt2_results/calibration.json",
    "picker_prompt2_results/events.jsonl",
    "picker_prompt2_results/freeze2.json",
    "picker_prompt2_results/servers.jsonl",
    "picker_prompt2_results/summary.json",
    "pickprompt2_evidence/CONTINUE.txt",
    "pickprompt2_evidence/desk.stderr.txt",
    "pickprompt2_evidence/desk.stdout.txt",
    "pickprompt2_evidence/interrupted_attempt.json",
    "pickprompt2_evidence/owner_interruption_account.json",
    "pickprompt2_evidence/p1_source_history.txt",
    "pickprompt2_evidence/post_run_harness2_tests.txt",
    "pickprompt2_evidence/pre_call_manifest.json",
    "pickprompt2_evidence/pre_run_harness2_tests.txt",
    "pickprompt2_evidence/recovery.stderr.txt",
    "pickprompt2_evidence/recovery.stdout.txt",
    "pickprompt2_evidence/recovery_validation.json",
    "pickprompt2_evidence/resume_interrupted.py",
    "pickprompt2_evidence/run_path.txt",
    "pickprompt2_evidence/session_recovery.json",
    "pickprompt2_evidence/validate_recovery.py",
    "server_manager.py"
  ],
  "line_count_entries": 42,
  "citations_verified": 5,
  "searched": [
    {
      "pattern": "\"valid\": false",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results",
      "output_mode": "count"
    },
    {
      "pattern": "\"valid\": false, \"raw_text\": \"[^\"]*(\\\\\"[^\"]*)*\"",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results/B1-E4B_DEV.jsonl",
      "output_mode": "content",
      "-o": true
    },
    {
      "pattern": "\"valid\": false, \"raw_text\": \".{0,60}",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results/B1-E2B_DEV.jsonl",
      "output_mode": "content",
      "-o": true
    },
    {
      "pattern": "\"generated_tokens\": 96",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results",
      "output_mode": "count"
    },
    {
      "pattern": "^(import|from)|macro_recall|Fraction|def metrics",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/interrupted_dev_attempt/attempt_source/harness2.py",
      "output_mode": "content"
    },
    {
      "pattern": "\"id\": \"S\\d+\"|\"choice\": \"[A-Z]+\"",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/interrupted_dev_attempt/A0_DEV.jsonl",
      "output_mode": "content",
      "-o": true,
      "head_limit": 40
    },
    {
      "pattern": "\"id\": \"S\\d+\"|\"choice\": \"[A-Z]+\"",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results/A0_DEV.jsonl",
      "output_mode": "content",
      "-o": true,
      "head_limit": 40
    },
    {
      "pattern": "^\\{\"id\": \"S\\d+\"",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results/B2-E2B_DEV.jsonl",
      "output_mode": "content",
      "-o": true,
      "head_limit": 8
    },
    {
      "pattern": "interrupt|build|error|grammar|n_ctx|model_draft|draft",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results",
      "glob": "*.server.log",
      "output_mode": "content",
      "-i": true,
      "head_limit": 60
    },
    {
      "pattern": "harness_sha256",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt1_inputs/freeze.json",
      "output_mode": "content"
    },
    {
      "pattern": "\"(selected|outcome|best_macro_recall)\"",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt1_inputs/summary.json",
      "output_mode": "content"
    },
    {
      "pattern": "\"split\": \"TEST\"",
      "path": "/data/data/com.termux/files/home/review/pickprompt2/copy_r1/picker_prompt2_results/set.jsonl",
      "output_mode": "count"
    }
  ],
  "verdict": "VERDICT: APPROVED"
}

Reviewer response verbatim:

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

## Exact reviewer invocation

Working directory: `/termux-home/review/pickprompt2/copy_r1`. Environment: `PONYTAIL_DEFAULT_MODE=off`, `PYTHONDONTWRITEBYTECODE=1`. Request passed as one final argument; exact request retained in request_r1.txt.

```text
claude -p --model claude-opus-5-5 --effort medium --safe-mode --restricted --disable-slash-commands --tools Read,Grep,Glob,Bash --allowedTools Read Grep Glob 'Bash(python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness2.py)' 'Bash(python3 -B -m unittest discover -s benchmark/strategic_selector/picker_prompt -p test_harness.py)' 'Bash(wc -l *)' 'Bash(sha256sum *)' --permission-mode dontAsk --permission-prompts none --strict-mcp-config --mcp-config '{"mcpServers":{}}' --no-session-persistence --output-format stream-json --verbose <contents-of-request_r1.txt>
```

## Coder supplemental checks after review

These checks are Coder evidence outside the frozen review copy; the approved candidate did not change. Compared all 72 B3 rows to B1 for each model: byte-value equality for all fields except the prescribed variant/action changes. Confirmed all 25 B1 invalid outputs hit 96 generated tokens and repeated whitespace. Invalid R1 JSON is counted wrong in recall but is not an APPROACH/GREET R1 violation; valid violations alone disqualify every selectable variant. Full evidence: supplemental_review_checks.json.

Final live HEAD/origin/main/ls-remote remain 8a840407b0fade86c90ed981080aa27224da3c64. Git tracked diff empty; exactly the three manifest paths are untracked. Source/input hashes and no-server check passed. Before/after review identities match. Reviewer new suite 14 tests (no skips, actual saved-results audit enabled); original suite 4 tests (one saved-prior-results skip).

## Candidate manifest

```json
[
  {
    "path": "benchmark/strategic_selector/picker_prompt/README2.md",
    "bytes": 5499,
    "sha256": "6660e2d5197d1d0a0e253fa689e95adda1ee117c1943e48504bc4a6dced878bd",
    "lines": 92
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/harness2.py",
    "bytes": 18498,
    "sha256": "c6b5d3883ca617c0dff224bfd5912981daf7c06dfb8f322a29436a2dfd3c5bd7",
    "lines": 367
  },
  {
    "path": "benchmark/strategic_selector/picker_prompt/test_harness2.py",
    "bytes": 18167,
    "sha256": "e2652bd44730e10734086960da905f1035aa5694225de51197f57d55373ca7dc",
    "lines": 317
  }
]
```

## Interrupted DEV attempt and metric correction

{
  "attempt": "/termux-home/robot_runs/picker_prompt2_20261017T000147Z",
  "reason": "non-prompt metric correction before any selection, JSON or TEST: exact Fraction macro prevents float drift bypassing token/name ties",
  "exit": 143,
  "rows": {
    "A0_DEV.jsonl": 72,
    "A2_DEV.jsonl": 19
  },
  "json_or_test_calls": 0,
  "prompt_and_all_other_function_ASTs_unchanged": true,
  "models_stopped": true
}

The partial attempt is retained in full. It was stopped before any JSON, selection or TEST. Only metrics() plus an import changed in the scoring harness: exact rational macro computation removes floating drift that could bypass the fixed token/name tie-break. All other AST nodes, including prompt text, grammar, labels, options and scoring bodies, are identical across attempts. No prompt was tuned and no invalid JSON or TEST item retried. Final DEV is complete in a fresh folder with the disclosed later session recovery; the partial rows remain separate, not silently discarded.

## Session interruption and DEV continuation

{
  "utc": "2026-10-17T10:13:10.561304+00:00",
  "reason": "Parent session interrupted; server log ends Received second interrupt, terminating immediately; original process and server absent. No completed response exists for the in-flight sixth B2-E2B item.",
  "saved_b2_e2b": 5,
  "prefix_sha256": "4f5a28814f484468407d10be4635fc83f34e9acbe79abf8210660bf9718b01e3",
  "untouched": {
    "A0_DEV.jsonl": "24cc02dd57b1fa23ee3be24c7eea3d8afbe239cc786647ae76d7ece3225c5d69",
    "A2_DEV.jsonl": "d37a50ef31d819833190bcb12bfa85f1c7acad693cb5165b31d80839e06ccd0d",
    "A4_DEV.jsonl": "a96661c06cff825ccd87a86e666f1da360bf01a649149e5de0bb9dc62fe775db",
    "A5_DEV.jsonl": "f62b07625d023257d1de815e6b8a0fb6db6f2aded8a7443954629975766997f9",
    "B1-E4B_DEV.jsonl": "fc294f57115aa843fffa99c39f706d04649004a15ec04d6b75f56a78a48cacff",
    "B3-E4B_DEV.jsonl": "c228238a5f213c9bfe993f7592b3afc18ba71e3357dbae9ddbed8fcce61e2c41",
    "B2-E4B_DEV.jsonl": "6fda03695e0726d5bcbcc14b35c3a93d34ecf683d84c62a42b6916dc4c5292f7",
    "B1-E2B_DEV.jsonl": "3716477a7e3c23210c74b164a855aff8c3ccdcd487983cc6501ba460e1f12f39",
    "B3-E2B_DEV.jsonl": "a9b0af2898a3cf80ceb4af2125711ce9ac3432ab29f478ec6fc5d469fc5c9496"
  },
  "source_sha256": "7d50f92667521b7618cccb6ee8c6fa8fdc8c1e3ef94cc4f1c3351d0ef70379a1",
  "policy": "Continue remaining DEV ids only; no saved JSON (valid or invalid) retried; same frozen functions/prompts; original server log retained; in-flight unfinished request status unknown."
}

The session terminated the original runner and server after nine complete variants plus five B2-E2B rows. Its final in-flight sixth DEV response was never recorded; generation status is UNKNOWN. A recovery script outside the repository imported the unchanged frozen harness functions and appended only the remaining DEV ids. Every previously saved row, valid or invalid, was retained byte-for-byte. No TEST or selection existed before recovery. The original forced E2B stop has no completed cleanup record; no server remained before recovery, which has its own log/memory/cleanup record. No uninterrupted-run or exactly-once-generation claim is made for the unrecorded in-flight request.

Owner clarification (2026-10-17): the Coder reached its five-hour usage limit, then the phone shut down from low battery. This cause is owner-reported; power logs were not independently checked.

## UNKNOWN / limitations

- Synthetic balanced policy items do not establish real-world navigation or conversation quality.
- Timings are not measurements, and sustained heat, energy and robot co-residency are not tested.
- JSON schema adherence and these six policy facts are the scope; open-ended reasoning quality is not assessed.
- No production integration, deployment, canonical-doc edits, staging, commit or push.

## Human gate

Stop for explicit “yes commit”. Stage only the three manifest paths, message `benchmark: picker desk test 2, E4B and JSON think-then-answer (PICKPROMPT-2)`. Push needs separate “yes push”.

## Owner-authorized local commit

Owner said “yes commit”. Created `3fab51a3030e8fd026cc85ab5563b534ac74c4ba` with the exact requested message and only the three frozen manifest paths. Verified all committed blob SHA-256 hashes against the reviewed manifest. Working tree clean; main ahead of origin/main by one commit. Not pushed; separate “yes push” required. Existing Git identity: root <root@localhost.localdomain>. No model calls or tests were repeated for this Git-only operation.

## Owner-authorized push

Owner separately said “yes push”. Push succeeded:

```text
To https://github.com/abdelhamidzoghlamii-ui/Pixel-robot.git
   8a84040..3fab51a  main -> main
```

Live remote verification:

```text
3fab51a3030e8fd026cc85ab5563b534ac74c4ba	refs/heads/main
## main...origin/main
```

HEAD, origin/main and live remote main all equal `3fab51a3030e8fd026cc85ab5563b534ac74c4ba`. Working tree clean. Both task stages are committed and pushed. TEST remains unscored; no qualifying variant or production picker change.
