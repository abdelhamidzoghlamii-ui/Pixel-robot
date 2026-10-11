# CAMPAIGN_P1_FIX2 Coder report

Owner follow-up 2026-10-05: F6 resolved. Keep the evidence-backed passed preflight label.
The charger attempt was the second, blocked invocation under the same stem; no other stem.
Explicit "yes commit" received. Reviewed code unchanged; RUN_INDEX.md resolution and manifest
provenance updated. Local commit completed; push is not authorized and was not run.

EXECUTION PROFILE: Coder | Codex CLI | gpt-6.1-sol | effort medium | Ponytail lite.
Fallback AGY/gemini-3.1-pro-high was not used.
REVIEWER: Claude Code CLI headless | claude-opus-5-5 | effort N/A | Ponytail off,
PONYTAIL_DEFAULT_MODE=off, permission-mode plan, Edit/Write/NotebookEdit/Bash/WebFetch/WebSearch disallowed.
No silent reviewer fallback. No commit/staging/push, models or motor action.

## HEAD and status

Initial HEAD 10be01e296815e2173855afebd49cb0e52ee66fd; matched required base 10be01e296815e2173855afebd49cb0e52ee66fd.
Initial git status --short was empty. Final status:

```text
 M benchmark/campaign/RUN.md
 M benchmark/campaign/RUN_INDEX.md
 M benchmark/campaign/phase1.py
 M benchmark/campaign/runtime.py
 M benchmark/campaign/self_check.py
?? benchmark/campaign/checks/p1_fix2/
?? benchmark/campaign/diagnostics.py
?? benchmark/campaign/frozen_manifest_campaign_p1_fix2.json
?? benchmark/campaign/runs/archive_sha256_p1_fix2.json
?? benchmark/campaign/runs/owner_preflight_p1_fix.json
?? benchmark/campaign/runs/owner_preflight_p1_fix.stderr
?? benchmark/campaign/runs/owner_preflight_p1_fix.stdout
?? benchmark/campaign/runs/owner_preflight_p1_fix2.json
?? benchmark/campaign/runs/owner_preflight_p1_fix2.stderr
?? benchmark/campaign/runs/owner_preflight_p1_fix2.stdout
?? benchmark/campaign/runs/owner_session_p1a_fix.json
?? benchmark/campaign/runs/owner_session_p1a_fix.stderr
?? benchmark/campaign/runs/owner_session_p1a_fix.stdout
?? benchmark/campaign/runs/owner_session_p1a_fix_block_01_L0.json
?? benchmark/campaign/runs/owner_session_p1a_fix_block_02_L1.json
?? benchmark/campaign/runs/owner_session_p1a_fix_block_03_L2.json
?? benchmark/campaign/self_check_fix2.py
```

Only benchmark/campaign/ candidate changes. Ignored archive server logs are listed explicitly below.

## Findings verified against owner bytes

1. L1 ZERO M2 calls: 36/36 insufficient-box slots. L2 ZERO: 9/9.
   All their live 640 detections were fewer than two. Ceiling/table orientation is an inference,
   not established by these JSON files. New fallback guarantees actual inference for low-box slots.
2. L1/L2 warm starts: gates 482.128/489.819s and warm_start=true. L0 skin 21.538 ->27.891C,
   +6.353C over180s. These runs could not reach the idle+1.5C gate before timeout; that does
   not prove the target physically unreachable with unlimited cooling. Fixed pauses replace gates.
3. All blocks policy0 capped99.4%, min1401MHz vs cpuinfo_max1803MHz; policy4/6 zero.
   Verified all four committed SPEED1 owner_session_speed1_fix blocks report zero caps for all
   three policies (camera/Gemma absent in that runner). No cause assigned and no cap writes.
4. llama-server PSS L0 first3796.284MiB, peak3805.124MiB (~3.89GB), last3403.189MiB;
   L1 first3215.384MiB, last2864.691MiB (~2.93GB); L2 2861.688–2864.704MiB.
   Selector medians1770.5/1935.2/1817.3ms. Observation only, no causal claim.
   Exact source-derived fields: benchmark/campaign/checks/p1_fix2/owner_findings.json.

## Evidence discrepancy requiring owner resolution

The task requests owner_preflight_p1_fix labelled NOT VALID — REFUSED at start with the
reason quoted from stderr. The supplied JSON instead records PREFLIGHT ONLY — NO TIMING
and setup_complete=true, stdout confirms passed/cleanup, stderr only the ONNX Runtime
unsupported-Android warning. There is no refusal reason to quote. All supplied bytes are
archived unchanged, with the explicit contradiction in RUN_INDEX. The owner was asked
whether to retain the evidence-backed passed label or provide the refused attempt's path.
This requested fixed-label requirement remains unresolved; no refusal was fabricated.

## Design by step

Step1: shared M2 scheduler selects live input at >=2 boxes, fixed fallback otherwise;
all L1/L2/L3/L4 paths covered. L0 has no M2/context. Per-call scene/live_boxes/input_boxes,
wall/ORT inference times; per-block live/fallback counts. Fallback selector context marked
(fallback scene). M2 no-inference and >32 boxes fail closed. Existing PIL loader reused.
Fallback SHA-256s: speed_photo.jpg 4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8;
speed_input.json 35694e263f3894597e711bd2c2045bc55b3790f0bcd32d75baae11e60c20e32f.

Step2: common setup;300s idle split180OFF/60OFF/60ON-no-inference; one180s L0 warm-up,
WARM-UP — NOT A RESULT;600s pause before each measured180s block, defaultL0,L1,L2,L3,L4,L0.
No skin gate code remains. Pause camera/YOLO/M2/selector stopped, Gemma retained, same
power/thermal/cap/memory readers. Subsets validate/retain L0 warm-up. First measured skin
sets T_ref; later absolute delta>1.5C NOT COMPARABLE — START TEMP, other failures retained.
Battery<80 refuses setup; <25 before idle/warm-up/pause/block clean stop and unrun list.
86min scheduled default; print85–90min estimate plus setup/build/transition/cleanup overhead.

Review-round corrections: diagnostics script is stripped before the root wrapper appends its
semicolon. Offline sh -n checks exact wrapped text (including campaign taskset prefix), and
also reproduces the original syntax failure. Preflight now decodes/checks fallback input and
runs diagnostic snapshots inside each layout's safe root scope; mocked failure refusals added.
Discovery bounded to depth5, read_elapsed_s retained. Printed idle plan corrected.
camera_state_source/docs explain requested state and existing transition checks, no independent
snapshot state readback. These updates were frozen and reviewed again.

Step3: start/end diagnostics record battery level/temp, all discovered thermal-zone type/temp,
cooling type/cur_state, policy0 min/max, best-effort readable boost min/max/power-hint nodes,
camera state, Android status, raw node output. Required battery/cap reads fail closed;
optional unreadables explicit. Pause root readers independently owned/pinned/cleaned.
Idle OFF/ON phases have separate cap summaries. Record only; no cap adjustment.

Step4:15 owner files copied unchanged and SHA-verified. Existing committed evidence untouched.
Index contains the prescribed session/block labels and exact metrics; first-preflight label
cannot truthfully match requested refusal because supplied files show pass (see discrepancy).

Step5: all required self-check/direct-script tests pass, including fallback/counts/context,
T_ref boundary, battery refuse/stop/unrun, pause schedule/read wiring/cleanup and diagnostics.
Default dry run: NOT VALID — DRY RUN (all hardware mocked); warm-up,6 pauses,6 blocks.
Checks/results below. Initial cycle test invocation used wrong filename and exited2;
preserved, then corrected run_cycle_safety_test.py passes. No live timing or root validation.
The cycle harness prints [MOTOR] for mocks only; motors were not actuated.

## Owner commands / RUN.md

benchmark/campaign/RUN.md has:
1. Fresh preflight owner_preflight_p1_fix3 (fix2 stem already exists), shell noclobber.
2. One full session owner_session_p1_fix2, approximately85–90min; warm-up+all6blocks+pauses.
Setup: upright mount, cluttered table with3–4objects, lights on,nothing moving,battery>=80%,
charger unplugged,screen on,stay in native Termux; exit all agents/processes first.
Exact preflight/session/dry files to send are listed there. Pauses/idle live in session JSON.
No coder real timed block/session/root preflight; proot cannot reach root,agents resident
invalidate timing. Live root/battery/optional diagnostic-node discovery/Android sensors and
services/camera-only caps/restarts/fallback M2/YOLO overlap/Gemma HTTP/600s cooling/native
cleanup remain mock-only paths for owner validation.

## Archive inventory (verified unchanged)

| Stem | Evidence label |
|---|---|
| owner_preflight_p1_fix | Source: PREFLIGHT ONLY — NO TIMING. Passed. Requested NOT VALID — REFUSED at start (reason quoted from stderr); no measurement is unsupported; owner decision pending. |
| owner_preflight_p1_fix2 | PREFLIGHT ONLY — NO TIMING. Passed. |
| owner_session_p1a_fix | SESSION COMPLETE; L0 VALID; L1 and L2 NOT VALID. |
| block_01_L0 | VALID. Full cycle/no M2; metrics verified above and in RUN_INDEX. |
| block_02_L1 | NOT VALID — WARM START (gate 482 s) and NO M2 LOAD (36/36 slots < 2 boxes). |
| block_03_L2 | NOT VALID — WARM START (gate 490 s) and NO M2 LOAD (9/9 slots < 2 boxes). |

| Path | Bytes | SHA-256 |
|---|---:|---|
| benchmark/campaign/runs/owner_preflight_p1_fix.json | 7989 | 2395edf28eecef1245db6b38c23b2ae752844d68dabdf1329d99c8581ac7d295 |
| benchmark/campaign/runs/owner_preflight_p1_fix.stderr | 239 | 1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca |
| benchmark/campaign/runs/owner_preflight_p1_fix.stdout | 437 | cc6516794e0c95df50874e870c78259efe993c3eaa12f6f48216d3ab927997ac |
| benchmark/campaign/runs/owner_preflight_p1_fix_llama-server.log | 2270 | a22944d60e15409ef0be0dd98d95b833d8304ca1616de8ae4c16307f13a890d3 |
| benchmark/campaign/runs/owner_preflight_p1_fix2.json | 8001 | 47e0d1f7ebfea8816b14629a72946a95d5d19caf7a985546f955ec8d1a1b0f23 |
| benchmark/campaign/runs/owner_preflight_p1_fix2.stderr | 239 | 1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca |
| benchmark/campaign/runs/owner_preflight_p1_fix2.stdout | 438 | a2ef6bc1dc6d2326a5ae22603873e49665b0d7b7534d1a144140b05f1f591615 |
| benchmark/campaign/runs/owner_preflight_p1_fix2_llama-server.log | 2270 | b46132877ffeb4da0a9f3925a35a3275aa00ce4ee3a93522d494b7547d259045 |
| benchmark/campaign/runs/owner_session_p1a_fix.json | 936483 | 68bcdc292e8084640ed2e7e44a661b22154412418be5b8f25153ad70703a74b6 |
| benchmark/campaign/runs/owner_session_p1a_fix.stderr | 239 | 1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca |
| benchmark/campaign/runs/owner_session_p1a_fix.stdout | 4708 | c6960c3394452f802868f38bccac15250d655e9ae575854a1e6db71d451c9ebb |
| benchmark/campaign/runs/owner_session_p1a_fix_block_01_L0.json | 259914 | 4a9b07131ed55855b31c77c3dc2c1e4e1586cbdd290e0113d6763e52b3877eed |
| benchmark/campaign/runs/owner_session_p1a_fix_block_02_L1.json | 270203 | b252ba3d0a7f084a10795dd47d59fb74a40f03dc906965bb70a94dc9d1dc4146 |
| benchmark/campaign/runs/owner_session_p1a_fix_block_03_L2.json | 262783 | 4c4e7822f5fd4555fbf188a045919ebccceceeb1893f2a45b372dd88718dbe83 |
| benchmark/campaign/runs/owner_session_p1a_fix_llama-server.log | 24041 | 37e626d3bb53936236628ec7c95175461abafc7abf0cfa201537bf3f46f6d27a |

## Check results

```json
{
  "self_check": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "self_check_fix2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check_fix2.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "speed1_self_check": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/relate_anything/speed1/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "desk2_self_check": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/relate_anything/desk2/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "power_map_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/power_map/test_power_map.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "coresidency_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/coresidency/test_coresidency.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "detector_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "test_detector_robotcam.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "mission_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "test_run_mission.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "cycle_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "test_run_cycle_safety.py"
    ],
    "exit_code": 2,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "dry_run": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/phase1.py",
      "--dry-run",
      "--output",
      "/termux-home/robot/benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2.json"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct-script only; agents resident; no live validation"
  },
  "cycle_units_r2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "run_cycle_safety_test.py"
    ],
    "exit_code": 0,
    "scope": "offline direct-script; corrected test filename; no live validation"
  },
  "self_check_r2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks; final candidate"
  },
  "self_check_fix2_r2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check_fix2.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks; final candidate"
  },
  "self_check_fix2_r3": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check_fix2.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks; final diagnostic discovery pattern"
  },
  "self_check_review_r2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks and shell syntax only; review findings F1/F2/F3/F4/F5"
  },
  "self_check_fix2_review_r2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check_fix2.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks and shell syntax only; review findings F1/F2/F3/F4/F5"
  },
  "self_check_review_r2_final": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks; reviewed fixes candidate"
  },
  "dry_run_review_r2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/phase1.py",
      "--dry-run",
      "--output",
      "/data/data/com.termux/files/home/robot/benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2.json"
    ],
    "exit_code": 0,
    "scope": "offline mocks; reviewed fixes candidate"
  }
}
```

git diff --check passed. Frozen manifest: benchmark/campaign/frozen_manifest_campaign_p1_fix2.json; 77 hashed files. Reverified live/frozen/archive hashes after review. Manifest itself is not self-hashed.

## Independent review — verbatim result texts

### review_campaign_p1_fix2_r1.json

# CAMPAIGN_P1_FIX2: independent review, round 1

**Verdict: CHANGES REQUESTED**

The design meets most of the requirements, but the new diagnostics command contains a shell syntax error, so it would fail the first time it runs on the phone. That stops the owner's 86-minute session at the first idle phase. Preflight never runs that code, so preflight would pass and the session would still fail.

I only read files in the frozen copy. I did not edit anything, run commands, run motors, or start another reviewer.

## Findings

### F1 — BLOCKING: diagnostics script fails with a shell syntax error (`benchmark/campaign/diagnostics.py:37-53`)
- `snapshot()` builds a multi-line script that ends with a newline (`…for f in $paths; do read_node "$f"; done\n`).
- `coresidency.root` (`benchmark/coresidency/coresidency.py:125`) wraps every command as `{ <cmd>; } </dev/null >FILE 2>&1; …`. Here that becomes `…done\n; }`.
- A `;` at the start of a line is a syntax error in mksh, bash, dash and ash. The command exits nonzero, and `snapshot` raises `RuntimeError('diagnostics root read failed')`.
- **Failure:** `rest('IDLE', …)` calls `diag.snapshot` right after setup, so the session ends `NOT VALID — SESSION INCOMPLETE` before the warm-up. Every block would hit the same failure in `live_block` (line 383).
- Every other `root()` caller in the repo sends single-line scripts with no trailing newline (for example `duty_cycle.py:108`), so nothing earlier would have exposed this.
- The mocks replace `cr.root` with a lambda, so the shell parse is never tested.
- **Fix:** remove the trailing newline (or end the script with `true`), and add a check that the wrapped text parses, e.g. run `sh -n` offline on the exact `{ prefix+script; }` string.

### F2 — MAJOR: preflight skips the new live paths (`phase1.py:592-641`)
- Preflight calls `diag.battery` and `rt.fallback_hashes()`, but never `diag.snapshot` or `rt.fallback_input()` (the PIL load and image/box check in `sb.load_speed_input`).
- RUN.md says preflight and session "share setup", but this new setup work first runs inside the timed session.
- A fallback-load failure would first appear at L1, about 40 minutes in. F1 shows exactly the kind of bug preflight should catch.
- **Fix:** in preflight, call `diag.snapshot(cr, False)` inside a root mask scope, and call `rt.fallback_input()` once.

### F3 — MINOR: diagnostics `find` is unbounded (`diagnostics.py:45-46`)
- It searches all of `/sys/devices/platform`, `/sys/kernel` (which can include debug/tracing trees) and more, about 40 times per session. Twice of those are inside each block's setup, after the camera has started.
- The `cr.root` timeout is 60 s. Runtime and heat from this are unknown and add to the 85–90 minute estimate.
- **Fix:** limit the search depth or the directories searched, and record how long each diagnostics read takes.

### F4 — MINOR: plan line is misleading (`phase1.py:659`)
- It prints "300 s idle (last 60 s camera OFF, then 60 s camera ON…)".
- What actually runs is 180 s OFF, then 60 s OFF, then 60 s ON, all within the 300 s. RUN.md describes this correctly; only the printed line is wrong.

### F5 — MINOR: "camera state" is not read from the device (`diagnostics.py:65`)
- `camera_on` only records what the runner asked the camera to do. Nothing reads the RobotCam process or frame state back.
- Accept this with a note in the docs, or add a `pidof com.pixelrobot.robotcam` read.

### F6 — OPEN: `owner_preflight_p1_fix` label conflicts with the owner's files (needs the owner)
- I confirmed the conflict in the archived files:
  - The JSON has `label: PREFLIGHT ONLY — NO TIMING` and `setup_complete: true`.
  - stdout ends "PREFLIGHT ONLY — NO TIMING".
  - stderr contains only the ONNX platform warning.
- So no refusal reason can be quoted. The Coder kept the bytes unchanged and flagged the conflict explicitly instead of inventing a refusal. That is the right handling.
- The task's required label is still not met, so the owner must decide this before any commit.

## Requirements that are met
- **M2 fallback:**
  - Fallback is used whenever there are fewer than 2 live boxes, in every M2 layout; L0 is unchanged.
  - Both SHA-256s are hard-coded and checked at setup and at load.
  - `speed_input.json` really points at `speed_photo.jpg`, and its box format (`class_name`/`box_xyxy`) matches the live detector.
  - Each call records `scene`, `live_boxes` and `input_boxes`; each block records live and fallback call counts.
  - The `(fallback scene)` label is added to both M2 context and the selector context.
  - A slot that runs no inference now fails instead of being counted.
- **Schedule:**
  - 300 s idle, then one L0 warm-up labelled `WARM-UP — NOT A RESULT`.
  - A fixed 600 s pause before every measured block, with the camera stopped, YOLO/M2/selector released, Gemma kept loaded, and power, caps, skin and memory sampled.
  - The skin gate is removed.
  - `T_ref` comes from the first measured block, and a delta over 1.5 °C gets the `NOT COMPARABLE` label without overriding other invalid labels.
  - Subset runs keep the warm-up and pauses.
- **Battery:** below 80% refuses setup. Below 25% stops cleanly before the idle phases, warm-up, each pause and each block, keeps `unrun_blocks`, and still stops the server.
- **Archive:** the inventory bytes and SHA-256s match the supplied list. The fix2 preflight, session, L0, L1 and L2 labels match the task wording.
- **Checks:** all reported checks exited 0. The one exit 2 was a wrong test filename, which was kept and then re-run correctly.
- **Dry run:** labelled NOT VALID. Scheduled time is 86 min.
- **Manifest:** I could not find or verify a frozen manifest file in this copy.

## Live paths still only tested with mocks
- Root diagnostics shell execution (F1), optional boost/power-hint node discovery, and how long `find` takes.
- Real fallback image load and M2 inference on that image, running alongside YOLO.
- Battery capacity read through root; the 80% refusal and 25% stop on real hardware.
- The 600 s pause machinery: prepare, root mask scope, workers, power-sampler coverage, caps/memory/LMK during pauses, and cleanup.
- Camera-only cap response in the camera-ON idle phase, and the repeated camera stop/start cycles.
- Live thermal-zone, cooling-device and policy0 reads; logcat; PSS.
- Root and su affinity readback; selector HTTP, context and timeouts with the fallback label; Gemma health over 86 minutes.
- Real start-temperature comparability across six blocks, and native cleanup.
- Whether the whole session fits in 85–90 minutes.

No live validation is claimed: agents were resident and proot cannot reach root. No Git mutation is authorized. After F1 and F2 are fixed, the next review round can propose a commit, which still needs a separate "yes commit" and "yes push".

**Final verdict: CHANGES REQUESTED**

### review_campaign_p1_fix2_r2.json

**Verdict: APPROVED (round 2).** One condition applies at the commit gate: the owner must resolve F6 before the commit proposal is accepted. This is a review only. I made no edits and ran no commands. Everything below comes from reading files in the frozen copy, including `benchmark/campaign/frozen_manifest_campaign_p1_fix2.json`.

## Round-1 fixes checked against the source

- **F1 (diagnostics script parses):** fixed. `cr.root` passes the command as an argument to `su -c` with no extra layer of quoting, so the single quotes in `printf` are safe. With `script.strip()`, the script ends in `done; }`, which the shell accepts. The `sh -n` test in `self_check_fix2.py` covers both the new form and the round-1 form that failed.
- **F2 (preflight checks):** fixed. `preflight()` checks battery ≥80, checks both fallback hashes, decodes the photo through `load_speed_input` (482×640, 2–32 boxes), and takes a diagnostics snapshot inside the masked root wrapper for every layout. L0 is always included, so `--blocks` subsets still check the warm-up layout. The mocked refusal tests now include fallback, diagnostics and battery80.
- **F3:** fixed. Discovery uses `find -maxdepth 5` and each snapshot records `read_elapsed_s`.
- **F4:** fixed. The printed plan is 180 s OFF + 60 s OFF + 60 s ON, and the estimate is 300+180+780×6 = 86 min scheduled.
- **F5:** fixed. `camera_state_source` and RUN.md both say the snapshot records the requested camera state. `pm.camera_start` waits for a real frame before returning (`power_map.py:168`), and `rt.stop_camera` checks that capture ended.

## Requirement coverage
1. **Fallback (met).**
   - Live input is used when there are at least 2 boxes; otherwise the fixed photo and committed boxes are used, after re-verifying both hashes (`runtime.py:31`).
   - The box format matches the existing `speed_block` caller (`box_xyxy`, `class_name`).
   - `inference_ms <= 0` now fails closed, and `LiveOps.relate` refuses fewer than 2 boxes.
   - Each call records `scene`, `live_boxes` and `input_boxes`; each block records `live_calls` and `fallback_calls`.
   - Fallback context is labelled `(fallback scene)`, including selector appends. More than 32 boxes still fails closed, and L0 is unchanged.
2. **Schedule (met).**
   - The order is idle phases, then the L0 warm-up, then a fixed 600 s pause before every measured block.
   - The warm-up is relabelled `WARM-UP — NOT A RESULT`, with its original label kept in `result_validity`.
   - The skin gate is removed. `compare_start` sets T_ref from the first measured block and marks a delta over 1.5 °C as `NOT COMPARABLE — START TEMP`; boundaries are tested.
   - Setup refuses below 80%. A clean `BatteryStop` below 25% is checked before the warm-up, before each pause, inside each rest phase and before each block, and leaves the rest in `unrun_blocks`.
3. **Diagnostics (met, read-only).** Battery level and temperature, every thermal zone, cooling `cur_state`, policy0 min/max, optional boost and power-hint nodes, and Android status are recorded at start and end of each phase and block. Each idle sub-phase keeps its own `caps_by_policy`. Nothing is written to caps.
4. **Archive (met, except the F6 label).** All 15 files in `runs/archive_sha256_p1_fix2.json` match the manifest hashes. The session and block labels in RUN_INDEX match the requested wording.
5. **Mocks, RUN.md and manifest (met).** Mock coverage is present, RUN.md gives one fresh preflight (`_p1_fix3`) and one full-session command with the exact files to send, and the manifest is frozen.

## Findings

**F6 — owner-evidence contradiction (unresolved, owner action needed).** I confirmed the Coder's claim from the bytes:
- `owner_preflight_p1_fix.json` has `"label": "PREFLIGHT ONLY — NO TIMING"` and `"setup_complete": true`.
- The stdout ends with `PREFLIGHT ONLY — NO TIMING Evidence: …`.
- The stderr (239 bytes, same hash as the other runs) contains only the ONNX platform warning.

There is no refusal reason to quote. Writing the requested `NOT VALID — REFUSED at start (reason …)` label would invent evidence. The Coder's RUN_INDEX row is the only honest option: it labels what the source shows, quotes the requested label and states the discrepancy. This blocks the commit gate, not the code: the owner must confirm either (a) the refused charger attempt used a different stem that was never supplied, or (b) the source label should stand.

**Minor observations (none block approval):**
- **N1 — brief mislabelling window.** `live_block` writes the warm-up file (and each block file) with its own label before `main` rewrites it with `WARM-UP — NOT A RESULT` or the comparability label. If the process is killed between those two writes, `_warmup_L0.json` could be left showing `VALID`. The session JSON and the filename still identify it, so the risk is low.
- **N2 — estimate may be optimistic.** Each block, pause and idle phase takes a start and end snapshot (about 38 snapshots in total). Each one runs `find` over `/sys/kernel` (including tracing/debugfs events) and `/sys/devices/platform` at depth 5. Together with 7 model builds and camera stop checks, real time could exceed 90 minutes. `read_elapsed_s` will show this. A slow read hits the 60 s `cr.root` timeout and stops the session safely; it does not produce bad data.
- **N3 — no end snapshot on failed blocks.** `diagnostics_end` is not recorded when a block fails, so failed blocks have only start diagnostics.
- **N4 — small gaps between idle phases.** The 300 s idle runs as three separate rest phases, each re-preparing its monitors, so there are short gaps between them. This is acceptable.

## Live paths still only mock-tested (owner validation needed)
- Root `su` and the transient-shell taskset/readback wrapper running the new multi-line diagnostics script; real `find` time and optional-node discovery.
- Real battery capacity reads, the 80% refusal and a real 25% clean stop.
- Thermal-zone, cooling, policy0 and boost/power-hint reads, including zones that error on read.
- Camera-OFF and camera-ON idle phases and whether the camera alone causes policy0 capping; camera restarts across 16 phases and blocks.
- Fallback M2 inference with the real photo and boxes on LITTLE, BIG and MID, its overlap with YOLO-640, and L3 skip behaviour.
- Selector HTTP with the `(fallback scene)` context appended, and timeouts.
- Six real 600 s pauses with Gemma resident: power coverage, memory/PSS, thermal stops, charger state.
- T_ref comparability on real skin readings; the full ~86–95 minute session; logcat/LMK; native cleanup and screen restore.
- No live timed block, session or root preflight was run by the Coder. These checks are not valid with agents resident, and proot cannot reach root.

## Next steps
No Git change is authorised here. After the owner resolves F6, the commit proposal (including force-adding the Git-ignored `_llama-server.log` files) needs an explicit "yes commit", and the push a separate "yes push".

**Final verdict: APPROVED**, on condition that the owner resolves F6 before the commit gate.


## Commit preparation

Reviewer APPROVED. Proposed commit (still requires owner decision): "benchmark: campaign phase-1 M2 fallback, fixed pauses, LITTLE-cap diagnostics, session A archive".
Exact candidate list below includes ignored server logs; force-add only after human gate.

- benchmark/campaign/RUN.md
- benchmark/campaign/RUN_INDEX.md
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_block_01_L0.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_block_02_L1.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_block_03_L2.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_block_04_L3.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_block_05_L4.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_block_06_L0.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_block_01_L0.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_block_02_L1.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_block_03_L2.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_block_04_L3.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_block_05_L4.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_block_06_L0.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_r2_warmup_L0.json
- benchmark/campaign/checks/p1_fix2/coder_dry_p1_fix2_warmup_L0.json
- benchmark/campaign/checks/p1_fix2/coresidency_units.stderr
- benchmark/campaign/checks/p1_fix2/coresidency_units.stdout
- benchmark/campaign/checks/p1_fix2/cycle_units.stderr
- benchmark/campaign/checks/p1_fix2/cycle_units.stdout
- benchmark/campaign/checks/p1_fix2/cycle_units_r2.stderr
- benchmark/campaign/checks/p1_fix2/cycle_units_r2.stdout
- benchmark/campaign/checks/p1_fix2/desk2_self_check.stderr
- benchmark/campaign/checks/p1_fix2/desk2_self_check.stdout
- benchmark/campaign/checks/p1_fix2/detector_units.stderr
- benchmark/campaign/checks/p1_fix2/detector_units.stdout
- benchmark/campaign/checks/p1_fix2/dry_run.stderr
- benchmark/campaign/checks/p1_fix2/dry_run.stdout
- benchmark/campaign/checks/p1_fix2/dry_run_review_r2.stderr
- benchmark/campaign/checks/p1_fix2/dry_run_review_r2.stdout
- benchmark/campaign/checks/p1_fix2/mission_units.stderr
- benchmark/campaign/checks/p1_fix2/mission_units.stdout
- benchmark/campaign/checks/p1_fix2/owner_findings.json
- benchmark/campaign/checks/p1_fix2/power_map_units.stderr
- benchmark/campaign/checks/p1_fix2/power_map_units.stdout
- benchmark/campaign/checks/p1_fix2/results.json
- benchmark/campaign/checks/p1_fix2/self_check.stderr
- benchmark/campaign/checks/p1_fix2/self_check.stdout
- benchmark/campaign/checks/p1_fix2/self_check_fix2.stderr
- benchmark/campaign/checks/p1_fix2/self_check_fix2.stdout
- benchmark/campaign/checks/p1_fix2/self_check_fix2_r2.stderr
- benchmark/campaign/checks/p1_fix2/self_check_fix2_r2.stdout
- benchmark/campaign/checks/p1_fix2/self_check_fix2_r3.stderr
- benchmark/campaign/checks/p1_fix2/self_check_fix2_r3.stdout
- benchmark/campaign/checks/p1_fix2/self_check_fix2_review_r2.stderr
- benchmark/campaign/checks/p1_fix2/self_check_fix2_review_r2.stdout
- benchmark/campaign/checks/p1_fix2/self_check_r2.stderr
- benchmark/campaign/checks/p1_fix2/self_check_r2.stdout
- benchmark/campaign/checks/p1_fix2/self_check_review_r2.stderr
- benchmark/campaign/checks/p1_fix2/self_check_review_r2.stdout
- benchmark/campaign/checks/p1_fix2/self_check_review_r2_final.stderr
- benchmark/campaign/checks/p1_fix2/self_check_review_r2_final.stdout
- benchmark/campaign/checks/p1_fix2/speed1_self_check.stderr
- benchmark/campaign/checks/p1_fix2/speed1_self_check.stdout
- benchmark/campaign/diagnostics.py
- benchmark/campaign/phase1.py
- benchmark/campaign/runs/archive_sha256_p1_fix2.json
- benchmark/campaign/runs/owner_preflight_p1_fix.json
- benchmark/campaign/runs/owner_preflight_p1_fix.stderr
- benchmark/campaign/runs/owner_preflight_p1_fix.stdout
- benchmark/campaign/runs/owner_preflight_p1_fix2.json
- benchmark/campaign/runs/owner_preflight_p1_fix2.stderr
- benchmark/campaign/runs/owner_preflight_p1_fix2.stdout
- benchmark/campaign/runs/owner_preflight_p1_fix2_llama-server.log
- benchmark/campaign/runs/owner_preflight_p1_fix_llama-server.log
- benchmark/campaign/runs/owner_session_p1a_fix.json
- benchmark/campaign/runs/owner_session_p1a_fix.stderr
- benchmark/campaign/runs/owner_session_p1a_fix.stdout
- benchmark/campaign/runs/owner_session_p1a_fix_block_01_L0.json
- benchmark/campaign/runs/owner_session_p1a_fix_block_02_L1.json
- benchmark/campaign/runs/owner_session_p1a_fix_block_03_L2.json
- benchmark/campaign/runs/owner_session_p1a_fix_llama-server.log
- benchmark/campaign/runtime.py
- benchmark/campaign/self_check.py
- benchmark/campaign/self_check_fix2.py
- benchmark/campaign/frozen_manifest_campaign_p1_fix2.json

STOP: no commit without explicit "yes commit"; no push without separate "yes push". Evidence-label clarification also remains pending.

## Authorized commit receipt

Commit: 914ce55e792a76f7ba24726bb739e7c4b531dbec

Message: benchmark: campaign phase-1 M2 fallback, fixed pauses, LITTLE-cap diagnostics, session A archive

Exact committed files: same 78-file proposal listed above. Only RUN_INDEX.md documentation and manifest provenance changed after review to record the owner's F6 decision. Original reviewed manifest preserved in campaign/frozen_manifest_campaign_p1_fix2_r2.json in Downloads. Owner archive hashes reverified unchanged; staged diff whitespace check passed.

Final git status --short: empty. No push performed.

## Authorized push receipt

Owner authorized push. git push origin main succeeded: 10be01e..914ce55 main -> main. Commit 914ce55e792a76f7ba24726bb739e7c4b531dbec published to origin/main. Final git status --short: empty.
