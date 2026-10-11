# CAMPAIGN_P23 Coder report

Status: APPROVED — ready for owner commit decision. The failed AGY round3 attempt received no request and returned CHANGES REQUESTED without substantive review. Owner explicitly authorized round3 redo with Claude Code Opus5.5 medium; its verdict determines readiness. Owner M3 decision: fixed2160-second wall-clock grid, transitions inside slots; inference-based duty fraction. No owner live run has been executed.

Execution: Coder, Codex CLI interactive, gpt-6.1-sol, high, Ponytail lite. Independent Reviewer rounds1–2: Claude Code CLI headless, claude-opus-5-5, high, Ponytail off. Owner first changed round3 to AGY Gemini3.1ProHigh; its missing-input attempt is retained. Owner then explicitly authorized Claude Code claude-opus-5-5 --effort medium for round3 redo, Ponytail off; no silent substitution. Candidate remains uncommitted; no push, motors, native live preflight, rehearsal or timed session. Agents resident invalidate timings; proot cannot reach root. E4: no DOC DIFF; docs/ untouched. Smaller implementation chosen: reuse Phase-1 scheduler/readers and existing detector/M2 builders; no separate hardware reader, model or dependency. Raw telemetry has numbered atomic 60-second checkpoints with 2-second overlap, retaining completed sensor and inference work rows after a hard kill.

## HEAD and status

At entry HEAD was 074d5b92ce32330d6fb3c04d9d0ee7db0556ef3d (FIX4D), status --short empty. Native path and /termux-home/robot refer to the same tree. Read session instructions/role/workflow, campaign audit/run/source/tests/evidence, decisions #121/#125-129, and #128 CONFIRM SizePolicy setting; checks/p23/entry_inventory.json records full-byte inventory of campaign and power_map inputs.

Current HEAD: 074d5b92ce32330d6fb3c04d9d0ee7db0556ef3d

```text
 M benchmark/campaign/RUN.md
 M benchmark/campaign/RUN_INDEX.md
 M benchmark/campaign/phase1.py
 M benchmark/campaign/self_check_fix4.py
?? benchmark/campaign/checks/p23/
?? benchmark/campaign/frozen_manifest_campaign_p23.json
?? benchmark/campaign/phase23.py
?? benchmark/campaign/runs/archive_sha256_p1_fix4d.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d.stderr
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d.stdout
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_01_L0.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_02_L1.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_03_L2.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_04_L3.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_05_L4.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_06_L0.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_idle_1.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_idle_2.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_idle_3.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_01_L0.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_02_L1.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_03_L2.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_04_L3.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_05_L4.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_06_L0.json
?? benchmark/campaign/runs/owner_rehearsal_p1_fix4d_warmup_L0.json
?? benchmark/campaign/runs/owner_session_p1_fix4d.json
?? benchmark/campaign/runs/owner_session_p1_fix4d.stderr
?? benchmark/campaign/runs/owner_session_p1_fix4d.stdout
?? benchmark/campaign/runs/owner_session_p1_fix4d_block_01_L0.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_block_02_L1.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_block_03_L2.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_block_04_L3.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_block_05_L4.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_block_06_L0.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_idle_1.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_idle_2.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_idle_3.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_pause_01_L0.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_pause_02_L1.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_pause_03_L2.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_pause_04_L3.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_pause_05_L4.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_pause_06_L0.json
?? benchmark/campaign/runs/owner_session_p1_fix4d_warmup_L0.json
?? benchmark/campaign/self_check_p23.py
```

Two archived llama-server.log files are ignored by the existing .gitignore, but explicitly included in the frozen manifest and proposed commit list. They will be force-added with the approved candidate when the owner authorizes the commit. No staging has been performed.

## Owner Phase-1 archive — numbers verified against bytes

All 40 source files were copied unchanged to benchmark/campaign/runs/. Every source/archive SHA-256 and byte length was asserted identical before freeze and independent review. Complete file inventory: runs/archive_sha256_p1_fix4d.json. No committed evidence was changed. Every requested number matched, so no owner discrepancy decision is needed.

| Stem | Files | Bytes | Fixed label |
|---|---:|---:|---|
| owner_rehearsal_p1_fix4d | 20 | 2,413,779 | REHEARSAL PASS — NOT VALID as results |
| owner_session_p1_fix4d | 20 | 15,722,614 | SESSION COMPLETE |

Rehearsal: all phases, 6 pauses and 6 blocks ran; live M2 in L1–L4; 0 read re-reads, 0 thermal retries. Session: last block END at 5767.3 s (about 96 min), 0 errors, 0 re-reads, Android status 0 throughout raw thermal samples, 0 LMK kills. Warm-up NOT A RESULT.

| 180 s block | Fixed runner label | YOLO-320 done/due | M2 inference median/P95 ms | Selector median ms | Time-weighted W | Start minus T_ref C |
|---|---|---|---|---:|---:|---:|
| First L0 | VALID | 180/180 | — | 1733 | 4.10 | 0.00 |
| L1 | VALID | 180/180 | 4919/6451 | 2819 | 4.99 | +0.69 |
| L2 | NOT COMPARABLE — START TEMP | 180/180 | 1176/1323 | 2129 | 4.72 | +1.92 |
| L3 | VALID | 109/180; 71 skipped by design | 1313/1760 | 2956 | 5.10 | +1.37 |
| L4 | NOT COMPARABLE — START TEMP | 180/180 | 871/2001 | 3686 | 5.41 | +2.02 |
| Final L0 | NOT COMPARABLE — START TEMP | 180/180 | — | 1775 | 4.04 | +1.86 |

T_ref = 29.5405 C. Exact offending deltas +1.920603, +2.021300, +1.856501 C; D2 checked pause-end skin while block start skin was read after setup. M2 uses inference_stats (not end-to-end M2 wall stats). Power uses current_now, time-weighted nominal 0.37 s receipt samples: exact means 4.097385, 4.991161, 4.715040, 5.095653, 5.411304, 4.038437 W. L4 policy6 capped 43.8874%; camera-ON idle policy0 capped 98.4538%, camera-OFF 0%. Owner selected L2 for Phases 2–3; archived noncomparable blocks are never promoted to VALID.

## Part B — one gate reading, one active start

phase1.py changes are necessary to fix the setup-heat discrepancy and reuse its scheduler for P23; all its existing self-check importers were rerun. Shared start_gate returns the actual final checked row, and cooling.end_skin_c / cooling.reading / skin_start use that row. No independent cached-temperature decision or reread of a non-empty answer is introduced.

Phase 1 builds monitors/models and starts the camera first, then monitored camera-ON/inference-OFF cooling to T_ref+1.5 C (up to the existing 300 s). The final fresh reading immediately precedes measured t0; setup heat is excluded consistently from measured timing but represented in the starting temperature. Fixed 600 s camera-OFF pauses remain; cooling extension moved after block setup. First block establishes reference, above/below-band comparability labels remain. LMK evidence now starts at the post-gate measured origin; an actual live_block mock with a300-second gate verifies that the LMK window covers all180 measured seconds. The power sample origin is shifted with the cadence origin to retain a real start bracket.

P23 builds both sessions before 300 s camera-OFF idle with llama loaded. Its camera-OFF gate targets idle-start skin+1.0 C for at most 900 s; the SAME final row determines cold_start and recorded skin_start. origin is set immediately on return, before gate-file write or camera start. Camera startup and restart heat/time are INCLUDED in active time; frame/selector slot origin is camera readiness. Unreached gate still proceeds with WARM START — heat-timing results NOT VALID. Regression checks cover same-row identity, reached/unreached limits, band boundaries and one-call parser failure.

## Runner and reused code

One measured invocation is one mode/session. --mode endurance|fixed|adaptive, adaptive --t-hi default37 C (finite25..43 range), --preflight, ONE --dry-run live rehearsal of all three modes, --dry-run --mock offline. Mandatory live current-code rehearsal proof is passed with --rehearsal; failed/mocked/other-code proof is refused. Evidence stems cannot overwrite JSON/server evidence, atomic writes/per-phase files retain incomplete evidence.

Detection is exactly #128 CONFIRM, one size per 1 Hz frame. Existing code reused:

```python
# power_map.py
def size_policy(cadence):
    return None if cadence == 'off' else cr.SizePolicy(interval_s=int(cadence))
def next_size(policy, now=None):
    return 320 if policy is None else policy.next_size(now)
size = next_size(policy)
detections = detector.detect(r['image'], size)
```

```python
# detector_size_policy.py
if now - self.last_large >= self.interval_s:
    self.last_large = now
    return 640
return 320
```

P23 opts into SizePolicy(interval_s=5) in p.run_cycle: 640 replaces 320, first640 at slot5, 47 small/11 large complete service slots in60s (the last2s are unscheduled); legacy Phase-1 default stays 320-every-frame-plus640. pm.build_detector('mid') reuses two intra-op / one inter-op threads, pinned callers/workers4–5, both YOLO sizes.

```python
# speed1/variants.py
CLUSTERS = {'MID': ({4, 5}, 2), 'BIG': ({6, 7}, 2), 'LITTLE': ({0, 1, 2, 3}, 4)}
NAME = 'relsgg-vits16'
def build(variant, cluster):
    cpus, threads = CLUSTERS[cluster]
    return sb.build(NAME, cpus, threads, lambda name, threads: make_head(variant, threads))
# phase23.py
self.head,self.caller,self.tids=self.v.build('fp32','BIG')
```

```python
# phase1.py reused ordering, guarded at planned stop
with lock:
    image, boxes, frame = shared['frame'] or (None,None,None)
if name == 'L2':
    if m2(image, boxes, frame, slot) is None:
        return
# Context is built from those M2 relations, then:
row = ops.select(len(record['selector']), text)
```

Latest COMPLETED640 is stored by the YOLO worker, M2 only immediately before selection. Existing live/fallback/top32 context format and positive inference-time validation remain; forced real fallback in rehearsal feeds a selector. Accuracy NOT EVALUATED. No inference objects run during camera-OFF pauses; sessions/llama remain loaded. Monitor root/su/pump masks exclude4–7; processguard, battery80/25, charger, foreground cores, pinning, #127 sensors, screen/wake restoration, heartbeat and bounded selector transport are reused. Camera state changes and PSS read are serialized to preserve camera-OFF pid expectations. The watchdog is armed after model setup and temporarily disarmed for bounded camera transitions; only the main thread renews the heartbeat, including bounded worker drains. Memory/PSS waits for transitions; other monitors continue. Session proof hashes cover all18 reused code/config paths.

Continuous fast1s, dumps4.87s, memory5.13s, shared5.23s and current_now0.37s sampling. 60s windows include due/done/skipped/ms, M2/ms, selector/ms/misses, receipt-weighted watts with >1.5s gap rejection, skin/status, per-policy caps/tailunknown, MemAvailable/PSS, empty-answer and thermal rereads (separate and total). Events: first policy4/6window>=10% capped, skin35/37/39/41/43, sampledstatuschanges, stopreason, relative to first active origin. Each active/pause phase, cycle and the totals retain camera_start_s, inference_s, camera_stop_s, confirmed camera_off_s and camera_on_without_inference_s. Raw transition timestamps are retained and all intervals are clipped to the measurement. Inference_s spans camera readiness through the last completed call, including inter-call waits; it is not summed parallel CPU time. Inference + confirmed OFF + cameraON-without-inference partitions measured walltime; START/STOP are subsets of cameraON-without-inference. Duty fraction uses inference_s / measured wall seconds. Skin extrema, active caps, pause mean/count, restarts/failed attempts and active capping on every policy remain.

Endurance max30min or earlier SEVERE status3 as regular measured stop. Critical/status4, battery45C, CPU110C three1s samples, stale/sensor/process/cores/camera/server/charger failures take emergency precedence. All emergencies NOT VALID; no exception is accepted as measured performance here. Only SEVERE with all other checks passes VALID — STOPPED AT SEVERE. Fixed/adaptive use36min measured limits. Owner M3 preserves12 cycles of180-second slots (120active slot/60pause slot), with camera transitions included; actual inference/OFF durations vary, and duty_description says so. Fixed phase origins/end limits and the measured mode end follow the2160-second grid; observed_end/completion_lag_s record later evidence/cleanup. Adaptive uses60s minimum fromSTART invocation includingstart, 30s confirmedOFF minimum before restart and2C hysteresis; a terminal pause may be truncated by the36min termination, with no short restart. Transition/termination timing and any review correction are described in the review resolution below.

## Validity and FIX4D hard rules

Cold gate, cadence including skipped frames/call deadlines, H4 camera, memory/PSS, power coverage, monitored readers, cleanup and LMK govern validity. Late monitor/cleanup failure downgrades main AND each mode file and rehearsal_pass. At most3 FIX4D empty-answer rereads in EACH contiguous180s bucket within idle, gate and mode; short final bucket gets samecap. Setup cap3 unchanged, mode/final cleanup separately capped. Thermal attempt/time bounds remain FIX3D, with no new allowance. This scales Phase-1 cap without hiding a burst in a long-run average. Unknown cap tail is explicit; capped percent reuses power_map's later-sample ownership with no invented tail.

ONE re-read only for answer carrying nothing: stdout AND su stderr blank, Popen launch failure or HTTPtimeout. EVERY non-empty answer is judged ONCE by its BASE parser. FIX3D thermal empty-read/time rules and /health2-of-2 remain. All transport/parser/retry modules byte-identical toFIX4D, p.LiveOps/bounded_selector/preflight/memory_row_bad/lmk_window AST-identical. Extended sweep exercises own parsers through actual cr.root mocked transports and real RootShell fake su, including nonempty bad/partial/malformed/stderr-only/status4 answers, blankanswers, launchfailures, HTTPtimeout and all earlierprobes. No new raw reader.

## Checks and rehearsal coverage

- PASS (exit0): `benchmark/campaign/self_check.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix2.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix3.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix3b.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix3c.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix3d.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix4.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix4c.py`.
- PASS (exit0): `benchmark/campaign/self_check_fix4d.py`.
- PASS (exit0): `benchmark/campaign/self_check_p23.py`.
- PASS (exit0): `benchmark/relate_anything/speed1/self_check.py`.
- PASS (exit0): `benchmark/relate_anything/desk2/self_check.py`.
- PASS (exit0): `benchmark/coresidency/test_coresidency.py`.
- PASS (exit0): `benchmark/power_map/test_power_map.py`.
- PASS (exit0): `benchmark/thermal_char/test_thermal_char.py`.
- PASS (exit0): `benchmark/duty_cycle/test_duty_cycle.py`.
- PASS (exit0): `benchmark/camera_power/test_camera_power.py`.
- PASS (exit0): `test_detector_robotcam.py`.
- PASS (exit0): `test_run_mission.py`.
- PASS (exit0): `run_cycle_safety_test.py`.

All are offline tests/mocks with native Termux Python -B; no timed/live benchmark or root attempt. New self_check_p23.py covers real reused SizePolicy scheduler and L2ordering, PartB, cold reached/unreached, fullvirtual30/36mincontrollers, fixed12cycles, adaptive hysteresis/minimums, SEVERE/emergency, OFF/restart, windows/events, burstretrycap, camera/PSS lock, heartbeat, proof/freshstems/latevalidity/latebattery and extendedFIX4D reader sweep. M3 regressions additionally use3-second START/2-second STOP across12 cycles and check exact2160-second measured duration, transition fields in phases/cycles/totals, clipped interval boundaries, partition/sum consistency and inference-based duty fractions. AST syntax and git diff --check passed. The real LiveSession.cycle/run_cycle regression covers fractional tails and a full1800.3-second run with1.0007-second camera drift on a cooperative virtual clock; host scheduling delays cannot become simulated cadence failures. Runtime deadline/count failures are unchanged. It is offline and uses no camera or models. The rehearsal regression also runs the real scheduler through LiveSession.cycle with3-second starts and three40-second active slots: one selector per phase, first forcedfallback followed by two liveM2 calls. Standalone checks/p23/coder_mock_p23_r3* is NOT VALID and rehearsal_pass=false.

An initial Debian interpreter lacked numpy/PIL/requests; native Termux Python was used for final checks. The old D2 main-wiring mock failed after the intentional fix and was updated to assert new post-setup gate wiring; final corresponding checks passed. Initial failed attempt logs are outside the candidate in /tmp; final logged commands/exit codes are in checks/p23/results.json. M3 testing exposed host scheduling sensitivity in accelerated clocks; the slot fixtures use10x pacing and the long drift fixture a cooperative virtual clock, with all original count/deadline assertions retained. All20 checks were rerun after the shared Phase1 fixes; P23-only follow-up changes also receive a final importer self-check. R2 regressions cover slow640 cadence, complete-slot mock planning,18 proof hashes, bracket-emergency reasons, work checkpoints, amlaunch retry failure counts, adaptive restart reserve and main-thread drain/BatteryStop handling.

Owner live rehearsal includes allthree modes, liveM2 each, forcedfallbackM2+selectorcontext once in each mode's first active phase, fixed3pauses, adaptivehighpause AND lowrestart, actualcamera restarts, allshared monitors/cleanup; specifically its adaptive run must restart the camera and no failed start attempt may be hidden. Pass cannot be true for mock/failed/partial/old-code evidence. Real thermalcurve may not reach thresholds in the rehearsal; pass remainsfalse, and sessions stayblocked. Emergencies are mocked, never deliberately provoked by owner. Preflight alone is not rehearsal.

## RUN.md owner instructions

Appended complete P23 section: existing ownerchecklist, nativeforeground/noagents/motorsOFF, freshstems, atomic evidence, signal/hardkill recovery, ONE allmodesrehearsal command, separately endurance/fixed/adaptive commands with mandatory --rehearsal, exactstem/perphase files to send. Expected rehearsal12–15min; endurance37–40min / fixed&adaptive43–46min including setup/idle, plus up to15mincoldgate. M3 keeps measuredfixedmode2160s; camera transition overhead is inside the slots, while setup/idle/gate and final cleanup remain outside the measurement. Rule verbatim: "recharge to >= 80 % and let the phone rest >= 30 min between sessions" (also rehearsal tofirstsession). The runbook is the reviewable ownercommand artifact; no live run was executed here.

## Reviewer CLI configuration and smoke

Claude rounds1–2: version2.1.294, --effort high supported (used). --allowedTools help syntax is comma- or space-separated patterns, example Bash(git *). Exact allowed/disallowed configuration in reviewcommand below: dontAsk, disabled settingssources/MCP, tools Bash/Read/Grep/Glob, only Bash(python3 -I -B -c *), Bash(cat *), Bash(grep *), Bash(ls *), Bash(sha256sum *), plusRead/Grep/Glob; Edit/Write/NotebookEdit/WebFetch/WebSearch denied. Python authorizations are read-only by request; before/afterfull-byte checks independentlyverify nonmutation. The allowed Python pattern is not an OS sandbox; the request explicitly prohibits writes, and full-tree before/after checks verify byte identity. Plan/inspection fallback was not needed for Claude. AGY round3 uses the inspection-only configuration described below. Smoke scratch /tmp/campaign_p23_review_smoke, resultJSON permission_denials includes touchx, actualx_exists=false. Smoke result verbatim:

`print(1)` ran, and `touch x` was refused.

1. **`python3 -I -B -c 'print(1)'`** ran and printed `1`.
2. **`touch x`** was denied by tool permissions. The actual tool result was:
   > Permission to use Bash has been denied because Claude Code is running in don't ask mode.

   As you instructed, I didn't try to create `x` with any other tool or command. The command never ran, so `x` shouldn't exist, though I didn't run a separate check for it.

Requests are under100KB. The entire code/run diff is losslesslygzip/base64 embedded with exactuncompressedSHA/length (Claude requests include an in-memory decode command; Gemini uses read tools on the plain patch); identical full plain patch also retained outsidecopy. Nothing omitted from codediff; data entries onlypath/bytes/SHA. Complete isolated candidate source is inspectable. No `.git`/untrackedweight directories copied; review uses supplied testoutputs andactualsource byinspection. Entire live/copy regularfile byteinventories before/after retained outsidecopy, including ignoredlivefiles.

Failed AGY round3 configuration, explicitly authorized by owner: `agy --model gemini-3.1-pro-high --effort high --mode plan --sandbox --output-format json -p ...`, fresh isolated copy. A workspace PreToolUse hook permits ONLY view_file/view_file_outline/view_code_item/grep_search/list_dir/find_by_name; all commands, edits, artifacts and network tools denied. First smoke could not run print(1) because terminal permission was denied; inspection-only fallback selected. Second smoke actually read READONLY PROBE127 via view_file and refused touchx/run_command AND write_to_file. `x` absent. Full smoke/config/request/log files retained externally under downloads/campaign; global AGY settings unchanged. No inspection commands or tests are run by Gemini; complete supplied checks and actual source are reviewable with read tools. Before/after identity covers extra hook config and live tree.

Claude round3 redo uses the identical smoke-proven tool restrictions from rounds1–2 and `--model claude-opus-5-5 --effort medium` per the latest owner instruction. Fresh copy/request/output names use r3_redo; stdin is delivered by the previously proven Claude invocation. Earlier AGY artifacts are preserved unchanged.

## Independent reviews — verbatim, untruncated

### campaign_p23_r1

Request bytes 88255; exit 0; copy byte identity True; live byte identity True.

Exact command:

```sh
cd /termux-home/review_campaign_p23_r1 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail mode: off. Follow them and end with a final verdict: APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort high --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23_r1.txt > /termux-home/storage/downloads/campaign/review_campaign_p23_r1.json 2> /termux-home/storage/downloads/campaign/review_campaign_p23_r1.stderr
```

Verbatim result text:

# CAMPAIGN_P23 r1 — Independent review (Claude Code CLI, claude-opus-5-5, Ponytail off)

Scope: I reviewed the full patch, which matches the stated SHA-256 and byte count. The git blob ids of `phase23.py`, `self_check_p23.py`, `phase1.py`, `self_check_fix4.py`, `RUN.md` and `RUN_INDEX.md` match the patch index lines. I read the surrounding source, the supplied check logs, the coder's mock evidence and the archived owner bytes. Nothing was written or executed beyond read-only hashing and JSON reads. This is not a git checkout, so I could not re-run the `git show`-based sweep; I re-established what it checks with hashes instead (see Verified).

## Verified OK
- **Part A archive:** all 40 owner `p1_fix4d` files match the supplied SHA-256 values and `archive_sha256_p1_fix4d.json`. The frozen manifest's 131 entries all match. I recomputed the labels and numbers from the bytes and they match RUN_INDEX:
  - T_ref 29.5405 °C; start deltas 1.92 / 2.02 / 1.86 °C.
  - Watts 4.10 / 4.99 / 4.72 / 5.10 / 5.41 / 4.04.
  - YOLO-320 done/due 180/180, and 109/180 for L3.
  - L4 policy6 capped 43.9 %; camera-ON idle policy0 capped 98.5 %.
  - 0 re-reads, 0 thermal retries, 0 LMK kills, session end at +5767.3 s.
  - D2 cooling ran at pause end (e.g. 70.6 s extra before block 6).
- **Unchanged readers:** `runtime.py`, `power.py`, `diagnostics.py` and `coresidency.py` are byte-identical to `frozen_manifest_campaign_p1_fix4d.json`. The phase1 hunks touch only `run_cycle`, `live_block`, `start_gate` and `main`. `LiveOps`, `bounded_selector`, `preflight`, `memory_row_bad` and `lmk_window` are untouched. `docs/` and models are untouched.
- **Configuration matches the request:**
  - M2 runs on BIG cpus 6–7 with 2 threads (`variants.CLUSTERS['BIG']`) and is checked by `check_pinning(tids,{6,7})`.
  - YOLO uses `build_detector('mid')` and `camera_start(rate=1)`; `SizePolicy(5)` replaces 320 with 640 (`phase1.py:78-82,183-214`).
  - L2 runs M2 on the latest completed 640 frame, then the selector, with the same fallback and top-32 rules.
- **Part B (P23):** `start_gate` returns the same reading object that is used for the gate decision, `skin_start` and `gate.json`. The origin is set immediately after it (`phase23.py:329-331`). Camera start is inside active time, and slot origins begin at camera readiness (`phase23.py:227-228`).

## Findings

**H1 — High: the unchanged end-boundary rule makes most measured runs NOT VALID — cadence misses.**
`phase1.py:250` (unchanged) flags any YOLO/M2/selector row with `ended_s > duration`. P23 now passes non-integer and drifting durations: `remaining = planned_end − now` after camera start (`phase23.py:363-365`).

Measured inputs from the owner P1 bytes:
- RobotCam frame period is 1.00037–1.00075 s in every block (from `t − age_s` of the reads).
- YOLO-320 finishes 0.18–0.38 s after its slot; a lone 640 about 0.57 s after.

Concrete triggers:
- **Fixed:** the last slot of each roughly 117.x s active ends at `floor(D)+0.2` (320) or `+0.6` (640). It crosses whenever `frac(D)` is smaller. That is about 25 % per active, about 95 % over 12 actives.
- **Endurance to TIME LIMIT:** the slot origin is set right after the first frame, so frame lag grows by `(T−1)·k`, reaching about 0.67–1.35 s at 1800 s. The last slot's frame therefore arrives after D, so a crossing is nearly certain. A 30-min endurance can only end VALID if it stops at SEVERE (controlled end).
- **Adaptive:** the final active segment, truncated by the deadline, has the same problem (its last selector or YOLO crosses about 20–40 % of the time).

This is untested. `MockSession.cycle` (`phase23.py:272-286`) bypasses `run_cycle`. `check_replacement` uses an integer 60 s and instant frames. The 180 s rehearsal endurance cannot show the drift.

Fix: use integer or explicitly bounded slot planning. Don't start slots whose frame or call can't finish by the planned end, but still require every slot due before the end. Add mocks that run the real `run_cycle` through `LiveSession.cycle` with fractional D and a 1.0007 s frame period.

**H2 — Medium: short final adaptive actives (or any very short active) fail the whole session.**
- `remaining ≤ 0` raises "camera startup consumed active interval" (`phase23.py:363-364`).
- A remaining time under about 5.6 s leaves the slot-0 selector in `ready.wait(timeout=15)`, which ignores `stop` (`phase1.py:127-128`). It then raises "L2 no 640 frame ready".
- Under about 9 s, slot-0 M2 plus the selector crosses the boundary.

Trigger: the adaptive pause meets its switch condition within about 10 s of the 36-min deadline.
Fix: don't start an active phase below a minimum remaining time; make the readiness wait stop-aware.

**M1 — Medium: a controlled end can falsely accept missing selector/M2 calls.**
The selector thread evaluates `stop_when` itself and returns silently (`phase1.py:131-138`). The main loop evaluates it separately (`phase1.py:166-169`). If the condition flaps, the selector thread stops while YOLO continues; examples are adaptive skin around T_hi between dumps, or endurance status going 3→2. A later controlled end then skips every count and boundary check (`phase1.py:250-257`). `scheduled_end` is computed but has no effect under a controlled end.
Fix: latch the stop decision once on the main thread, have the selector check only the latch, and require one selector/M2 per 20 s slot started before the latch.

**M2 — Medium: an emergency can be labelled VALID (race).**
`stop_reason` returns "EMERGENCY: Android CRITICAL…" as an ordinary controlled-end reason (`phase23.py:42-48,361`). Trigger: a status-4 row arrives between `guard()` and `status()`, and the post-phase reading shows status below 4.
- Fixed/adaptive then compute 'VALID'.
- Endurance overwrites the reason with 'SEVERE' (`phase23.py:374,425`).

Fix: any reason starting with EMERGENCY must be NOT VALID and must never be overwritten.

**M3 — Medium (owner decision): fixed timing is not exactly 12 × 120/60.**
The 180 s grid and 2160 s total are exact, but:
- Camera-OFF time per pause is about 60 − stop (about 3.6 s per #128) − evidence/bracket (0.37 s or more), so about 55–56 s.
- Inference time per active is about 120 − camera start (about 1–2 s, up to about 26 s with retries) − the previous phase's overhead.
- The camera stays ON with no inference between the active's end and the pause's stop, and that time is booked as pause.
- `totals` and `cycles` report wall-clock phase time only; there is no `camera_off_s` or `inference_s`.
- Adaptive's 60 s minimum active includes camera start.

Do not accept the RUN.md disclaimer in place of an owner decision. If the owner keeps the grid, record per-cycle and total camera-OFF and inference seconds.

**M4 — Medium: Phase 1 regression in the changed `live_block` — LMK window misaligned.**
`lmk_since` is taken before `start_gate` (`phase1.py:458-459`), which can run up to 300 s. The window `[lmk_since, lmk_since+180]` (`phase1.py:515`) can then miss the measured block. In-block kills would land in `outside_block_lines`, so `n_kills` would read 0. No test covers this.
Fix: take the window from the new t0.

**M5 — Medium (material uncertainty): 30-min frame drift.**
At a 1.0007 s period the lag reaches about 1.2–1.35 s by 1800 s. The slot is skipped once lag + detect ≥ 2 s, which a capped 640 (0.65 s or more, likely above 37 °C) can reach late in endurance. No live or mock test covers this.

**Low findings**
- **L1:** `phase_evidence` catches `BaseException` (`phase23.py:297-298`). A Ctrl-C or SIGTERM (`SystemExit`) arriving during the bracket is swallowed and the run continues. Catch `Exception` instead.
- **L2:** the idle, gate, active and pause files carry no NOT VALID / rehearsal / mock label. Mock mode rows contain `result_validity: 'VALID'` (`coder_mock_p23_r1_fixed.json`). Phase 1 labelled rehearsal block files.
- **L3:** the heartbeat stays armed through camera transitions and drains. A slow camera start with retries can take more than 15 s and trigger the self-SIGTERM (`phase23.py:182-193,210-221`).
- **L4:** `adaptive_low_restart` doesn't require adaptive's own camera restart, because `camera_restarts` is summed across modes (`phase23.py:442-446`). Camera-start attempts above 1 aren't counted as failures.
- **L5:** `code_hashes` omit reused modules: `coresidency.py`, `power_map.py`, speed1/variants, `detector_size_policy.py`, `robotcam_reader.py`.
- **L6:** re-reads after the mode's end (final camera stop, close, LMK, screen restore) are not capped. Battery below 25 % is labelled a monitor failure, not Phase 1's clean battery stop.
- **L7:** `checks/p23` holds stale failing logs that are not in `results.json`:
  - `benchmark_*.stderr`: PIL ImportError, or "NO TESTS RAN".
  - `run_cycle_safety_test.stderr`: PIL ImportError.
  - `speed_block_self_check.stderr`: script path does not exist.

  Also, the new-file headers in the patch read `abenchmark/…` and `bbenchmark/…`, missing the `/`.
- **L8:** the inventory lists `m2_p95_ms` as null for L2/L3, but the bytes contain 1323.1 and 1759.7 ms.
- **L9:** raw samples exist only in the main JSON, which is written at mode end, so a hard kill loses them.

## Explicit answers
- **Can a re-read replace any non-empty answer?** No, as far as I can see. The reader modules are byte-identical to FIX4D, and `phase23.py` adds no reread, retry, transport or parser. `start_gate` only polls after a valid parsed reading (a gate wait) and passes reader errors straight up. H4 frame polling, the pidof owner re-check, /health 2-of-2 and the FIX3D thermal rule are unchanged. Caveat: I could not re-run the git-based sweep.
- **Is Part B correct for both runners?** P23: yes. P1: the same-reading semantics are correct, but M4 is a regression.
- **Does the rehearsal cover every session path?** Every live path, in shortened form. Correctly mock-only: SEVERE, #127 emergencies and refusals. Warm-start (unreached gate) and battery stop are also mock-only but RUN.md doesn't say so. Not covered by any test: H1(b) and M5 (full-length drift), H2 and M1.
- **Are fixed/adaptive timing, emergency handling and evidence validity correct?** No (H1, H2, M1, M2, M3).
- **Is hardware validation beyond the owner rehearsal needed?** Yes:
  - Any fix changes the code hashes, so a fresh rehearsal is mandatory.
  - 30-min drift and boundary behaviour cannot be shown in 180 s, so it needs either deterministic handling plus mocks, or an owner test.
  - The Phase 1 `live_block` change needs its own P1 rehearsal before any reuse.

Approval is not commit authorization; any code change needs a new review.

CHANGES REQUESTED

### campaign_p23_r2

Request bytes 97152; exit 0; copy byte identity True; live byte identity True.

Exact command:

```sh
cd /termux-home/review_campaign_p23_r2 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail mode: off. Follow them and end with a final verdict: APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort high --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23_r2.txt > /termux-home/storage/downloads/campaign/review_campaign_p23_r2.json 2> /termux-home/storage/downloads/campaign/review_campaign_p23_r2.stderr
```

Verbatim result text:

# CAMPAIGN_P23 r2 — Independent review (Claude Code CLI, claude-opus-5-5, Ponytail off)

**Verdict: CHANGES REQUESTED.** Almost every round-1 finding is fixed, and the owner's M3 decision is implemented as stated. One new problem blocks approval: the candidate adds a rule that marks the run NOT VALID whenever any single YOLO call takes more than 1 s (N1). Phase 1 never had that rule, the owner didn't ask for it, and the measured sessions are likely to trip it once the phone gets hot.

Scope: I read the full plain-text patch, the candidate sources (`phase23.py`, `phase1.py`, `self_check_p23.py` and the reused runtime/coresidency/power_map/variants code), the owner archive bytes, the coder's mock evidence and the final check logs in `checks/p23`. Nothing was written, staged or committed. I only ran read-only `python3 -I -B -c`, `ls` and `sha256sum`. No motors, models or hardware were touched.

## Verified (read-only)
- **Candidate integrity:**
  - All 127 entries in `frozen_manifest_campaign_p23.json` match on bytes and SHA-256.
  - The git blob ids of all 6 patched files match the patch `index` lines.
  - New-file headers are now `--- /dev/null` / `+++ b/...` (L7).
- **Base and unchanged readers:**
  - This directory is not a git checkout, so I rebuilt the base in memory by reverse-applying the patch hunks. The rebuilt `phase1.py`, `self_check_fix4.py`, `RUN.md` and `RUN_INDEX.md` match the SHA-256 values in `frozen_manifest_campaign_p1_fix4d.json`, so the base really is FIX4D.
  - An AST comparison of base and candidate `phase1.py` shows only `run_cycle`, `live_block`, `start_gate` (new) and `main` changed. `LiveOps`, `bounded_selector`, `preflight`, `memory_row_bad` and `lmk_window` are AST-identical.
  - `runtime.py`, `power.py`, `diagnostics.py`, `coresidency.py` and the other self-checks are byte-identical to FIX4D.
  - `lag_probe.py` imports only `write`, which is unchanged.
- **Part A archive:** all 40 `owner_*_p1_fix4d*` files are byte-identical to the owner originals in `~/storage/downloads/campaign` and to `archive_sha256_p1_fix4d.json`.
  - `verified_numbers` now carries L2 P95 1323 ms (raw 1323.15) and L3 P95 1760 ms (raw 1759.72), which fixes L8.
  - The RUN_INDEX labels and numbers match the request.
- **Check logs:** `results.json` lists exactly the 20 commands, all exit 0. `checks/p23` holds only those logs, the r2 mock and the inventory; there are no stale logs. The argparse errors in some stderr files are the expected "fresh stem" and "--mock" refusals.
  - Mock evidence: every mock file (idle, gate, active, pause, samples and mode files) is labelled `NOT VALID — DRY RUN`, and `rehearsal_pass` is false.
- **Configuration:**
  - M2 runs fp32 on BIG cpus 6–7 with 2 threads, checked by `check_pinning(tids,{6,7})` (`phase23.py:166-167`).
  - YOLO uses `build_detector('mid')`, `camera_start(rate=1)` and `SizePolicy(5)` with `last_large=t0` (`phase1.py:78-82`); 640 replaces 320 and there is one inference per slot.
  - In L2, M2 runs on the latest completed 640 frame and the selector runs after it.
  - Motors are never imported.

## Findings

**N1 — High (new): any YOLO call over 1.0 s now makes the run NOT VALID, and that is likely under heat.**
- **Where:** `phase1.py:233` (`if replacement and stamp()-a>1.: cadence_missed=True`), documented at `RUN.md:487`.
- **Why it's new:**
  - Phase 1 had no limit on a single call; a slot only failed when a frame was actually skipped (`due_frames` skips once the loop is ≥1 s past the slot).
  - With the new capture-time pacing, a 1.2 s 640 call loses no frame: the next frame, captured at +1.0007 s, is still processed. The rule therefore reports a cadence miss when nothing was missed.
  - Neither E1–E4 nor M3 asks for this rule, and the instruction was "same Phase-1 validity".
- **Concrete trigger, from owner bytes:**
  - YOLO-640 in the owner P1 session ran at P95 723–802 ms and max 843 ms while skin was only 29.5–38.1 °C.
  - In the P1 L2 block, skin rose from 31.5 to 37.0 °C in 180 s.
  - CPU capping starts near 37 °C (#128 CONFIRM). #126 measured the medium cores capped to 1.0–1.2 GHz near SEVERE.
  - Endurance runs until SEVERE (45 °C), and fixed runs about 24 min of hot active time. One 640 call at ≥1.0 s then labels the session `NOT VALID — cadence misses`, in exactly the regime these sessions exist to measure.
- **Uncertainty:** 640 timing under capping in this exact configuration has not been measured.
- **Fix:** either
  - restore Phase 1's skip-based rule (keep "ended past D" and the selector miss checks, and use a larger budget only when planning the final slots), or
  - get an explicit owner decision accepting the 1 s per-call deadline.

  Either way, add a regression test with a slow 640 call (1.0–1.9 s) that loses no frame.

**N2 — Low: each fixed active plans 5 selector/M2 slots, not 6, and this isn't stated.**
- **Where:** `phase1.py:85` uses `floor(duration/20)`; `phase23.py:434-437` passes the remaining time.
- **Trigger:** camera start plus the previous phase's evidence lag leave about 115.6–118 s, so slot 100 is never planned. That is 60 instead of 72 selector calls over 12 cycles, about 17 % less selector work than "every 20 s while active".
- **Disclosure:** the data are recorded (`slot_plan`), but RUN.md and `duty_description` don't state the count.
- **Mock mismatch:** `MockSession.cycle` (`phase23.py:311-319`) plans 2 selectors per 40 s active, while the real scheduler plans 1. The mock evidence overstates the schedule.
- **Fix:** state the planned counts per mode, or have the owner confirm.

**N3 — Low: rare false "adaptive active below minimum".**
- **Where:** `phase23.py:404` checks `deadline-now ≥ 60`, but `camera_start_at` is taken later (after the phase dict, path and flushed print). Line 505 then requires `end-camera_start_at ≥ 60-1e-6`.
- **Trigger:** if `deadline-now` lands within the few-millisecond gap above 60 s, a correctly allowed restart is labelled NOT VALID.
- **Fix:** take the timestamp once, or check against a small margin.

**N4 — Low: an emergency raised during the evidence bracket is recorded only as an evidence error.**
- **Where:** `phase23.py:338-339`. The bracket calls `wait` → `guard` → block_limit, which raises `EMERGENCY`; that is caught as `evidence_error`.
- **Trigger:** this happens on the final endurance phase, or on the last phase at the deadline. The run is still NOT VALID ("phase evidence failed"), but `stop_reason` and the stop event stay `SEVERE` / `TIME LIMIT`.
- **Fix:** record any `EMERGENCY` caught there as the reason.

**N5 — Low: the rehearsal proof's code hashes still miss some imported code.**
- **Where:** `phase23.py:544-548`.
- **Missing:** `server_manager.py` (llama-server binary and thread flags), `speed1/session.py` (imported by `rt.imports`), coresidency's `measure` module, and the selector case file.
- **Mitigation:** `preflight` already records `server_command`.

**N6 — Low: three smaller watchdog and battery-stop gaps.**
- The memory/PSS/charger-status monitor is blocked by `memory_lock` for the whole camera transition (`phase23.py:240`). The power sampler still checks the charger, but the claim that all other monitors keep running is not exactly true.
- The `run_cycle` join after a controlled stop (`phase1.py:261`, up to 25 s) doesn't renew the main heartbeat. A selector call that is still running and takes over 15 s would trigger self-SIGTERM.
- A battery stop hit first in the selector thread becomes a `RuntimeError` (`phase1.py:165-167`). The result is a traceback and non-zero exit, and `stop_reason` reads 'EMERGENCY OR FAILURE' even though the label correctly says battery stop.

**N7 — Low: documentation and evidence gaps.**
- `RUN.md:387` (mock-only list) omits these paths, which are only exercised by mocks: the SEVERE stop, the late-restart suppression, and camera-start retries.
- `RUN.md:369` says "second signal ignored", but that only holds in `main()`'s final cleanup. The same caveat applies to P1.
- Session phase files keep 'IN PROGRESS — NOT VALID' forever; the final validity exists only in the mode and main JSON.
- The 60 s checkpoints hold sensor samples only. YOLO/M2/selector rows for a running active (all 30 min of endurance) appear only when the phase ends.
- If the `am` launch inside `rt.camera_start` fails and its one allowed re-read succeeds, that failure goes into the re-read ledger but not into `camera_failures`.

## Explicit answers
- **Can any re-read replace ANY non-empty answer?** No.
  - The readers are unchanged (byte-level and AST checks above). `phase23.py` adds no transport, parser, re-read or retry.
  - Both `start_gate` paths poll only after a valid parsed reading, and a reader error propagates on the first read; `check_gate` tests this with a single read.
  - The second `camera(False)` in `close()` is a separate cleanup attempt. Any earlier stop failure stays in `cleanup_errors`, so the run stays NOT VALID.
  - H4 frame polling, the pidof owner re-check, /health 2-of-2 and the FIX3D thermal rule are unchanged. The extended sweep output covers all 19 re-read call names plus the thermal reader.
- **Is Part B correct for both runners?**
  - P23: yes. The same reading object is the gate decision, `skin_start` and `gate.reading`. The origin is set right after it. Camera start belongs to the active interval, and slot origins begin at camera readiness.
  - P1: yes. Setup and camera start come before the gate, the gate waits camera-ON with inference idle, `t0` follows the final reading, power rows are shifted correctly, and the LMK window is re-anchored after the gate.
- **Does the mandatory rehearsal cover every session path?**
  - Covered live in shortened form: setup, idle, cold gate, endurance time limit, 3 fixed cycles with pauses and camera restarts, adaptive high/low switching and adaptive's own camera restart, live and fallback M2 and selector contexts, the controlled-end drain, cleanup and LMK.
  - Not covered live: the 30–36 min heat regime (including N1), the SEVERE stop, all 12 fixed cycles, late-restart suppression, camera-start retries, warm start, battery stop and emergencies.
  - Labelling is mostly correct; the gaps are in N7.
  - Rehearsal feasibility depends on the real cooling curve; the docs say so.
- **Is M3 implemented as stated?** Yes.
  - Fixed mode uses the absolute 2160 s grid: 12 × 180 s cycles of 120 s active and 60 s pause. Camera start and stop happen inside the slots, and lag after a boundary goes to `observed_end` / `completion_lag_s`.
  - `phase_timing` (`phase23.py:68-81`) clips all intervals to the measured slot. Confirmed OFF time ends at the next START. START and STOP times are subsets of camera-on-without-inference, and the three parts add up to the measured wall time.
  - All five fields and the inference-based duty fraction are present for phases, cycles and totals. `duty_description` makes no "exactly 120/60" claim.
  - Adaptive's minimum active counts from START and its minimum pause from confirmed OFF. A terminal pause may end at 36 min without a restart, which is consistent with the decision.
  - `check_m3_timing` covers 12 cycles with 3 s starts and 2 s stops, the partition, sums, duty fraction and the adaptive minimums.
- **Are fixed/adaptive timing, emergency handling and evidence validity correct?** Timing and emergency precedence are correct. Validity is not, because of N1; N3 and N4 are minor.
- **Is hardware validation needed beyond the owner rehearsal?** Yes.
  - Any fix to N1 changes the code hashes, so a fresh rehearsal is required.
  - The full-length heat regime can only be checked in the measured sessions.
  - The changed Phase 1 gate needs its own new P1 rehearsal before any P1 reuse.

## Round-1 status
| Item | Status | Evidence |
|---|---|---|
| H1 | RESOLVED | Complete-slot planning (`phase1.py:84-85`, `173`, `192`); capture-time pacing (`phase1.py:225`); real `LiveSession.cycle` covers 4.6/8.1/41.4/1800.3 s with a 1.0007 s camera period |
| H2 | RESOLVED | No restart with under 60 s left (`phase23.py:404-409`, `446`); readiness wait checks stop every 20 ms (`phase1.py:131-135`); minimum pause from `camera_off_at` |
| M1 | RESOLVED | Only the main thread runs `stop_when` (`phase1.py:177`, `254`); selector only guards; required = started slots plus fully due slots; unissued selector allowed only after a cancelled M2 (`phase1.py:274-285`) |
| M2 | RESOLVED | `reason or …` latches (`phase23.py:430`, `450`); EMERGENCY makes the run NOT VALID (`phase23.py:499`); CriticalRace test |
| M3 | RESOLVED | Implemented as stated (above) |
| M4 | RESOLVED | `phase1.py:515-516`; test asserts the 1400–1580 window |
| M5 | UNRESOLVED | Drift is fixed, but a capped 640 call now fails at 1 s instead of about 2 s (N1) |
| L1 | RESOLVED | `except Exception` (`phase23.py:339`); SystemExit test |
| L2 | RESOLVED | All mock and rehearsal files labelled; `checks_validity` is separate from `result_validity` |
| L3 | RESOLVED | Watchdog armed after setup (`phase23.py:181-182`); paused only during camera transitions (memory caveat in N6) |
| L4 | RESOLVED | Adaptive's own restart required; `attempts-1` counted; real fixed-rehearsal test gives fallback/live/live |
| L5 | RESOLVED (as raised) | All five modules named in round 1 are now hashed; remaining gap is N5 |
| L6 | RESOLVED | Mode and final cleanup windows; `BatteryStop` keeps its cause; late battery stop sets the run invalid and `rehearsal_pass` false |
| L7 | RESOLVED | No stale logs; headers fixed |
| L8 | RESOLVED | 1323.1 / 1759.7 ms, raw values kept |
| L9 | RESOLVED | Atomic 60 s checkpoints with 2 s overlap (inference-row caveat in N7) |

Approval is not commit authorization; any code change needs a new review.

CHANGES REQUESTED

### campaign_p23_r3

Request bytes 99703; exit 0; copy byte identity True; live byte identity True.

Exact command:

```sh
cd /termux-home/review_campaign_p23_r3 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 agy --model gemini-3.1-pro-high --effort high --mode plan --sandbox --output-format json -p 'Your complete review instructions are on standard input. Ponytail mode: off. Read-only inspection: do not use commands or create plan artifacts; only read tools. Follow all instructions and end with APPROVED or CHANGES REQUESTED.' < /termux-home/storage/downloads/campaign/review_request_campaign_p23_r3.txt > /termux-home/storage/downloads/campaign/review_campaign_p23_r3.json 2> /termux-home/storage/downloads/campaign/review_campaign_p23_r3.stderr
```

Verbatim result text:

I have read the `REVIEWER.md`, `WORKFLOW.md`, and `HANDOFF.md` instructions. 

However, the standard input provided to this session was completely empty. There is no frozen task, base commit, candidate diff, surrounding source context, or verification output provided for me to review.

Per `REVIEWER.md`: "Review the frozen task, base, complete candidate diff including new files, relevant surrounding source, and actual verification output. Do not accept a summary instead."

Because the review candidate and all required evidence are missing, I cannot perform the correctness or safety review. Please invoke the reviewer again with the complete review request piped to standard input.

CHANGES REQUESTED


### campaign_p23_r3_redo

Request bytes 99933; exit 0; copy byte identity True; live byte identity True.

Exact command:

```sh
cd /termux-home/review_campaign_p23_r3_redo && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail mode: off. Follow them and end with a final verdict: APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort medium --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23_r3_redo.txt > /termux-home/storage/downloads/campaign/review_campaign_p23_r3_redo.json 2> /termux-home/storage/downloads/campaign/review_campaign_p23_r3_redo.stderr
```

Verbatim result text:

# CAMPAIGN_P23 round 3 (redo): independent review

Reviewer: Claude Code CLI, claude-opus-5-5, Ponytail off.

**Verdict: APPROVED.** All round-1 and round-2 findings are resolved and the owner's M3 decision is implemented as stated. I found four new Low items; none blocks approval.

**Scope.** Everything I did was read-only: `Read`/`Grep`/`Glob`, plus in-memory `python3 -I -B -c` hashing, AST comparison and JSON reads. I could not re-run the offline checks myself. I judged them from the actual log bytes in `checks/p23`.

## Integrity checks
- **Patch:** 119995 bytes, SHA-256 `d1af37b6…a8bc0`, matching the request.
- **Patched files:** the git blob ids of all 6 match the patch `index` lines.
- **Base is FIX4D:** I rebuilt the base in memory by reverse-applying the hunks. `phase1.py` (`0dba4cd6…`), `self_check_fix4.py`, `RUN.md` and `RUN_INDEX.md` match `frozen_manifest_campaign_p1_fix4d.json`.
- **Unchanged modules:** `runtime.py`, `power.py`, `diagnostics.py` and `coresidency.py` are byte-identical to FIX4D. The only FIX4D manifest mismatches are the 4 patched files.
- **AST comparison:** only `run_cycle`, `live_block`, `start_gate` (new) and `main` changed. `LiveOps`, `bounded_selector`, `preflight`, `memory_row_bad` and `lmk_window` are AST-identical.
- **Frozen manifest:** all 127 entries in `frozen_manifest_campaign_p23.json` match; all 121 `checks/p23` and `runs/*p1_fix4d*` files are covered. Spot hashes match the request (mock JSON, session JSON, `results.json`, archive inventory).
- **Part A:** all 40 archived files equal their owner sources in `~/storage/downloads/campaign` (source and destination SHA-256). I recomputed from the block bytes:
  - T_ref 29.5405 °C; start deltas +1.92 / +2.02 / +1.86 °C; labels VALID ×3 and NOT COMPARABLE — START TEMP ×3.
  - Watts 4.10 / 4.99 / 4.72 / 5.10 / 5.41 / 4.04; YOLO-320 180/180 except L3 109/180.
  - M2 inference median/P95: L1 4919/6451, L2 1176/1323, L3 1313/1760, L4 871/2001 ms.
  - Selector medians match; L4 policy6 capped 43.9 %; 0 re-reads; 0 LMK kills.
- **Checks:** `results.json` lists the 20 commands, all exit 0. The `campaign_self_check_p23.stdout` log shows every PASS line. Stderr holds only the ORT warning and the expected argparse refusals.
- **Mock evidence:** every coder mock file is labelled `NOT VALID — DRY RUN`, and `rehearsal_pass` is false.

## Explicit answers
1. **Can any re-read replace any non-empty answer?** No.
   - `phase23.py` adds no transport, parser, re-read or retry. Every reader it uses is an unchanged FIX4D wrapper (`read_dump`, `fast_check`, `memory_sample`, `power.sample/sampler`, `server_ok`, `diag.battery`, `camera_start`, `stop_camera`, `lmk_window`).
   - `start_gate` only polls again after a valid, parsed reading. A reader error propagates on the first read (`check_gate` asserts exactly one call).
   - `read_dump` raises at status ≥ `STATUS_STOP=4` and keeps FIX3D. /health stays 2-of-2, and the pidof re-check is unchanged.
   - The second `camera(False)` in `close()` is a separate cleanup attempt. Any earlier failure stays in `cleanup_errors`, so the run stays NOT VALID.
2. **Is Part B correct in both runners?** Yes.
   - **P23:** one reading object serves as the gate decision, `skin_start` and `gate.reading`. The origin is taken right after it (`phase23.py:403-404`). Camera start belongs to active time, and slots start at camera readiness (`phase23.py:265-267`). Models are built before idle.
   - **P1:** setup and camera start come before the gate. The gate waits with the camera on and inference idle, while monitors run. `t0` follows the final reading, power rows are re-based, and the LMK window is re-anchored (`phase1.py:519-528,567-569`).
3. **Does the mandatory rehearsal cover every required normal session path?** Yes, in shortened form:
   - setup, idle, cold gate, endurance time limit;
   - 3 fixed cycles with pauses and camera restarts;
   - adaptive high and low switching plus its own camera restart;
   - live and fallback M2 and selector contexts, drain, cleanup, LMK.

   Mock-only paths, stated at `RUN.md:386-389,543-545`: SEVERE, emergencies, refusals, warm start, battery stop, near-deadline restart suppression, camera retries, and full-length drift. The 30–36 min heat regime can only be seen in the real sessions.
4. **Is M3 implemented as stated?** Yes.
   - Fixed mode uses an absolute 12 × 180 s grid within 2160 s (`phase23.py:439-440`).
   - `phase_timing` (`phase23.py:68-81`) clips intervals to each slot. Camera start and stop fall inside the slots.
   - The five fields plus `duty_fraction = inference_s/wall` are present for phases, cycles and totals, and they add up to wall time.
   - `duty_description` and RUN.md make no "exactly 120/60" claim (`RUN.md:466`).
   - Adaptive keeps a 60 s minimum active counted from START and a 30 s minimum from confirmed OFF.
5. **Is every old and new finding resolved?** Yes; see the tables below.

## Round-1 status
| Item | Status | Evidence |
|---|---|---|
| H1 | RESOLVED | Complete-slot planning (`phase1.py:85-86,176,186`) and capture-time pacing (`phase1.py:228`). Real `LiveSession.cycle` test covers 4.6 / 8.1 / 41.4 / 1800.3 s with a 1.0007 s camera period. |
| H2 | RESOLVED | Stop-aware readiness wait (`phase1.py:133-137`); no adaptive restart with less than 61 s left (`phase23.py:426,468`). |
| M1 | RESOLVED | `stop_when` runs only on the main thread (`phase1.py:180,256`); the selector only guards; required/started/cancelled accounting at `phase1.py:286-297`. |
| M2 | RESOLVED | `reason or …` latch (`phase23.py:452,472`); EMERGENCY makes the run NOT VALID (`phase23.py:526`); CriticalRace test. |
| M3 | RESOLVED | As in answer 4. |
| M4 | RESOLVED | `phase1.py:527-528`; test asserts the [1400, 1580] window. |
| M5 | RESOLVED (code) | Drift removed; no per-call limit. A slot is skipped only once a call reaches about 2 s, the same as Phase 1. |
| L1 | RESOLVED | `except Exception` (`phase23.py:353`); SystemExit test. |
| L2 | RESOLVED | Labels on every phase and checkpoint file (`phase23.py:369`). |
| L3 | RESOLVED | Watchdog armed after setup and paused during camera transitions (`phase23.py:181,238-254`). |
| L4 | RESOLVED | Adaptive's own restart is required; extra attempts and recovered launches count as failures. |
| L5 | RESOLVED | 18 module hashes (`phase23.py:571-577`). |
| L6 | RESOLVED | Mode and final cleanup windows are capped; battery stop keeps its label. |
| L7 | RESOLVED | No stale logs; correct new-file headers. |
| L8 | RESOLVED | The inventory has P95 1323 / 1760 ms (but see new item R3-2). |
| L9 | RESOLVED | Atomic 60 s checkpoints including completed work. |

## Round-2 status
| Item | Status | Evidence |
|---|---|---|
| N1 | RESOLVED | The per-call >1 s rule is gone (`phase1.py:231-235`). Only the 3.35 s terminal budget remains, plus the skip, deadline and count rules. Regressions at 1.01 / 1.5 / 1.9 s pass. |
| N2 | RESOLVED | The mock uses the same formulas with a 3 s START / 2 s STOP (`phase23.py:309-334`). RUN.md states 5 selectors per cycle (60 total) and 1 per rehearsal phase (3 total); only the first active forces fallback. |
| N3 | RESOLVED | 60 + 1 s reserve at both check points (`phase23.py:426,468`); NearDeadline test. |
| N4 | RESOLVED | The bracket exception sets `evidence_stop_reason` and that becomes the stop reason (`phase23.py:355,480,500-501`); test. |
| N5 | RESOLVED | `server_manager`, `session`, `measure` and the cases file are hashed; the test asserts 18 entries. |
| N6 | RESOLVED | Memory lock disclosed (`RUN.md:535`); main-thread-only drain heartbeat; `BatteryStop` keeps its type (`phase1.py:280`). |
| N7 | RESOLVED | Incremental checkpoints of completed work; recovered `am` launch counted as a failure; authority and signal wording (`RUN.md:540-545`). |

## New findings (all Low; none blocks approval)
- **R3-1 — Low.** `phase23.py:484-485`: the pause→active switch records `skin_c` from the post-pause `skin_end` reading, not from the reading that met the low threshold. `coverage()` (`phase23.py:558`) relies on that value.
  - Trigger: a slight sensor uptick between the two readings sets `adaptive_low_restart` false. The rehearsal then fails even though the restart really happened. It fails closed, but it could cost a rehearsal.
- **R3-2 — Low (documentation).** `RUN_INDEX.md:50-51`: the L2 and L3 M2 cells show only medians, although the table says median/P95. The byte-derived P95 values are 1323 and 1760 ms.
- **R3-3 — Low.** `phase23.py:494-497`: in the rehearsal, a camera-stop failure at the end of a mode is recorded but doesn't end the session. The next mode's idle tries the stop again, and only fails the session if that second stop also fails. The evidence stays NOT VALID and `rehearsal_pass` stays false, so this only slightly contradicts `RUN.md:502-503`.
- **R3-4 — Low (labelling).** `phase23.py:355`: any non-battery failure during the evidence bracket is labelled `EMERGENCY: evidence bracket`, including a plain monitor failure. The run is still NOT VALID. Also, fixed-mode `switches` rows carry adaptive `threshold_c` values, which is cosmetic.

## Material uncertainties
- **Rehearsal feasibility:** in the shortened adaptive mode, the phone has to cool below the final gate reading minus 1 °C with at least 61 s left. That depends on the real cooling curve (disclosed). If it doesn't, the rehearsal must be rerun.
- **Capped 640 calls:** if a YOLO-640 call reaches about 2 s while capped near SEVERE, the next slot is skipped and the run becomes NOT VALID under the same Phase 1 rule. Only the measured sessions can show whether this happens.
- **Phase 1 gate on hardware:** the changed P1 gate is not validated on hardware; a new P1 rehearsal is needed before any P1 reuse (disclosed).
- **My verification:** all checks ran offline in proot with agents resident. I couldn't run git or the tests, so I relied on the logs plus my own hash and AST reconstruction.

Approval is not commit authorization: commit needs an explicit yes, and push a separate yes. Any code change after this review needs a new owner workflow and review. No DOC DIFF.

APPROVED



## Review resolution and commit preparation

Round 1: CHANGES REQUESTED (verbatim above). The first isolated reviewer copy and whole live tree were byte-identical before/after; no fallback or silent reviewer substitution occurred.

Corrections implemented before the next fresh review:

- H1/M5: complete terminal service slots, capture-time pacing, and a real LiveSession.cycle/run_cycle regression spanning1800.3 simulated seconds with1.0007-second camera drift. No planned misses or overdue calls are excused.
- H2: no final adaptive restart with less than60 seconds remaining; stop-aware readiness wait. Camera startup, inference and confirmed OFF seconds are recorded separately.
- M1/M2: main-thread-only latched stopping, mandatory counts retained for early stops, emergency validity/precedence fixed; temperature/status fluctuations cannot clear a stop.
- M4: Phase1 LMK epoch anchored after the cooling gate; actual caller regression with300-second gate.
- L1/L2: interrupts propagate while atomic partial files persist; mock/rehearsal phase/raw/results always NOT VALID, structural checks separate.
- L3/L4: bounded camera transitions do not falsely expire the watchdog; adaptive's own camera restart required; failed attempts counted even when a retry succeeds.
- L5/L6: current-code proof expanded to14 reused paths; mode/final cleanup rereads capped; normal and late battery-below25 stops retain their cause and invalidate results without being rewritten as a monitor crash.
- L7/L8/L9: old failed/no-tests logs removed; new-file diff paths corrected; actual L2/L3 P95 retained in own inventory; atomic60-second raw checkpoints with2-second overlap survive hard kills.
- Additional completeness checks: active capping includes every CPU policy; windows record separate and total FIX4D/FIX3D reread counts without altering any parser or thermal bounds; below-band start regression retained.

M3 owner decision: preserve exactly12 x180-second wall-clock slots over2160 seconds. Camera transitions are inside120-second active and60-second pause slots; there is no claim of120seconds inference/60seconds confirmedOFF. New helper clips all five timing fields to measuredslots and partitions inference/OFF/cameraON-withoutinference; START/STOP are subsets of cameraON-withoutinference. Duty fraction is inference_s / measuredwalltime, including perphase/cycle/totals. Fixed measuredphaseorigins/endlimits followthegrid; observed reader/evidence/cleanup lag is outside measured2160seconds. JSON duty_description and RUN.md state this decision and operational definitions. Adaptive minimumactive is60seconds fromSTART invocation, includingstart, minimum restartpause30seconds fromOFF confirmation. A terminalpause canend at36minutes withoutrestarting, asdocumented; shortnormal-endactives areinvalid. Regression:12 fullfixedcycles with3-secondSTART/2-secondSTOP overhead, allfieldpartitions/sums/duty/limits, adaptiveminimums and clippedinterval boundaries.

Round2 request includes round1 result text VERBATIM and asks explicitly whether H1,H2,M1-M5,L1-L9 are resolved and M3 implemented as stated. Same Claude/model/effort/permissions/Ponytailoff configuration; freshisolated copy and fullbefore/aftertreebyteidentity required. No furtherownerpermission orfallback is assumed.

Rehearsal coverage correction before round2: full20-second selector slot planning leaves just one selector in a40-second fixed rehearsal active afterstartup. Forced fallback now occurs only in the first active phase ofeachmode; subsequentfixedactives exercise liveM2. A REAL LiveSession.cycle/run_cycle/controller regression with3-secondstarts requires scenesfallback/live/live and allthree pauses. This preserves the shortened180-second fixed rehearsal and doesnot alter session behavior. Accelerated wall-clock mocks exposed host scheduler sensitivity; the short slotfixtures nowuse10x pacing and the full1800.3-second/1.0007-period fixture uses a cooperativevirtualclock. No runtime failure check was weakened. All20commands rerunforM3; ownP23importer rerun afterthese latestchanges.

Round2 CHANGES REQUESTED: N1 skip-based cadence restored,terminalbudget3.35s only; slow6401.01/1.5/1.9svalidregressions. N2 matchingmockcomplete-slotplans,3sSTART/2sSTOP andtypicalselectorcountRUNdisclosure. N3 adaptive1srestartreserve. N4 bracketemergency overridesmode/stopreason. N5 currentproof18paths inclserver_manager,session,measure,cases. N6 memory-transitionserializationdisclosed; main-onlydrainheartbeatandBatteryStoptypepreserved. N7 incrementalworkcheckpoints,amlaunchretryfailurecounts,validityauthority/signal/mock-onlydisclosures. ALL20checks rerun,newstandaloneR3mockNOTVALID,refreeze. Owner explicitly changed finalreview to AGY Gemini3.1ProHigh; inspection-only readtoolhook passedactualread/refusedtouch+write smoke. No Claude fallback was substituted silently.

Round3 AGY invocation completed exit0,copy/live byteidentityTRUE,99703-byte requestfile. Gemini response says stdin was empty and it could not inspect the candidate; returnedCHANGES REQUESTED without substantive review. Coder used Claude-style stdin redirection with AGY -p generic prompt; AGY did not deliver that request to the model. This is an invocation failure, NOT a correctness review or approval. Three invoked rounds exhausted. STOP/noautomaticretry/no commitproposal. A further invocation requires an explicit owner decision about the round limit. A corrected AGY invocation must pass complete request content as the -p argument, with the proven read-only hook and another fresh isolated copy, and retain byteidentity checks; no such retry has run. Candidate remains frozen/uncommitted.

Owner authorization supersedes the previous stop: "redo the third review with claude code opus 5.5 medium". Round3 redo uses Claude Code2.1.294,claude-opus-5-5,--effort medium,Ponytailoff,the previously smoke-proven readonlyallowed/disallowedconfiguration and a fresh review_campaign_p23_r3_redo copy. Frozen127manifestentries stillmatch; all20priorcheckexits0; no code/evidence changes sincefreeze and no extra hardware/benchmark/test runs. FailedAGYattempt remains preserved verbatim with its identities; redo status/verdict will decide readiness. No commit/push authorization implied.

Round3 redo verdict APPROVED (Claude Code Opus5.5 medium). Request99933bytes,exit0,failureNone,copy2497/live4097 byteidentityTRUE. Every R1 H1,H2,M1-M5,L1-L9 and R2 N1-N7 RESOLVED; M3 implementedasstated. Four new nonblockingLow findings retained without postreview changes: R3-1 adaptive low-switch skin recorded after decision can falsely fail rehearsal; R3-2 RUN_INDEX L2/L3 cells omit P95 though actual values are retained in inventory/report; R3-3 rehearsal mode-end camera stop failure remains invalid but next mode can try cleanup again; R3-4 bracket labels monitor failure EMERGENCY and fixedswitches carry cosmeticadaptive thresholds. All finalverbatimfindings/uncertainties above. Candidate unchanged; no newcode/docs/evidence edits afterapproval. Proposed exact128-file commit includes both ignoredserverlogs, to force-add only after owner saysyescommit; pushonlyseparateyespush. No DOC DIFF/livevalidation.

Final review verdict: APPROVED

Proposed one commit (human authorization remains pending):

`benchmark: campaign phase-1 session archive, start-temp fix, phase 2/3 runner (L2 endurance, fixed and adaptive duty cycle)`

Exact file list, INCLUDING ignored serverlogs and the manifest itself:

```text
benchmark/campaign/RUN.md
benchmark/campaign/RUN_INDEX.md
benchmark/campaign/checks/p23/_run_cycle_safety_test.stderr
benchmark/campaign/checks/p23/_run_cycle_safety_test.stdout
benchmark/campaign/checks/p23/_test_detector_robotcam.stderr
benchmark/campaign/checks/p23/_test_detector_robotcam.stdout
benchmark/campaign/checks/p23/_test_run_mission.stderr
benchmark/campaign/checks/p23/_test_run_mission.stdout
benchmark/campaign/checks/p23/camera_power_test_camera_power.stderr
benchmark/campaign/checks/p23/camera_power_test_camera_power.stdout
benchmark/campaign/checks/p23/campaign_self_check.stderr
benchmark/campaign/checks/p23/campaign_self_check.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix2.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix2.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix3.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix3.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix3b.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix3b.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix3c.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix3c.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix3d.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix3d.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix4.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix4.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix4c.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix4c.stdout
benchmark/campaign/checks/p23/campaign_self_check_fix4d.stderr
benchmark/campaign/checks/p23/campaign_self_check_fix4d.stdout
benchmark/campaign/checks/p23/campaign_self_check_p23.stderr
benchmark/campaign/checks/p23/campaign_self_check_p23.stdout
benchmark/campaign/checks/p23/coder_mock_p23_r3.json
benchmark/campaign/checks/p23/coder_mock_p23_r3.stderr
benchmark/campaign/checks/p23/coder_mock_p23_r3.stdout
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_active_01.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_active_02.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_gate.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_idle.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_pause_01.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_pause_02.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_samples_0001.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_samples_0002.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_samples_0003.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_samples_0004.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_adaptive_samples_0005.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_active_01.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_gate.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_idle.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_samples_0001.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_samples_0002.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_samples_0003.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_samples_0004.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_endurance_samples_0005.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_active_01.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_active_02.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_active_03.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_gate.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_idle.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_pause_01.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_pause_02.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_pause_03.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_samples_0001.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_samples_0002.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_samples_0003.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_samples_0004.json
benchmark/campaign/checks/p23/coder_mock_p23_r3_fixed_samples_0005.json
benchmark/campaign/checks/p23/coresidency_test_coresidency.stderr
benchmark/campaign/checks/p23/coresidency_test_coresidency.stdout
benchmark/campaign/checks/p23/desk2_self_check.stderr
benchmark/campaign/checks/p23/desk2_self_check.stdout
benchmark/campaign/checks/p23/duty_cycle_test_duty_cycle.stderr
benchmark/campaign/checks/p23/duty_cycle_test_duty_cycle.stdout
benchmark/campaign/checks/p23/entry_inventory.json
benchmark/campaign/checks/p23/power_map_test_power_map.stderr
benchmark/campaign/checks/p23/power_map_test_power_map.stdout
benchmark/campaign/checks/p23/results.json
benchmark/campaign/checks/p23/speed1_self_check.stderr
benchmark/campaign/checks/p23/speed1_self_check.stdout
benchmark/campaign/checks/p23/thermal_char_test_thermal_char.stderr
benchmark/campaign/checks/p23/thermal_char_test_thermal_char.stdout
benchmark/campaign/frozen_manifest_campaign_p23.json
benchmark/campaign/phase1.py
benchmark/campaign/phase23.py
benchmark/campaign/runs/archive_sha256_p1_fix4d.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d.stderr
benchmark/campaign/runs/owner_rehearsal_p1_fix4d.stdout
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_01_L0.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_02_L1.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_03_L2.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_04_L3.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_05_L4.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_block_06_L0.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_idle_1.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_idle_2.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_idle_3.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_llama-server.log
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_01_L0.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_02_L1.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_03_L2.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_04_L3.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_05_L4.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_pause_06_L0.json
benchmark/campaign/runs/owner_rehearsal_p1_fix4d_warmup_L0.json
benchmark/campaign/runs/owner_session_p1_fix4d.json
benchmark/campaign/runs/owner_session_p1_fix4d.stderr
benchmark/campaign/runs/owner_session_p1_fix4d.stdout
benchmark/campaign/runs/owner_session_p1_fix4d_block_01_L0.json
benchmark/campaign/runs/owner_session_p1_fix4d_block_02_L1.json
benchmark/campaign/runs/owner_session_p1_fix4d_block_03_L2.json
benchmark/campaign/runs/owner_session_p1_fix4d_block_04_L3.json
benchmark/campaign/runs/owner_session_p1_fix4d_block_05_L4.json
benchmark/campaign/runs/owner_session_p1_fix4d_block_06_L0.json
benchmark/campaign/runs/owner_session_p1_fix4d_idle_1.json
benchmark/campaign/runs/owner_session_p1_fix4d_idle_2.json
benchmark/campaign/runs/owner_session_p1_fix4d_idle_3.json
benchmark/campaign/runs/owner_session_p1_fix4d_llama-server.log
benchmark/campaign/runs/owner_session_p1_fix4d_pause_01_L0.json
benchmark/campaign/runs/owner_session_p1_fix4d_pause_02_L1.json
benchmark/campaign/runs/owner_session_p1_fix4d_pause_03_L2.json
benchmark/campaign/runs/owner_session_p1_fix4d_pause_04_L3.json
benchmark/campaign/runs/owner_session_p1_fix4d_pause_05_L4.json
benchmark/campaign/runs/owner_session_p1_fix4d_pause_06_L0.json
benchmark/campaign/runs/owner_session_p1_fix4d_warmup_L0.json
benchmark/campaign/self_check_fix4.py
benchmark/campaign/self_check_p23.py
```

frozen_manifest_campaign_p23.json records candidate sizes/SHA256 (excludes itself to avoid recursivehash); source archivehashes matched again at freeze. Report is external torepo and is not in proposedcommit. No docs/DOC DIFF/models/committed-evidence modifications. Commit only after owner says "yes commit"; push only after separate "yes push". Reviewer approval never overrides this gate.

Skipped: live/root/timed validation (owner only, agents must exit). Risks: realcamera/sensor/thermal/cooling timing remains unvalidated until mandatory ownerrehearsal; no heat-timing validity asserted for a warmstart or an offline mock.


## Commit receipt

Owner authorized "yes commit". Created `49bc13bc6a3f1eb1fca242d344a4cdb1c9c77541` with the approved title and exact128files, including both force-added ignored serverlogs. Parent `074d5b92ce32330d6fb3c04d9d0ee7db0556ef3d`; committed bytes match every frozen entry and reviewed manifest. Working tree status --short: ''. No push; separate "yes push" remains required. The reviewed report above is the precommit record. Staged whitespace check found only a literal trailing space at benchmark/campaign/checks/p23/duty_cycle_test_duty_cycle.stdout:13 in unchanged approved raw output; preserved byte-for-byte, all other staged paths passed. Full receipt: campaign/commit_campaign_p23.json. No code/evidence/doc changes after review.


## Push receipt

Owner authorized "yes push". Pushed main to origin; git ls-remote verifies refs/heads/main at `49bc13bc6a3f1eb1fca242d344a4cdb1c9c77541`. Working tree clean. Earlier no-push statements describe the pre-push record. No live validation or DOC DIFF.
