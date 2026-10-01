# CLEANUP 1 — Coder report (co-residency + thermal_char)

**Role:** Coder. Claude Code CLI, `claude-opus-5-5`, medium effort, Ponytail lite.
**Reviewer:** Codex CLI 0.158.0, `gpt-6-sol`, effort medium, read-only sandbox, Ponytail off. Each round ran in a fresh `codex exec` session: `codex exec -m gpt-6-sol -c model_reasoning_effort="medium" -s read-only -C /termux-home/robot - < request.md`. No fallback was used.
**Base:** `main` at `3584b5eddd3639c00c37b24e6e31f4d86fce2f2b`. Before any edit I checked HEAD, the clean tracked tree and all 12 frozen SHA-256 values (CORESIDENCY_FIX1_REPORT.md, THERMAL_CHAR_FIX1_REPORT.md), and all matched. At the end, HEAD and the clean tracked tree still hold. Nothing was staged, committed or pushed.

## Outcome: STOPPED. Round 3 verdict is REQUEST CHANGES (3 blocking findings open)

The task allows 3 review rounds. Round 3 was not APPROVE or APPROVE WITH NOTES, so I stopped. I made **no edits after round 3**. The files on disk equal what round 3 reviewed: `sha256sum` against `cleanup1_review_artifacts/round3/frozen.sha` shows no difference.

### Open round-3 findings (not fixed), with my reading
1. **B5 has no cooldown gate.** `mtp_snapshot()` starts RobotCam and the MTP Gemma without `thermal_gate()`. The task says "before each block (and before the Gemma loads)". B5 is the untimed RAM snapshot, so I read "block" as B1-B4 and did not gate B5. A gate there would be about 1 line (`thermal_gate(..., 'B5', smoke)`, recorded in the B5 JSON). This needs a decision on whether B5 counts as a block.
2. **A pid left after the force-stop is accepted.** `camera_end_failed()` checks only "capture kept advancing" and "force-stop failed (rc)". `pids_after_force_stop` is recorded but not judged. The task says exactly this: "then record pidof. INCOMPLETE only if capture kept advancing or the force-stop failed". The reviewer wants a pid left after the force-stop to fail too. This needs a decision. The fix would be 1 line, but Android can restart a process for other reasons, so it could add false INCOMPLETEs.
3. **Capped-time attribution.** Each in-block 1 s sample counts the interval since the previous sample (round-1 fix). Across a skipped slot (e.g. capped at 1 s, next reading at 5 s uncapped), the 1-5 s gap is credited to the 5 s reading. The tail after the last sample (< 1 s plus at most one detection) is not counted. This is a real limitation of the convention, because a skipped slot is no evidence either way. A fix would count gaps > 1.5 × 1 s as "unknown" (like unreadable values, flagged) and add the tail. It is not applied, because it would need a 4th round.
4. Non-blocking (all rounds): SMOKE.md keeps stale prose ("each gated on zone9 ≤ idle + 4 °C", and the INCOMPLETE list lacks the new causes). It is unchanged, because the task allows SMOKE.md/RUN.md edits only if commands change, and no command changed.

## What changed (frozen → final)

### coresidency.py
1. **Heat stop:** the zone9 > 80 stop (main.py regex parsing, `PAUSE_ZONE`, `heat_reached`) is removed. The block now stops at the first of:
   - Android status ≥ 4 (CRITICAL);
   - battery ≥ 45.0 °C;
   - BIG/MID/LITTLE max ≥ 110 °C in 3 consecutive 1 s samples (`cpu_fault`).

   These are checked between frames by the pure `block_limit()`. A limit stop is a valid result. `heat_stop` now records `limit`, `reason`, `time_to_limit_s` and `reading` (latest skin, status, battery, CPU max and scaling_max at the stop). The report rows are "stop limit" (`-` if none) and "time to limit s" (`-` if none).
2. **Readings:**
   - One persistent root shell per run (`RootShell`), with the layout discovered at start (`discover`: lowest zone of each type BIG/MID/LITTLE, every `policy*/cpuinfo_max_freq`; exits if incomplete).
   - Per block, a 1 s monitor (CPU zones, battery temp, scaling_max_freq per policy) and a 5 s monitor (`dumpsys thermalservice` via `root()`, the #123 form). Both are stamped at completion and start before `camera_start`; the runner waits ≤ 10 s for the first of each.
   - Fail closed (block stopped, INCOMPLETE): no skin + status reading for 60 s. I also kept thermal_char's "no CPU-zone / battery reading for 5 s", because otherwise the fault and battery stops would be blind.
   - Readings are stored as `fast` and `dumps` in the block JSON, with t in s from block start (negative = before the camera start).
   - **Copied** from thermal_char.py, because thermal_char imports coresidency: `RootShell` and `TEMP_RE`/`parse_dump` verbatim (END marker renamed), and `to_int`. `discover` and the 1 s sample script are cut down to the limit inputs.
3. **Report rows:**
   - VIRTUAL-SKIN start/end/max;
   - Android status max;
   - "policy capped s (% of block)": time-weighted over in-block samples, % of `duration_s`, and any unreadable scaling_max_freq → INCOMPLETE (round 1);
   - "lowest scaling_max MHz 0/4/6" (policies from the layout);
   - "stop limit";
   - the idle skin in the header.
4. **Gate** (`thermal_gate`, before every block and before the loads): it waits until skin ≤ idle skin + 1.5 °C AND z9 ≤ idle + 4 °C. Idle skin is a dump read at runner start, after oneshot's 5 min idle; the run refuses to start without it. If the gate is not reached in 15 min, the block starts anyway as `warm_start`. The report shows `gate wait s` as e.g. "900 warm start", not INCOMPLETE. The loads line shows the loads' wait and warm start. `--smoke` still skips the wait.
5. **Limit at start** (round-3 finding of FIX1): a limit reached with 0 frames processed and no failed reads gives one line, "not run: limit at start (reason; skin, status, battery, CPU max, scaling_max)", INCOMPLETE. The other rules are skipped for that block.
6. **RobotCam end check:** `camera_end_check(rootf)`.
   - After STOP, capture counts as stopped once every read for ≥ 3 s is `missing` or the same (session, frame). A new frame or an unreadable read restarts the window, and the check gives up at 15 s.
   - Then `am force-stop com.pixelrobot.robotcam` and `pidof`, both in the #123 form; the force-stop also runs when capture did not stop.
   - `camera_end_failed()` fails only when capture was not shown stopped or the force-stop failed.
   - It runs after every B1-B4 STOP (in `finally`, so it also covers CoresLost, abort and a raising broadcast), after B5's STOP (recorded; a failure makes the run INCOMPLETE), and in main's final cleanup (printed).
   - `--resume` refuses run folders without `block_stops` in run.json (older runner).

### thermal_char.py (item 6 only)
- The `time.sleep(2)` + `pidof` after the load stop is replaced by `cr.camera_end_check(root_file)` in both the normal and the abort path.
- The report fails `after_stop.robotcam` only through `camera_end_failed()`. A cached pid is no longer INCOMPLETE.
- Docstring updated.

### Tests
- `test_coresidency.py`:
  - New `check_cleanup1`: limit edges (status 3/4, battery 44.9/45.0, CPU 2 vs 3 consecutive), skin/status loss fail-closed at 60 s, CPU/battery loss at 5 s, frame_loop with a limit at start, gate reached / warm start / z9 still gates / no skin, end check (stopped, still advancing, unreadable reads, force-stop rc 1 and timeout), capped-time weighting and unreadable flag, discovery through the real RootShell with a fake su and sysfs (decoy zones).
  - Updated fix1 checks: limit at start is now INCOMPLETE with its readings; fail-closed line.
  - E2E smoke: B4 stops on status 4 while z9 stays at 90 and status 3 does not stop; new rows asserted; warm-start report row; the process stays cached after STOP until the force-stop; a force-stop after the last start in the full, SIGTERM and resume runs; B5 end check; a new failed-STOP-broadcast child run.
- `test_thermal_char.py`: a fake `am` for the force-stop; STOP now leaves the pid cached, as on the phone; every scenario asserts capture stopped, force-stop rc 0 and no pid after; a report unit check (cached pid OK; advancing / force-stop failure → INCOMPLETE).
- I removed a `coresidency/__pycache__/` that my first test run created; it was not in the frozen candidate.

## New SHA-256 (final = round-3 frozen)
```
3e3cd4e21f476d32e5784ebf51b65b9076cb5cd1b6b233688033d3044cdbabae  coresidency/SMOKE.md
85a6e4cb003bd4f56d501c186a8cbac1b528fb1aa4369380c34be25ad2743626  coresidency/coresidency.py
efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2  coresidency/oneshot.sh
dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0  coresidency/run_coresidency.sh
fca2eacd4880ddc98fe4846cc67719826e5952539e2dffee59b742ed7a48ec45  coresidency/test_coresidency.py
bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19  coresidency/test_oneshot.sh
46c49c727fa02fbc71b4fd5a97c1919a53652bf802a335cfe4a3dea9c03a336d  thermal_char/RUN.md
16787b0679c24b711eb9e370cdcc9a9942cf689718764874b8ec3e418f84d77b  thermal_char/oneshot.sh
c4df2fa2799f019fd67ac68580c9296001e1e8ea63d51f75ab906184864d682e  thermal_char/run_thermal_char.sh
e640e6f33e0c44585322f37394b68936a59b22adc32bde824b649d3d7a6ee2f6  thermal_char/test_oneshot.sh
e54d5a9338fa01837f316e58241007f58d5ba9e695787c0316d9ee14dc609b43  thermal_char/test_thermal_char.py
b5b0c812c535cb69987ffb726bd1a37d2ba4d341c7fedd20d7b13d23e8f89b84  thermal_char/thermal_char.py
  2882 coresidency/SMOKE.md
 67834 coresidency/coresidency.py
  7351 coresidency/oneshot.sh
   426 coresidency/run_coresidency.sh
 49281 coresidency/test_coresidency.py
  8056 coresidency/test_oneshot.sh
  3741 thermal_char/RUN.md
  7687 thermal_char/oneshot.sh
   424 thermal_char/run_thermal_char.sh
  8818 thermal_char/test_oneshot.sh
 33902 thermal_char/test_thermal_char.py
 43958 thermal_char/thermal_char.py
234360 total
```
Changed: `coresidency/coresidency.py` (was 65c572e0…), `coresidency/test_coresidency.py` (was 6ea51d6e…), `thermal_char/thermal_char.py` (was cbf73d12…), `thermal_char/test_thermal_char.py` (was a610c2bd…). Unchanged: both `oneshot.sh`, `run_*.sh`, `test_oneshot.sh`, `SMOKE.md`, `RUN.md`.

`git rev-parse HEAD` / `git status --short --untracked-files=no`:
```
3584b5eddd3639c00c37b24e6e31f4d86fce2f2b
(end)
```

## Check output (final candidate)
`security_reminder_hook`: `pgrep -fa '[s]ecurity_reminder_hook'` before every timing-sensitive test run returned **no process (rc 1)**. My first attempt used the plain pattern and matched only my own shell wrapper, whose command line contained the string. I killed nothing.
```
== git
HEAD 3584b5eddd3639c00c37b24e6e31f4d86fce2f2b; tracked changes: []
bash -n ok (all 6 shell files)
== py_compile:
/usr/bin/python3 3.13.5 py_compile ok
/data/data/com.termux/files/usr/bin/python 3.13.13 py_compile ok
== pgrep -fa '[s]ecurity_reminder_hook' before each timing-sensitive test: no process (rc 1)
== test_coresidency.py (native Termux python inside proot), rc 0:
block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS
cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS
RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure INCOMPLETE; cached pid recorded only: PASS
capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable scaling_max INCOMPLETE: PASS
layout discovery, 1 s sample and dump parsing: PASS
selector slots = those that start inside the block (smoke 1, full 9): PASS
limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop INCOMPLETE: PASS
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
failed STOP broadcast: force-stop still runs: PASS
== test_thermal_char.py, rc 0:
ok: parse_dump and every stop condition at its edge
ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, slow read stamped at completion, missing CPU zone refused
ok: interrupted request keeps its 5 streamed tokens; without a load stop it is a load failure
ok: a slow thermalservice dump is stamped at completion (start kept)
ok: duration: stop "planned load duration 8 s reached" after 8.0 s of load, exit 0
ok: skin: stop "VIRTUAL-SKIN 48.0 >= 48.0 degC" after 3.3 s of load, exit 0
ok: battery: stop "battery 45.0 >= 45.0 degC" after 3.3 s of load, exit 0
ok: status: stop "Android thermal status 5 >= 5 (EMERGENCY)" after 3.3 s of load, exit 0
ok: cpu: stop "CPU zone >= 110 degC in 3 consecutive 1 s samples (max [110.0, 110.0, " after 3.5 s of load, exit 0
ok: noskin: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.0 s of load, exit 1
ok: skinlost: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.0 s of load, exit 1
ok: sensorfail: stop "fail closed: no battery temperature reading for 6 s" after 6.3 s of load, exit 1
ok: noframes: stop "load failed: no new RobotCam frame for 2 s" after 4.5 s of load, exit 1
ok: serverdie: stop "load failed: llama-server request failed: RuntimeError: stream ended w" after 3.1 s of load, exit 1
ok: cores: stop "cores 4-7 lost (allowed [0, 1, 2, 3, 4, 5])" after 3.0 s of load, exit 1
ok: startup: stop "during startup: VIRTUAL-SKIN 48.5 >= 48.0 degC", exit 1
ok: detectfail: stop "load failed: frame/detect loop failed: ValueError: fake detector failu" after 3.0 s of load, exit 1
ok: sigterm: stop "aborted: SystemExit: 143" after 1.8 s of load, exit 143
ALL OK
== coresidency/test_oneshot.sh rc 0 (tail):
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run
== thermal_char/test_oneshot.sh rc 0 (tail):
ok: refuses with a thermal runner running
ok: refuses while the co-residency launcher holds its lock
ok: note with spaces reaches the runner
```

## Unverified
- Nothing ran natively on the phone; all tests are offline (fake su, sysfs, dumpsys, RobotCam frames, llama-server).
- The cut-down `discover` on the real layout (expected BIG=9, MID=10, LITTLE=11; policy0/4/6, as thermal_char found).
- `am force-stop` through `su -c` in the #123 form on the Pixel, its return code, and that pidof is then empty.
- That a stopped RobotCam really reads as `missing` within 3 s on the phone (README: STOP deletes frame.jpg).
- That `dumpsys thermalservice` every 5 s, beside the 5 s PSS su call and the 1 s root shell, does not stall the blocks. thermal_char measured the 1 s/5 s readers, but without coresidency's PSS sampler.
- That `thermal_char_run_20260930T233208Z` would now be complete: its only INCOMPLETE line was the cached pid. Not re-reported.
- Force-stopping the app between blocks means each block starts RobotCam from a cold app process. Earlier runs started blocks from a cached one. This follows from item 6 as specified.
- The three open round-3 findings above.

## Review rounds
| Round | Verdict | Findings → action |
|---|---|---|
| 1 | REQUEST CHANGES | B: unreadable reads counted as "capture stopped". **Fixed** (only `missing`/same frame is evidence; force-stop still attempted). B: capped time counted pre-block samples, used sample count, treated unreadable freqs as uncapped. **Fixed** (in-block, time-weighted, % of duration, unreadable → INCOMPLETE). B: CoresLost/abort skipped the end check. **Fixed** (in `finally`, plus main cleanup). NB: SMOKE.md prose. Not changed (task rule). |
| 2 | REQUEST CHANGES | B: B5's end check only in final cleanup, after the report. **Fixed** (recorded in B5 JSON, INCOMPLETE on failure). B: a raising STOP broadcast skipped force-stop. **Fixed** (`try: camera_stop() finally: camera_end_check()`), with a new failed-broadcast run test. |
| 3 | **REQUEST CHANGES** | B: B5 has no cooldown gate. B: a pid after force-stop is accepted (the task text says record only). B: capped-time attribution across skipped samples. NB: SMOKE.md. **Not fixed**: round limit reached. |

Reviewer sandbox: some of the reviewer's own shell commands exited with 182: 7, 5 and 10 per round, next to 7, 4 and 8 that succeeded. This is the same sandbox behaviour as in the previous tasks. There were no authentication, quota or model errors; stderr shows `model: gpt-6-sol`, `reasoning effort: medium`, `sandbox: read-only` each round. Every request contained the task, the frozen and candidate hashes, the full diff, the four complete changed files (numbered) and the check output. Requests, stdout and stderr: `~/storage/downloads/cleanup1_review_artifacts/round{1,2,3}/`.

## Final review (round 3), verbatim
```
**Verdict: REQUEST CHANGES.** The supplied checks pass, but the frozen candidate has three blocking gaps.

- **BLOCKING — B5 skips the required cooldown gate.** [mtp_snapshot()](/termux-home/robot/benchmark/coresidency/coresidency.py:878) starts RobotCam and Gemma without calling `thermal_gate()`. B5 is listed as a block, so it can run warm without a recorded wait or warm-start mark.

- **BLOCKING — a process left after force-stop is accepted.** [camera_end_failed()](/termux-home/robot/benchmark/coresidency/coresidency.py:183) checks the command’s return code but ignores `pids_after_force_stop`. The test explicitly accepts a remaining PID. That can report cleanup as successful while the app process is still present.

- **BLOCKING — capped seconds can be reported incorrectly as valid.** [block_summary()](/termux-home/robot/benchmark/coresidency/coresidency.py:956) assigns the interval *before* each sample to that sample’s frequency cap and omits the interval after the last sample. Across a skipped sample, a cap observed at 1 s and an uncapped reading at 5 s is reported as only 1 capped second, with no incomplete flag.

- **NON-BLOCKING —** [SMOKE.md](/termux-home/robot/benchmark/coresidency/SMOKE.md) retains stale gate and incomplete-result prose.
```

## Diff (frozen candidate → final)
```diff
diff -ru coresidency/coresidency.py ../cur/coresidency/coresidency.py
--- coresidency/coresidency.py	2026-09-30 16:58:21.611251853 +0000
+++ ../cur/coresidency/coresidency.py	2026-10-01 01:23:16.904767741 +0000
@@ -3,7 +3,8 @@
 opens USB/serial. Native Termux only (the robot runs ONNX Runtime and llama-server natively);
 started by oneshot.sh in this folder, which runs the thermal logger and answers cache-drop requests.
 
-Blocks, in order (180 s each; --smoke: 20 s, no cooldown gate):
+Blocks, in order (180 s each; --smoke: 20 s, no cooldown gate). Before each block and the loads, the cooldown gate
+waits for VIRTUAL-SKIN <= idle + 1.5 degC and zone9 <= idle + 4 degC; after 15 min the block starts as a warm start:
   B1_mix_nollm      RobotCam mode B rate 2 -> read_frame (fixed session, frame must advance) ->
                     Detector at SizePolicy drive mode sizes (320 each frame, 640 every 5 s)
   B2_only640_nollm  same, always 640
@@ -12,11 +13,17 @@
                     letter-scoring call every 20 s in its own thread (s1o, variant s1o_b1609dp_q40 prompts)
   B4_only640_gemma  B2 + the same
   B5_mtp_ram_snapshot  (not timed) Gemma with the conv_mtp drafter flags, one short request, RAM snapshot
-B1-B4 end early at the robot's live thermal pause (zone and threshold read from main.py): a result, not a failure.
+B1-B4 end early at the first block limit: Android thermal status >= CRITICAL, battery >= 45.0 degC, or a CPU zone
+(BIG/MID/LITTLE) >= 110 degC in 3 consecutive 1 s samples (fault stop): a result, not a failure. Fail closed (block
+stopped, INCOMPLETE): no VIRTUAL-SKIN + status reading for 60 s, no CPU zone or battery reading for 5 s. A limit met
+before the first frame: "not run: limit at start", INCOMPLETE.
 
 Every 5 s: MemAvailable/swap, PSS (root dumpsys meminfo) of the runner, llama-server, RobotCam app and
 camera provider, zone9/10/11 from the thermal log, battery power (camera_heat.py's method). Per block:
 LMK log lines, RobotCam/llama-server survival. Per read: status, read/decode/detect ms, size, frame age.
+Per block, every 1 s (one persistent root shell): CPU zones, battery temp, scaling_max_freq per cpufreq policy;
+every 5 s: Android thermal status and VIRTUAL-SKIN from `dumpsys thermalservice` (DECISIONS #123 form). After each
+block's STOP: capture must stop (no new frame for 3 s), then `am force-stop` the app (it stays cached after STOP).
 
 Usage: coresidency.py [--smoke] [--resume RUN_DIR] [--thermal-log PATH]
 """
@@ -26,6 +33,7 @@
 import json
 import math
 import os
+import queue
 import re
 import shutil
 import signal
@@ -66,14 +74,14 @@
 BLOCKS = [('B1_mix_nollm', 'mix', False), ('B2_only640_nollm', '640', False),
           ('B3_mix_gemma', 'mix', True), ('B4_only640_gemma', '640', True)]
 LARGE_S = 5  # main.LARGE_FRAME_INTERVAL_S: drive mode runs one 640 frame every 5 s
-# Heat stop: the robot's live thermal pause, main.Robot.run_mission `if get_temp() > 80:` with get_temp() reading
-# thermal_zone9 in whole degC (millidegC // 1000). Both values are read from main.py, never restated here.
-_MAIN = (ROBOT / 'main.py').read_text()
-_zone = re.search(r'def get_temp\(\):\n.*?thermal_zone(\d+)/temp', _MAIN, re.S)
-_limit = re.search(r'if get_temp\(\) > (\d+):', _MAIN)
-if not (_zone and _limit) or _zone.group(1) not in ('9', '10', '11'):
-    raise SystemExit('main.py: the live thermal pause (get_temp zone, get_temp() > N) was not found')
-PAUSE_ZONE, PAUSE_ABOVE_C = f'z{_zone.group(1)}', int(_limit.group(1))
+# Block limits. zone9 is no heat signal (capped near 100 degC within seconds, then held at 62-68 degC while the
+# phone heats); Android throttles on VIRTUAL-SKIN: status LIGHT 39, MODERATE 43, SEVERE 45, CRITICAL 46.5 (hard caps).
+THERMAL, CPUFREQ = '/sys/class/thermal', '/sys/devices/system/cpu/cpufreq'
+SKIN, CPU_TYPES = 'VIRTUAL-SKIN', ('BIG', 'MID', 'LITTLE')
+STATUS_NAMES = ['NONE', 'LIGHT', 'MODERATE', 'SEVERE', 'CRITICAL', 'EMERGENCY', 'SHUTDOWN']
+STATUS_STOP, BATTERY_STOP_C, CPU_FAULT_C, CPU_N = 4, 45.0, 110.0, 3
+FAST_S, DUMP_S, SKIN_WAIT_S, SENSOR_WAIT_S = 1, 5, 60, 5
+SKIN_GATE_C, GATE_MAX_S, GATE_POLL_S = 1.5, 15 * 60, 10
 DEFAULT_INSTRUCTION = 'Choose the single best next action for the robot.'  # ladder.py
 WARMUP = ('The robot is idle in the hallway.', {'wait': '', 'explore': ''}, 'Pick one.')  # ladder_worker.py
 MTP_PROMPT = 'In one short sentence, what does a home robot do?'
@@ -144,6 +152,45 @@
     raise RuntimeError('RobotCam did not publish a usable frame in 3 attempts')
 
 
+def camera_end_check(rootf=None, quiet_s=3, limit_s=15):
+    """After the STOP broadcast: capture has stopped once every read_frame for quiet_s s (up to limit_s) shows
+    evidence of no new frame: 'missing' (STOP deletes the files; an old frame reads as missing) or the same
+    (session, frame) again. A new frame or an unreadable one ('bad': no evidence) restarts the window. Then
+    `am force-stop` and pidof, also when capture did not stop, in the DECISIONS #123 form (rootf: coresidency.root
+    or thermal_char.root_file). The app process stays cached after STOP, so a pid before the force-stop is no failure."""
+    rootf = rootf or root
+    began = quiet_from = time.monotonic()
+    last, seen, unreadable = None, [], 0
+    while time.monotonic() - quiet_from < quiet_s and time.monotonic() - began < limit_s:
+        r = read_frame(FRAME_DIR)
+        if r['status'] == 'ok' and (r['session'], r['frame']) != last:
+            last, quiet_from = (r['session'], r['frame']), time.monotonic()
+            seen.append({'session': r['session'], 'frame': r['frame'], 't': round(quiet_from - began, 2)})
+        elif r['status'] not in ('ok', 'missing'):
+            unreadable, quiet_from = unreadable + 1, time.monotonic()
+        time.sleep(0.1)
+    out = {'capture_stopped': time.monotonic() - quiet_from >= quiet_s, 'frames_seen': seen,
+           'unreadable_reads': unreadable, 'check_s': round(time.monotonic() - began, 2)}
+    try:
+        rc, text = rootf('am force-stop com.pixelrobot.robotcam', 'forcestop')
+        out.update(force_stop_rc=rc, force_stop_output=text.strip()[:200])
+        out['pids_after_force_stop'] = rootf('pidof com.pixelrobot.robotcam', 'pidof')[1].split()
+    except (OSError, subprocess.SubprocessError) as e:
+        out.update(force_stop_rc=None, force_stop_output=f'{type(e).__name__}: {e}')
+    return out
+
+
+def camera_end_failed(c):
+    """Why a camera_end_check result fails the run (capture kept advancing, force-stop failed), else None."""
+    why = []
+    if not c.get('capture_stopped'):
+        why.append(f'capture not shown stopped {c.get("check_s")} s after STOP ({len(c.get("frames_seen", []))} '
+                   f'new frames, {c.get("unreadable_reads")} unreadable reads)')
+    if c.get('force_stop_rc') != 0:
+        why.append(f'am force-stop failed (rc {c.get("force_stop_rc")}: {c.get("force_stop_output")!r})')
+    return '; '.join(why) or None
+
+
 def meminfo_mib():
     m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
     return {'mem_available_mib': m['MemAvailable'] // 1024, 'swap_used_mib': (m['SwapTotal'] - m['SwapFree']) // 1024,
@@ -227,15 +274,19 @@
 
 
 def thermal_gate(path, idle, label, smoke):
-    """ladder.thermal_gate: wait until z9 <= idle + 4 degC; --smoke skips the wait."""
+    """Cooldown gate: wait until VIRTUAL-SKIN <= idle skin + 1.5 degC and z9 <= idle + 4 degC (ladder.thermal_gate);
+    not reached in 15 min: start anyway, marked warm_start. --smoke skips the wait."""
     began = time.monotonic()
-    while not smoke and (t := read_thermal(path))['z9'] > idle['z9'] + GATE_MC:
-        print(f'  [{label}] waiting: z9 {t["z9"] / 1000:.1f} degC, need <= {(idle["z9"] + GATE_MC) / 1000:.1f}, '
-              f'{time.monotonic() - began:.0f} s', flush=True)
-        time.sleep(10)
-    t = read_thermal(path)
-    t['waited_s'] = round(time.monotonic() - began)
-    return t
+    while True:
+        t, d = read_thermal(path), read_dump()
+        cool = t['z9'] <= idle['z9'] + GATE_MC and d['skin'] is not None and d['skin'] <= idle['skin'] + SKIN_GATE_C
+        waited = time.monotonic() - began
+        if smoke or cool or waited >= GATE_MAX_S:
+            break
+        print(f'  [{label}] waiting: skin {d["skin"]} degC, need <= {idle["skin"] + SKIN_GATE_C:.1f}; z9 '
+              f'{t["z9"] / 1000:.1f} degC, need <= {(idle["z9"] + GATE_MC) / 1000:.1f}; {waited:.0f} s', flush=True)
+        time.sleep(GATE_POLL_S)
+    return {**t, 'skin': d['skin'], 'status': d['status'], 'waited_s': round(waited), 'warm_start': not (smoke or cool)}
 
 
 class CoresLost(Exception):
@@ -496,26 +547,188 @@
 robotcam_reader.io = _Stamp
 
 
-def heat_reached(thermal_log):
-    """The live pause test on the newest thermal-log line: the reading if it is reached, else None. A line that
-    cannot be parsed is retried next frame (the sampler records it as a thermal error); a stale log still stops."""
+# ---------------------------------------------------------------- block limit readings
+# Copied from ../thermal_char/thermal_char.py (it imports this file, so no import back): RootShell and TEMP_RE/
+# parse_dump verbatim (END marker renamed), to_int; discover and the 1 s sample script cut down to the limit inputs.
+
+def to_int(v):
     try:
-        t = read_thermal(thermal_log)
-    except (OSError, ValueError, IndexError, KeyError):
+        return int(v)
+    except (TypeError, ValueError):
         return None
-    return t if t[PAUSE_ZONE] // 1000 > PAUSE_ABOVE_C else None
 
 
-def frame_loop(detector, mode, session, t0, duration, reads, thermal_log):
-    """Reads every new frame and detects on it until the block ends or the heat stop is reached.
-    Returns (last frame number processed, heat stop record or None)."""
+class RootShell:
+    """One persistent root shell for the sysfs reads (no new su per sample). Android service calls do not go
+    through it: they use root (DECISIONS #123)."""
+    END = '__coresidency_end_'
+
+    def __init__(self):
+        self.p = subprocess.Popen(['su'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=DEVNULL, text=True,
+                                  bufsize=1, start_new_session=True)
+        self.lines, self.n = queue.Queue(), 0
+        threading.Thread(target=self._pump, daemon=True).start()
+
+    def _pump(self):
+        for line in self.p.stdout:
+            self.lines.put(line.rstrip('\n'))
+        self.lines.put(None)
+
+    def run(self, script, timeout=5):
+        """Output lines of script; raises RuntimeError on timeout or a dead shell."""
+        self.n += 1
+        end = f'{self.END}{self.n}__'
+        try:
+            self.p.stdin.write(f'{script}\necho {end}\n')
+            self.p.stdin.flush()
+        except OSError as e:
+            raise RuntimeError(f'root shell: {e}') from e
+        out, deadline = [], time.monotonic() + timeout
+        while True:
+            try:
+                line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
+            except queue.Empty:
+                raise RuntimeError(f'root shell: no answer in {timeout} s') from None
+            if line is None:
+                raise RuntimeError(f'root shell exited (rc {self.p.poll()})')
+            if line == end:
+                return out
+            if line.startswith(self.END):  # the end of an earlier command that timed out: its output is not ours
+                out = []
+                continue
+            out.append(line)
+
+    def close(self):
+        try:
+            self.p.stdin.close()
+        except OSError:
+            pass
+        try:
+            self.p.wait(timeout=5)
+        except subprocess.TimeoutExpired:
+            self.p.kill()
+            self.p.wait()
+
+
+def discover(shell):
+    """The BIG/MID/LITTLE zone numbers (lowest zone of each type) and every cpufreq policy's cpuinfo_max_freq."""
+    script = (f'for f in {THERMAL}/thermal_zone*/type {CPUFREQ}/policy*/cpuinfo_max_freq; do [ -e "$f" ] || continue; '
+              "v=; read -r v 2>/dev/null <\"$f\"; printf '%s\\t%s\\n' \"$f\" \"$v\"; done")
+    zones, policies = {}, {}
+    for line in shell.run(script, timeout=30):
+        path, _, v = line.partition('\t')
+        if m := re.fullmatch(re.escape(THERMAL) + r'/thermal_zone(\d+)/type', path):
+            zones[int(m.group(1))] = v
+        elif m := re.fullmatch(re.escape(CPUFREQ) + r'/policy(\d+)/cpuinfo_max_freq', path):
+            policies[int(m.group(1))] = to_int(v)
+    cpu = {t: next((str(i) for i in sorted(zones) if zones[i] == t), None) for t in CPU_TYPES}
+    bad = [f'policy{p}' for p, v in policies.items() if not v]
+    if None in cpu.values() or not policies or bad:
+        raise SystemExit(f'layout: CPU zones {cpu}, policies {sorted(policies)}, without cpuinfo_max_freq {bad}; '
+                         'not started')
+    return {'cpu_zones': cpu, 'policies': {f'policy{p}': policies[p] for p in sorted(policies)}}
+
+
+def fast_keys(layout):
+    return ([('cpu', t, f'{THERMAL}/thermal_zone{i}/temp') for t, i in layout['cpu_zones'].items()] +
+            [('bat', 'temp', f'{BATTERY}/temp')] +
+            [('max', p, f'{CPUFREQ}/{p}/scaling_max_freq') for p in layout['policies']])
+
+
+def fast_sample(shell, keys):
+    """One 1 s reading (shell builtins only; an unreadable file gives an empty line). t = monotonic s when in hand."""
+    start = time.monotonic()
+    try:
+        lines = shell.run(f'for f in {" ".join(k[2] for k in keys)}; do v=; read -r v 2>/dev/null <"$f"; echo "$v"; done')
+        if len(lines) != len(keys):
+            raise RuntimeError(f'expected {len(keys)} values, got {len(lines)}: {lines[:5]!r}')
+        err = None
+    except RuntimeError as e:
+        lines, err = [''] * len(keys), str(e)
+    s = {'t': time.monotonic(), 't_start': start, 'cpu': {}, 'max': {}, **({'error': err} if err else {})}
+    for (kind, name, _), v in zip(keys, lines):
+        if kind == 'bat':
+            s['bat_c'] = None if to_int(v) is None else to_int(v) / 10  # power_supply temp is in tenths of degC
+        else:
+            s[kind][name] = to_int(v)
+    s['cpu_c'] = None if None in s['cpu'].values() else max(s['cpu'].values()) / 1000  # the fault stop's input
+    return s
+
+
+TEMP_RE = re.compile(r'Temperature\{mValue=([^,]+), mType=(-?\d+), mName=([^,]+), mStatus=(\d+)\}')
+
+
+def parse_dump(text):
+    """Android thermal status and the 'Current temperatures from HAL' section (name -> degC)."""
+    status = re.search(r'^Thermal Status: (\d+)\s*$', text, re.M)
+    sections = {}
+    for m in re.finditer(r'^(\S[^\n]*):\n((?:[ \t]+[^\n]*\n?)*)', text, re.M):
+        sections[m.group(1).strip()] = m.group(2)
+    hal = {}
+    for m in TEMP_RE.finditer(sections.get('Current temperatures from HAL', '')):
+        try:
+            v = float(m.group(1))
+        except ValueError:
+            continue
+        if not math.isnan(v):
+            hal[m.group(3)] = v
+    cached = sorted({m.group(3) for m in TEMP_RE.finditer(sections.get('Cached temperatures', ''))})
+    return {'status': int(status.group(1)) if status else None, 'skin': hal.get(SKIN), 'hal': hal,
+            'cached_names': cached}
+
+
+def read_dump():
+    """`dumpsys thermalservice` (DECISIONS #123 form): status and VIRTUAL-SKIN, None when absent; t = monotonic s
+    when the reading was in hand (the dump ran t_start..t)."""
+    start = time.monotonic()
+    try:
+        rc, text = root('dumpsys thermalservice', 'thermalservice', timeout=30)
+        d = {'rc': rc, **{k: v for k, v in parse_dump(text).items() if k in ('status', 'skin')}}
+    except (OSError, subprocess.SubprocessError) as e:
+        d = {'rc': None, 'error': f'{type(e).__name__}: {e}', 'status': None, 'skin': None}
+    return {'t': time.monotonic(), 't_start': start, **d}
+
+
+def monitor_loop(period, read, rows, stop):
+    k, began = 0, time.monotonic()
+    while not stop.wait(max(0.0, began + k * period - time.monotonic())):
+        rows.append(read())
+        k = max(k + 1, math.ceil((time.monotonic() - began) / period))  # an overrun skips slots, never bunches
+
+
+def block_limit(now, began, fast, dumps):
+    """The block limit met at `now` as (limit, text), else None; times are monotonic s, began = monitor start.
+    Each limit from its own readings (thermal_char.stop_reasons). Pure: tested offline."""
+    status = next((d for d in reversed(dumps) if d['status'] is not None), None)
+    skin = next((d for d in reversed(dumps) if d['skin'] is not None), None)
+    if status and status['status'] >= STATUS_STOP:
+        return 'android_status', f'Android thermal status {status["status"]} >= {STATUS_STOP} ({STATUS_NAMES[STATUS_STOP]})'
+    bat = [s for s in fast[-SENSOR_WAIT_S * 4:] if s['bat_c'] is not None]
+    if bat and bat[-1]['bat_c'] >= BATTERY_STOP_C:
+        return 'battery', f'battery {bat[-1]["bat_c"]:.1f} >= {BATTERY_STOP_C} degC'
+    cpu = [s for s in fast[-SENSOR_WAIT_S * 4:] if s['cpu_c'] is not None]
+    if len(cpu) >= CPU_N and all(s['cpu_c'] >= CPU_FAULT_C for s in cpu[-CPU_N:]):
+        return 'cpu_fault', (f'CPU zone >= {CPU_FAULT_C:.0f} degC in {CPU_N} consecutive 1 s samples '
+                             f'({[s["cpu_c"] for s in cpu[-CPU_N:]]})')
+    fresh = min(skin['t'] if skin else -math.inf, status['t'] if status else -math.inf)
+    if now - max(fresh, began) > SKIN_WAIT_S:
+        return 'fail_closed', f'no {SKIN} and Android status reading for {SKIN_WAIT_S} s'
+    for name, rows in (('CPU zone', cpu), ('battery temperature', bat)):
+        if now - max(rows[-1]['t'] if rows else -math.inf, began) > SENSOR_WAIT_S:
+            return 'fail_closed', f'no {name} reading for {SENSOR_WAIT_S} s'
+    return None
+
+
+def frame_loop(detector, mode, session, t0, duration, reads, limit):
+    """Reads every new frame and detects on it until the block ends or limit() returns a stop record.
+    Returns (last frame number processed, stop record or None)."""
     policy = SizePolicy(LARGE_S, 1/3)  # main.LARGE_FRAME_INTERVAL_S, PERSON_HEIGHT_FRACTION; drive mode
     last = None
     while time.monotonic() < t0 + duration:
-        # ponytail: checked between frames on the 5 s thermal log, so the stop (time_to_limit_s = when the runner
-        # saw it) can come up to ~5 s plus one detection after the zone crossed; a live su read would cut that
-        if (hot := heat_reached(thermal_log)) is not None:
-            return last, {'time_to_limit_s': round(time.monotonic() - t0, 2), 'zone_reading': hot}
+        # ponytail: checked between frames, so a stop comes up to one detection (~1.5 s) plus the 1 s / 5 s
+        # reading interval after the sensor crossed; a watcher thread could cut the frame part
+        if (hit := limit()) is not None:
+            return last, hit
         check_cores('during the block')
         _Stamp.at = None
         a = time.perf_counter()
@@ -545,49 +758,85 @@
 
 # ---------------------------------------------------------------- blocks
 
+def latest(fast, dumps):
+    """The newest value of each limit input (the readings recorded with a stop)."""
+    pick = lambda rows, k: next((r[k] for r in reversed(rows) if r.get(k) is not None), None)
+    return {'skin': pick(dumps, 'skin'), 'status': pick(dumps, 'status'), 'bat_c': pick(fast, 'bat_c'),
+            'cpu_c': pick(fast, 'cpu_c'), 'scaling_max': next((r['max'] for r in reversed(fast)), None)}
+
+
 def run_block(name, mode, gemma, ctx):
     """One block; raises CoresLost if cores 4-7 go away (the caller discards it and redoes it)."""
     check_cores('before ' + name)
     therm_start = thermal_gate(ctx['thermal_log'], ctx['idle'], name, ctx['smoke'])
-    print(f'[{name}] start: z9 {therm_start["z9"] / 1000:.1f} degC (waited {therm_start["waited_s"]} s)', flush=True)
+    print(f'[{name}] start: skin {therm_start["skin"]} degC, z9 {therm_start["z9"] / 1000:.1f} degC (waited '
+          f'{therm_start["waited_s"]} s{", WARM START" if therm_start["warm_start"] else ""})', flush=True)
     server = ctx['server'] if gemma else None
     if gemma and not server.alive():
         raise RuntimeError('llama-server is not healthy at block start')
-    lmk_since = time.time()
-    cam = camera_start()
-    t0 = time.monotonic()
-    duration = ctx['duration']
-    stop, samples, calls, reads = threading.Event(), [], [], []
-    threads = [threading.Thread(target=sampler_loop, args=(t0, ctx['thermal_log'], server, stop, samples), daemon=True)]
-    if gemma:
-        threads.append(threading.Thread(target=selector_loop, args=(ctx['selector'], ctx['cases'], t0, duration,
-                                                                    stop, calls), daemon=True))
-    for t in threads:
+    stop_mon, fast, dumps, keys = threading.Event(), [], [], fast_keys(ctx['layout'])
+    monitors = [threading.Thread(target=monitor_loop, args=(FAST_S, lambda: fast_sample(ctx['shell'], keys), fast,
+                                                            stop_mon), daemon=True),
+                threading.Thread(target=monitor_loop, args=(DUMP_S, read_dump, dumps, stop_mon), daemon=True)]
+    for t in monitors:
         t.start()
+    began = time.monotonic()
+    threads = []
     try:
-        last, heat = frame_loop(ctx['detector'], mode, cam['session'], t0, duration, reads, ctx['thermal_log'])
-        elapsed = time.monotonic() - t0
-        stop.set()  # a heat stop ends the selector's cadence here too (no slot starts after the block)
-        deadline = time.monotonic() + 2  # survival: a newer frame within 2 s (rate 2)
-        while True:
-            end = read_frame(FRAME_DIR, session=cam['session'])
-            if (end['status'] == 'ok' and last is not None and end['frame'] > last) or time.monotonic() > deadline:
-                break
-            time.sleep(0.1)
-        robotcam_pid = robotcam_pids()
-    finally:
-        stop.set()
+        while not (fast and dumps) and time.monotonic() - began < 10:  # first readings in before the first frame
+            time.sleep(0.05)
+        lmk_since = time.time()
+        cam = camera_start()
+        t0 = time.monotonic()
+        duration = ctx['duration']
+
+        def limit():
+            now = time.monotonic()
+            hit = block_limit(now, began, fast, dumps)
+            return hit and {'limit': hit[0], 'reason': hit[1], 'time_to_limit_s': round(now - t0, 2),
+                            'reading': latest(fast, dumps)}
+        stop, samples, calls, reads = threading.Event(), [], [], []
+        threads = [threading.Thread(target=sampler_loop, args=(t0, ctx['thermal_log'], server, stop, samples),
+                                    daemon=True)]
+        if gemma:
+            threads.append(threading.Thread(target=selector_loop, args=(ctx['selector'], ctx['cases'], t0, duration,
+                                                                        stop, calls), daemon=True))
         for t in threads:
-            t.join(timeout=120)
-        camera_stop()
-    if any(t.is_alive() for t in threads):  # e.g. a stalled selector request: it would overlap the next block
-        raise RuntimeError(f'{name}: a sampler/selector thread still running 120 s after the block; run stopped')
+            t.start()
+        try:
+            last, hit = frame_loop(ctx['detector'], mode, cam['session'], t0, duration, reads, limit)
+            elapsed = time.monotonic() - t0
+            stop.set()  # a limit stop ends the selector's cadence here too (no slot starts after the block)
+            deadline = time.monotonic() + 2  # survival: a newer frame within 2 s (rate 2)
+            while True:
+                end = read_frame(FRAME_DIR, session=cam['session'])
+                if (end['status'] == 'ok' and last is not None and end['frame'] > last) or time.monotonic() > deadline:
+                    break
+                time.sleep(0.1)
+            robotcam_pid = robotcam_pids()
+        finally:
+            stop.set()
+            for t in threads:
+                t.join(timeout=120)
+            try:
+                camera_stop()
+            finally:  # also on CoresLost/abort or a failed broadcast: the next block or run starts from a stopped app
+                cam_end = camera_end_check()
+    finally:
+        stop_mon.set()
+        for t in monitors:
+            t.join(timeout=60)
+    if any(t.is_alive() for t in threads + monitors):  # e.g. a stalled selector request: it would overlap the next block
+        raise RuntimeError(f'{name}: a sampler/selector/monitor thread still running after the block; run stopped')
     check_cores('after ' + name)
     therm_end = read_thermal(ctx['thermal_log'])
+    for r in fast + dumps:  # monotonic -> s from the block start (readings before the camera start are negative)
+        r['t'], r['t_start'] = round(r['t'] - t0, 3), round(r['t_start'] - t0, 3)
     return {'block': name, 'mode': mode, 'gemma': gemma, 'duration_s': round(elapsed, 2), 'planned_s': duration,
-            'heat_stop': {'zone': PAUSE_ZONE, 'above_c': PAUSE_ABOVE_C, 'reached_limit': heat is not None,
-                          **(heat or {'time_to_limit_s': None, 'zone_reading': None})}, 'camera': cam,
-            'thermal_start': therm_start, 'thermal_end': therm_end,
+            'heat_stop': {'reached_limit': bool(hit) and hit['limit'] != 'fail_closed',
+                          **(hit or {'limit': None, 'reason': None, 'time_to_limit_s': None, 'reading': None})},
+            'camera': cam, 'camera_end': cam_end, 'thermal_start': therm_start, 'thermal_end': therm_end,
+            'cpuinfo_max_khz': ctx['layout']['policies'], 'fast': fast, 'dumps': dumps,
             'survived': {'robotcam_process': bool(robotcam_pid), 'robotcam_pids': robotcam_pid,
                          'robotcam_new_frame_at_end': end['status'] == 'ok' and last is not None and end['frame'] > last,
                          'robotcam_end_status': end['status'],
@@ -640,13 +889,17 @@
             raise RuntimeError(f'B5: no usable RobotCam frame for the 640 detection ({r["status"]})')
         ctx['detector'].detect(r['image'], 640)
         snap = sample(time.monotonic(), ctx['thermal_log'], server)
-        return {'block': 'B5_mtp_ram_snapshot', 'cmd': server.cmd, 'load_s': server.load_s,
-                'reply': res.get('content'), 'timings': res.get('timings'), 'detected_640_on_frame': r['frame'],
-                'server_status_kb': server.status_kb(), 'sample': snap}
+        rec = {'block': 'B5_mtp_ram_snapshot', 'cmd': server.cmd, 'load_s': server.load_s,
+               'reply': res.get('content'), 'timings': res.get('timings'), 'detected_640_on_frame': r['frame'],
+               'server_status_kb': server.status_kb(), 'sample': snap}
     finally:
         if server:
             server.stop()
-        camera_stop()
+        try:
+            camera_stop()
+        finally:  # also when the broadcast failed: the force-stop is the fallback
+            cam_end = camera_end_check()
+    return {**rec, 'camera_end': cam_end}
 
 
 # ---------------------------------------------------------------- report
@@ -693,9 +946,26 @@
     s['max_swap'] = max((x['swap_used_mib'] for x in sm), default=None)
     for name in ('runner', 'llama_server', 'robotcam_app', 'camera_provider'):
         s['pss_' + name] = max((x['pss_kb'][name] for x in sm if name in x.get('pss_kb', {})), default=None)
-    z9 = [b['thermal_start']['z9'], b['thermal_end']['z9']] + [x['z9'] for x in sm if 'z9' in x] + \
-         ([b['heat_stop']['zone_reading']['z9']] if b['heat_stop']['reached_limit'] else [])  # the stop reading
+    z9 = [b['thermal_start']['z9'], b['thermal_end']['z9']] + [x['z9'] for x in sm if 'z9' in x]
     s['z9'] = (b['thermal_start']['z9'] / 1000, b['thermal_end']['z9'] / 1000, max(z9) / 1000)
+    # limit readings up to the block end (those taken before the camera start included: the start values)
+    skin = [d['skin'] for d in b['dumps'] if d['skin'] is not None and d['t'] <= b['duration_s']]
+    s['skin'] = (skin[0], skin[-1], max(skin)) if skin else (None, None, None)
+    s['status_max'] = max((d['status'] for d in b['dumps'] if d['status'] is not None and d['t'] <= b['duration_s']),
+                          default=None)
+    # capped time: each in-block 1 s sample stands for the time since the previous one (a skipped slot is counted);
+    # a sample with an unreadable scaling_max_freq is no evidence either way: counted apart, and INCOMPLETE
+    fast = [x for x in b['fast'] if 0 <= x['t'] <= b['duration_s']]
+    top = b['cpuinfo_max_khz']
+    capped = unread = 0.0
+    for prev, x in zip([0.0] + [x['t'] for x in fast], fast):
+        if any(x['max'].get(p) is None for p in top):
+            unread += x['t'] - prev
+        elif any(x['max'][p] < top[p] for p in top):
+            capped += x['t'] - prev
+    s['capped'] = (capped, 100 * capped / b['duration_s'] if b['duration_s'] else None)
+    s['max_unread'] = (sum(any(x['max'].get(p) is None for p in top) for x in fast), unread)
+    s['low_max'] = [min((x['max'][p] for x in fast if x['max'].get(p) is not None), default=None) for p in top]
     w = [x['battery_w'] for x in sm if 'battery_w' in x]
     s['w'] = statistics.mean(w) if w else None
     s['bat_status'] = sorted({x['battery_status'] for x in sm if 'battery_status' in x})
@@ -712,36 +982,46 @@
     out = []
     for b, s in zip(blocks, sums):
         heat = b['heat_stop']
-        # a heat stop is a result: rules that need time only apply as far as the block ran (at_start: stopped
-        # before its first frame; no_sample: before a second sample was due, so the first may finish after it)
-        at_start = heat['reached_limit'] and s['frames'] == 0 and not s['failed']
+        if why := camera_end_failed(b['camera_end']):
+            out.append(f'{b["block"]}: RobotCam end check: {why}')
+        if heat['reached_limit'] and s['frames'] == 0 and not s['failed']:  # met before the first frame
+            r = heat['reading']
+            out.append(f'{b["block"]}: not run: limit at start ({heat["reason"]}; skin {r["skin"]}, status '
+                       f'{r["status"]}, battery {r["bat_c"]}, CPU max {r["cpu_c"]} degC, scaling_max {r["scaling_max"]})')
+            continue
+        if heat['limit'] == 'fail_closed':
+            out.append(f'{b["block"]}: stopped fail-closed at {heat["time_to_limit_s"]} s: {heat["reason"]}')
+        # a limit stop is a result: rules that need time only apply as far as the block ran
+        # (no_sample: stopped before a second sample was due, so the first may finish after it)
         no_sample = heat['reached_limit'] and b['duration_s'] < SAMPLE_S
         if b['gemma']:  # the Gemma workload is one call per 20 s slot, each inside the block
             calls = b['selector_calls']
             # slots that can start inside the block: k * SELECTOR_S < planned length (selector_loop's bound; the
-            # block itself runs a little past it), or before the heat stop, which also stops the selector; a slot
-            # due in the last second before a heat stop is not required (the stop may beat the thread to it)
+            # block itself runs a little past it), or before the limit stop, which also stops the selector; a slot
+            # due in the last second before a limit stop is not required (the stop may beat the thread to it)
             end = heat['time_to_limit_s'] - 1 if heat['reached_limit'] else b['planned_s']
             slots = max(0, math.ceil(end / SELECTOR_S))
             late = [c['t'] for c in calls if c['started_s'] - c['t'] > 5]
-            # a call in flight at a heat stop ends after it by construction: not an overrun
+            # a call in flight at a limit stop ends after it by construction: not an overrun
             over = [] if heat['reached_limit'] else [c['t'] for c in calls if c['ended_s'] > b['duration_s']]
             if slots and s['sel'][0] == 0:
                 out.append(f'{b["block"]}: no successful selector call ({s["sel"][1]} failed)')
             elif s['sel'][0] < slots or late or over:
                 out.append(f'{b["block"]}: selector cadence missed: {s["sel"][0]}/{slots} successful calls, '
                            f'started >5 s late at slots {late}, ended after the block at slots {over}')
+        if s['max_unread'][0]:
+            out.append(f'{b["block"]}: {s["max_unread"][0]} 1 s sample(s) without every scaling_max_freq '
+                       f'({s["max_unread"][1]:.1f} s of the block not known capped or not)')
         if s['sample_errors'] or (s['min_avail'] is None and not no_sample):
             out.append(f'{b["block"]}: {s["sample_errors"]} sample(s) with root/PSS/thermal errors, '
                        f'{0 if s["min_avail"] is None else "some"} usable samples')
         sv = b['survived']
-        if s['failed'] or (s['frames'] == 0 and not at_start) or \
+        if s['failed'] or s['frames'] == 0 or \
                 not (sv['robotcam_new_frame_at_end'] and sv['robotcam_process']):
             out.append(f'{b["block"]}: RobotCam: {s["frames"]} frames, failed reads {s["failed"] or "none"}, '
                        f'new frame at end {sv["robotcam_new_frame_at_end"]}, process at end {sv["robotcam_process"]}')
         for size in (320, 640) if b['mode'] == 'mix' else (640,):
-            if at_start or (size == 640 and b['mode'] == 'mix' and heat['reached_limit'] and
-                            heat['time_to_limit_s'] < LARGE_S):
+            if size == 640 and b['mode'] == 'mix' and heat['reached_limit'] and heat['time_to_limit_s'] < LARGE_S:
                 continue  # stopped before this size's first frame was due
             if s[f'n{size}'] == 0 or (None in s[f'drift{size}'] and not s['drift_na']):
                 out.append(f'{b["block"]}: {s[f"n{size}"]} detections at {size}, drift first/last {DRIFT_S} s '
@@ -761,6 +1041,7 @@
     sums = [block_summary(b) for b in blocks]
     bad = problems(blocks, sums)
     mib = lambda kb: None if kb is None else kb / 1024
+    pols = '/'.join(p[6:] for p in blocks[0]['cpuinfo_max_khz']) if blocks else ''
     rows = [
         ('frames processed', lambda s: str(s['frames'])),
         ('failed reads by status', lambda s: ', '.join(f'{k} {v}' for k, v in s['failed'].items()) or 'none'),
@@ -782,7 +1063,12 @@
         ('LMK log lines (kill lines)', None),
         ('survived RobotCam / llama', None),
         ('zone9 start/end/max degC', lambda s: '/'.join(f'{v:.1f}' for v in s['z9'])),
+        ('VIRTUAL-SKIN start/end/max degC', lambda s: '/'.join(f(v, '.1f') for v in s['skin'])),
+        ('Android status max', lambda s: f(s['status_max'], 'd')),
+        ('policy capped s (% of block)', lambda s: f'{s["capped"][0]:.0f} ({f(s["capped"][1])}%)'),
+        (f'lowest scaling_max MHz {pols}', lambda s: '/'.join(f(v and v / 1000) for v in s['low_max'])),
         ('gate wait s', None),
+        ('stop limit', None),
         ('time to limit s', None),
         ('mean battery W', lambda s: f(s['w'], '.2f')),
         ('battery status', lambda s: ','.join(s['bat_status']) or 'n/a'),
@@ -794,12 +1080,15 @@
                                                       else f'QUERY FAILED rc {b["lmk"]["logcat_rc"]}'),
              'survived RobotCam / llama': lambda b: (f'{"yes" if b["survived"]["robotcam_new_frame_at_end"] and b["survived"]["robotcam_process"] else "NO"} / '
                                                      f'{ {True: "yes", False: "NO", None: "-"}[b["survived"]["llama_server"]]}'),
-             'gate wait s': lambda b: str(b['thermal_start']['waited_s']),
+             'gate wait s': lambda b: f'{b["thermal_start"]["waited_s"]}' + (' warm start' if b['thermal_start']['warm_start']
+                                                                              else ''),
+             'stop limit': lambda b: b['heat_stop']['limit'] or '-',
              'time to limit s': lambda b: (f'{b["heat_stop"]["time_to_limit_s"]:.1f}' if b['heat_stop']['reached_limit']
                                            else '-')}
     run = json.loads((out / 'run.json').read_text())
     lines = [f'Co-residency benchmark (DECISIONS #124), run {out.name}{"  [SMOKE: not a measurement]" if run["smoke"] else ""}',
-             f'block length {run["block_s"]} s; idle z9 {run["idle"]["z9"] / 1000:.1f} degC; runner sha256 {run["sha256"]["coresidency.py"][:12]}']
+             f'block length {run["block_s"]} s; idle z9 {run["idle"]["z9"] / 1000:.1f} degC, idle skin '
+             f'{run["idle"]["skin"]:.1f} degC; runner sha256 {run["sha256"]["coresidency.py"][:12]}']
     lines += [f'INCOMPLETE: {p}' for p in bad] + ['']
     width = 32
     lines.append(f'{"":{width}}' + ''.join(f'{b["block"]:>20}' for b in blocks))
@@ -809,7 +1098,8 @@
     if (out / 'loads.json').exists():
         ld = json.loads((out / 'loads.json').read_text())
         lines += ['', f'Gemma load (spawn to /health ok): cold {ld["cold_load_s"]:.2f} s [{ld["cold_state"]["mode"]}], '
-                      f'warm {ld["warm_load_s"]:.2f} s', 'server command: ' + ' '.join(ld['cmd'])]
+                      f'warm {ld["warm_load_s"]:.2f} s; gate wait {ld["thermal_start"]["waited_s"]} s'
+                      f'{" (warm start)" if ld["thermal_start"]["warm_start"] else ""}', 'server command: ' + ' '.join(ld['cmd'])]
     if (out / 'block_B5_mtp_ram_snapshot.json').exists():
         b5 = json.loads((out / 'block_B5_mtp_ram_snapshot.json').read_text())
         sm = b5['sample']
@@ -817,6 +1107,9 @@
         if errors:
             bad.append(f'B5_mtp_ram_snapshot: sample errors {errors}')
             lines.insert(2, f'INCOMPLETE: {bad[-1]}')
+        if why := camera_end_failed(b5['camera_end']):
+            bad.append(f'B5_mtp_ram_snapshot: RobotCam end check: {why}')
+            lines.insert(2, f'INCOMPLETE: {bad[-1]}')
         lines += ['', f'MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): '
                       f'MemAvailable {sm["mem_available_mib"]} MiB, swap used {sm["swap_used_mib"]} MiB, '
                       f'PSS llama-server {f(mib(sm.get("pss_kb", {}).get("llama_server")))} MiB, '
@@ -826,10 +1119,16 @@
                   'BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from '
                   'battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines '
                   'matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", '
-                  'or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Heat stop = the robot\'s '
-                  f'live pause ({PAUSE_ZONE} > {PAUSE_ABOVE_C} degC, main.Robot.run_mission), checked between frames '
-                  'on the 5 s thermal log; time to limit = when the runner saw it (up to ~5 s plus one detection late); '
-                  f'the block ends there and counts as run; drift is n/a if it ran under {2 * DRIFT_S} s.']
+                  'or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between '
+                  f'frames: Android thermal status >= {STATUS_STOP} ({STATUS_NAMES[STATUS_STOP]}), battery >= '
+                  f'{BATTERY_STOP_C} degC, CPU zone >= {CPU_FAULT_C:.0f} degC in {CPU_N} consecutive 1 s samples (fault '
+                  'stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading '
+                  'interval late); the block ends there and counts as run; drift is n/a if it ran under '
+                  f'{2 * DRIFT_S} s. Skin/status = `dumpsys thermalservice` every {DUMP_S} s; capped = time (each in-block 1 s sample '
+                  'counts the time since the previous one) with any policy\'s scaling_max_freq below its '
+                  'cpuinfo_max_freq, % of the block length. Gate: skin <= idle + '
+                  f'{SKIN_GATE_C} and z9 <= idle + {GATE_MC / 1000:.0f} degC; warm start = not reached in '
+                  f'{GATE_MAX_S // 60} min.']
     return '\n'.join(lines), bad
 
 
@@ -849,20 +1148,25 @@
         run = json.loads((out / 'run.json').read_text())
         if run['smoke'] != a.smoke:
             raise SystemExit(f'--resume: {out} was a {"smoke" if run["smoke"] else "full"} run')
-        if 'heat_stop' not in run:  # its blocks lack heat_stop/planned_s and were judged by the old cadence count
-            raise SystemExit(f'--resume: {out} was made by an older runner (no heat stop); start a new run')
+        if 'block_stops' not in run:  # its blocks lack the limit readings and were stopped on zone9
+            raise SystemExit(f'--resume: {out} was made by an older runner (no block limits); start a new run')
     else:
         out = OUT_ROOT / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}{"_smoke" if a.smoke else ""}'
         out.mkdir(parents=True)
     ctx = {'smoke': a.smoke, 'duration': SMOKE_S if a.smoke else BLOCK_S, 'thermal_log': a.thermal_log,
-           'server': None, 'selector': None, 'cases': load_cases()}
+           'shell': None, 'server': None, 'selector': None, 'cases': load_cases()}
     try:
         rc, who = root('id', 'id')
         if rc != 0:
             raise SystemExit(f'su failed: {who.strip()}')
         wait_cores('start')
-        ctx['idle'] = read_thermal(a.thermal_log)
-        print(f'thermal idle reading: z9 {ctx["idle"]["z9"] / 1000:.1f} degC; output {out}', flush=True)
+        ctx['shell'] = RootShell()
+        ctx['layout'] = discover(ctx['shell'])
+        ctx['idle'] = {**read_thermal(a.thermal_log), 'skin': read_dump()['skin']}  # after the launcher's idle
+        if ctx['idle']['skin'] is None:
+            raise SystemExit(f'no {SKIN} reading from dumpsys thermalservice at idle; not started')
+        print(f'thermal idle reading: z9 {ctx["idle"]["z9"] / 1000:.1f} degC, skin {ctx["idle"]["skin"]:.1f} degC; '
+              f'output {out}', flush=True)
         files = {'coresidency.py': __file__, 'robotcam_reader.py': robotcam_reader.__file__,
                  'detect_person.py': ROBOT / 'detect_person.py', 'detector_size_policy.py': ROBOT / 'detector_size_policy.py',
                  'server_manager.py': server_manager.__file__, 'main.py': ROBOT / 'main.py',
@@ -872,7 +1176,10 @@
         meta = {'started': utc(), 'smoke': a.smoke, 'block_s': ctx['duration'], 'idle': ctx['idle'],
                 'sha256': {k: sha256(p) for k, p in files.items()},
                 'model_bytes': {p: os.path.getsize(p) for p in (MODEL, DRAFT)},
-                'heat_stop': {'zone': PAUSE_ZONE, 'above_c': PAUSE_ABOVE_C},
+                'layout': ctx['layout'],
+                'block_stops': {'status': STATUS_STOP, 'battery_c': BATTERY_STOP_C, 'cpu_fault_c': CPU_FAULT_C,
+                                'cpu_consecutive': CPU_N, 'skin_wait_s': SKIN_WAIT_S, 'sensor_wait_s': SENSOR_WAIT_S,
+                                'gate_skin_c': SKIN_GATE_C, 'gate_z9_mc': GATE_MC, 'gate_max_s': GATE_MAX_S},
                 'server_cmd': server_cmd(), 'mtp_cmd': server_cmd(MTP_ARGS), 'python': sys.version,
                 'cpus_allowed': sorted(allowed_cpus())}
         if a.resume:
@@ -937,6 +1244,10 @@
             camera_stop()
         except (OSError, subprocess.SubprocessError) as e:
             print(f'[CAM] stop failed: {e}', file=sys.stderr)
+        if why := camera_end_failed(camera_end_check()):  # cleanup after an abort (each block and B5 check their own)
+            print(f'[CAM] end check: {why}', file=sys.stderr)
+        if ctx.get('shell'):
+            ctx['shell'].close()
 
 
 if __name__ == '__main__':
diff -ru coresidency/test_coresidency.py ../cur/coresidency/test_coresidency.py
--- coresidency/test_coresidency.py	2026-09-30 16:58:36.443251861 +0000
+++ ../cur/coresidency/test_coresidency.py	2026-10-01 01:27:02.472767851 +0000
@@ -8,6 +8,7 @@
 import json
 import os
 import re
+import shutil
 import signal
 import subprocess
 import sys
@@ -19,6 +20,38 @@
 
 HERE = Path(__file__).resolve().parent
 PORT = 18080
+POLICIES = {'policy0': 1803000, 'policy4': 2348000, 'policy6': 2850000}
+
+
+def dump_text(status=0, skin=35.0):
+    """`dumpsys thermalservice` as the phone prints it (thermal_char's test form, shortened)."""
+    hal = ''.join(f'\tTemperature{{mValue={v}, mType={t}, mName={n}, mStatus=0}}\n'
+                  for n, t, v in (('BIG', 0, 60.0), ('VIRTUAL-SKIN-CPU', -1, 36.0), ('VIRTUAL-SKIN', 3, skin)) if v is not None)
+    return (f'IsStatusOverride: false\nThermal Status: {status}\nCached temperatures:\n'
+            '\tTemperature{mValue=40.0, mType=3, mName=VIRTUAL-SKIN, mStatus=1}\n'
+            f'HAL Ready: true\nCurrent temperatures from HAL:\n{hal}'
+            'Current cooling devices from HAL:\n\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-2}\n')
+
+
+def make_sysfs(root_dir, cr):
+    """Fake su on PATH and a fake sysfs tree for the persistent root shell (zone 3 is a decoy: type CPU3)."""
+    b, th, cf, bat = (root_dir / n for n in ('bin', 'sys/thermal', 'sys/cpufreq', 'sys/battery'))
+    b.mkdir(parents=True, exist_ok=True)
+    (b / 'su').write_text('#!/bin/bash\nif [ "$1" = -c ]; then exec bash -c "$2"; fi\nexec bash\n')
+    (b / 'su').chmod(0o755)
+    os.environ['PATH'] = f'{b}:' + os.environ['PATH']
+    for i, typ in (('3', 'CPU3'), ('9', 'BIG'), ('10', 'MID'), ('11', 'LITTLE'), ('12', 'BIG')):
+        (th / f'thermal_zone{i}').mkdir(parents=True, exist_ok=True)
+        (th / f'thermal_zone{i}/type').write_text(typ + '\n')
+        (th / f'thermal_zone{i}/temp').write_text('60000\n')
+    for pol, mx in POLICIES.items():
+        (cf / pol).mkdir(parents=True, exist_ok=True)
+        (cf / pol / 'cpuinfo_max_freq').write_text(f'{mx}\n')
+        (cf / pol / 'scaling_max_freq').write_text(f'{mx}\n')
+    bat.mkdir(parents=True, exist_ok=True)
+    (bat / 'temp').write_text('300\n')
+    cr.THERMAL, cr.CPUFREQ, cr.BATTERY = str(th), str(cf), str(bat)
+    return th, cf, bat
 
 
 def fake_server(port, pid_file, *extra):
@@ -71,7 +104,8 @@
     cr.OUT_ROOT, cr.DOWNLOADS, cr.FRAME_DIR = root_dir, root_dir / 'downloads', str(frames)
     cr.MODEL = cr.DRAFT = str(gguf)
     cr.PORT = PORT
-    cr.SMOKE_S, cr.SAMPLE_S, cr.SELECTOR_S = 8, 2, 3
+    cr.SMOKE_S, cr.SAMPLE_S, cr.SELECTOR_S, cr.DUMP_S = 8, 2, 3, 1
+    th, cf, bat = make_sysfs(root_dir, cr)
     cr.allowed_cpus = lambda: set(range(8))
     cr.server_cmd = lambda extra=(): [sys.executable, __file__, '--fake-server', str(PORT),
                                       str(root_dir / 'server.pid'), *extra]
@@ -84,7 +118,7 @@
             return [{'class_name': 'person'}]
     cr.Detector = FakeDetector
 
-    cam = {'run': None}
+    cam = {'run': None, 'cached': False}
 
     def writer(stop, session):
         n = 0
@@ -98,11 +132,13 @@
 
     def fake_am(*args):
         log('am.log', ' '.join(args))
+        if os.environ.get('FAKE_STOP_FAIL') and args[0] == 'broadcast' and cam['run']:  # B1's STOP broadcast fails
+            raise subprocess.CalledProcessError(1, ['am', *args])
         if cam['run']:
             cam['run'].set()
-            cam['run'] = None
+            cam['run'] = None  # the process stays cached after STOP, as on the phone, until the force-stop
         if args[0] == 'start':
-            cam['run'] = threading.Event()
+            cam['run'], cam['cached'] = threading.Event(), True
             threading.Thread(target=writer, args=(cam['run'], f'{int(time.time() * 1000) % 0xffffff:x}'),
                              daemon=True).start()
     cr.am = fake_am
@@ -119,7 +155,18 @@
             out += '=== camera_provider 777\n        TOTAL PSS:   270,000\n'
             return 0, out
         if tag == 'pidof':
-            return (0, '4321\n') if cam['run'] else (1, '')
+            return (0, '4321\n') if cam['cached'] else (1, '')
+        if tag == 'forcestop':
+            assert cmd == 'am force-stop com.pixelrobot.robotcam', cmd
+            if cam['run']:  # force-stop also ends a capture that STOP did not end
+                cam['run'].set()
+                cam['run'] = None
+            cam['cached'] = False
+            return 0, ''
+        if tag == 'thermalservice':
+            assert cmd == 'dumpsys thermalservice', cmd
+            hot = hot_since['t'] is not None and time.monotonic() > hot_since['t'] + 4
+            return 0, dump_text(status=4 if hot else 3, skin=46.6 if hot else 35.0)
         if tag == 'lmk':
             return 0, ('logcat_rc=0\n--------- beginning of main\n\n=== lmk lines\n1727650000.123  123  456 I lowmemorykiller: Kill \'com.example\' (999), uid 10123\n'
                        '1727650001.000  123  456 I lowmemorykiller: psi threshold reached\n'
@@ -127,24 +174,28 @@
         return 0, 'uid=0(root)\n'
     cr.root = fake_root
 
-    hot = {'since': None}  # B4 reaches the live thermal pause 5 s after it is entered
+    # B4 reaches Android status CRITICAL (4) 4 s after it is entered, policy6 capped from 2 s in; status 3
+    # (SEVERE) everywhere else is no stop; z9 stays at 90 degC (no longer a stop)
+    hot_since = {'t': None}
     real_run_block = cr.run_block
 
     def run_block(name, *a):
-        hot['since'] = time.monotonic() if name == 'B4_only640_gemma' else None
+        hot_since['t'] = time.monotonic() if name == 'B4_only640_gemma' else None
         try:
             return real_run_block(name, *a)
         finally:
-            hot['since'] = None
+            hot_since['t'] = None
     cr.run_block = run_block
 
     def thermal():
         while True:
             stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
-            z9 = 90000 if hot['since'] is not None and time.monotonic() > hot['since'] + 5 else 36000
+            hot = hot_since['t'] is not None and time.monotonic() > hot_since['t'] + 2
+            (cf / 'policy6/tmp').write_text(f'{2400000 if hot else 2850000}\n')
+            os.replace(cf / 'policy6/tmp', cf / 'policy6/scaling_max_freq')  # atomic: the sysfs value is never empty
             with open(root_dir / 'thermal.log', 'a') as f:
-                f.write(f'{stamp} z9={z9} z10=35000 z11=34000\n')
-            time.sleep(1)
+                f.write(f'{stamp} z9=90000 z10=35000 z11=34000\n')
+            time.sleep(0.2)
 
     def watcher():
         while True:
@@ -171,20 +222,34 @@
 
 
 def check_run(run):
+    sys.path.insert(0, str(HERE))
+    import coresidency as cr
     names = ['B1_mix_nollm', 'B2_only640_nollm', 'B3_mix_gemma', 'B4_only640_gemma']
     for n in names:
         b = json.loads((run / f'block_{n}.json').read_text())
         done = [r for r in b['reads'] if 'detect_ms' in r]
         heat = b['heat_stop']
-        assert heat['zone'] == 'z9' and heat['above_c'] == 80, heat
-        if n == 'B4_only640_gemma':  # the fake log goes to z9 90 degC 5 s in: the block ends there, still valid
-            assert heat['reached_limit'] and 2 < heat['time_to_limit_s'] < 7.5, heat
-            assert heat['zone_reading']['z9'] == 90000 and b['duration_s'] < b['planned_s'], (heat, b['duration_s'])
-            assert all(r['t'] <= heat['time_to_limit_s'] for r in b['reads']), 'read after the heat stop'
-            assert len(done) >= 4, (n, len(done))
-        else:
-            assert not heat['reached_limit'] and heat['time_to_limit_s'] is None, heat
+        s = cr.block_summary(b)
+        if n == 'B4_only640_gemma':  # the fake goes to Android status 4 4 s in: the block ends there, still valid
+            assert heat['reached_limit'] and heat['limit'] == 'android_status' and 1 < heat['time_to_limit_s'] < 7.5, heat
+            assert heat['reading']['status'] == 4 and heat['reading']['skin'] == 46.6, heat
+            assert b['duration_s'] < b['planned_s'], (heat, b['duration_s'])
+            assert all(r['t'] <= heat['time_to_limit_s'] for r in b['reads']), 'read after the limit stop'
+            assert len(done) >= 3, (n, len(done))
+            assert s['status_max'] == 4 and s['skin'][2] == 46.6 and s['capped'][0] >= 1, s
+            assert s['low_max'] == [1803000, 2348000, 2400000], s['low_max']
+        else:  # status 3 (SEVERE) and z9 90 degC throughout: no stop
+            assert not heat['reached_limit'] and heat['limit'] is None and heat['time_to_limit_s'] is None, heat
             assert len(done) >= 10, (n, len(done))
+            assert s['status_max'] == 3 and s['skin'] == (35.0, 35.0, 35.0) and s['capped'] == (0, 0.0), s
+            assert s['low_max'] == list(POLICIES.values()), s['low_max']
+        assert s['max_unread'] == (0, 0.0), s
+        assert len(b['fast']) >= 5 and b['dumps'] and b['fast'][0]['t'] < 0, 'limit readings start before the camera'
+        assert all(x['cpu_c'] == 60.0 and x['bat_c'] == 30.0 for x in b['fast']), b['fast'][:2]
+        assert b['thermal_start']['skin'] == 35.0 and not b['thermal_start']['warm_start'], b['thermal_start']
+        ce = b['camera_end']
+        assert ce['capture_stopped'] and ce['check_s'] >= 3 and ce['force_stop_rc'] == 0, ce
+        assert ce['pids_after_force_stop'] == [], ce
         frames = [r['frame'] for r in done]
         assert frames == sorted(set(frames)), f'{n}: frame numbers must advance'
         assert all('read_ms' in r and 'decode_ms' in r and 'age_s' in r for r in done), n
@@ -207,13 +272,30 @@
     b5 = json.loads((run / 'block_B5_mtp_ram_snapshot.json').read_text())
     assert b5['cmd'][-6:] == ['--model-draft', b5['cmd'][-5], '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']
     assert b5['reply'] == 'fake reply' and b5['detected_640_on_frame'] > 0 and 'VmHWM' in b5['server_status_kb']
+    assert b5['camera_end']['capture_stopped'] and b5['camera_end']['force_stop_rc'] == 0, b5['camera_end']
     text = (run / 'report.txt').read_text()
     for label in ('frames processed', 'detect320 ms median/P95', 'selector ms median/P95', 'min MemAvailable MiB',
                   'peak PSS camera provider MiB', 'LMK log lines', 'zone9 start/end/max', 'mean battery W',
-                  'Gemma load', 'MTP snapshot'):
+                  'Gemma load', 'MTP snapshot', 'idle skin 35.0 degC'):
         assert label in text, label
-    row = next(l for l in text.splitlines() if l.startswith('time to limit s')).split()[4:]
-    assert row[:3] == ['-', '-', '-'] and 2 < float(row[3]) < 7.5, row
+    row = lambda label: next(l for l in text.splitlines() if l.startswith(label))[len(label):].split()
+    assert row('time to limit s')[:3] == ['-', '-', '-'] and 1 < float(row('time to limit s')[3]) < 7.5, text
+    assert row('stop limit') == ['-', '-', '-', 'android_status'], text
+    assert row('VIRTUAL-SKIN start/end/max degC')[0] == '35.0/35.0/35.0' and \
+        row('VIRTUAL-SKIN start/end/max degC')[3] == '35.0/46.6/46.6', text
+    assert row('Android status max') == ['3', '3', '3', '4'], text
+    assert row('policy capped s (% of block)')[:2] == ['0', '(0%)'] and row('policy capped s (% of block)')[6] != '0'
+    assert row('lowest scaling_max MHz 0/4/6') == ['1803/2348/2850'] * 3 + ['1803/2348/2400'], text
+    assert row('gate wait s') == ['0'] * 4, text
+    with tempfile.TemporaryDirectory() as tmp:  # a gate not reached in 15 min: "warm start", not INCOMPLETE
+        shutil.copytree(run, Path(tmp) / 'r')
+        b1 = Path(tmp) / 'r/block_B1_mix_nollm.json'
+        b = json.loads(b1.read_text())
+        b['thermal_start'].update(waited_s=900, warm_start=True)
+        b1.write_text(json.dumps(b))
+        warm, bad = cr.report(Path(tmp) / 'r')
+        assert not bad and next(l for l in warm.splitlines() if l.startswith('gate wait s')).split()[3:6] == \
+            ['900', 'warm', 'start'], warm
     assert 'INCOMPLETE' not in text, text
     assert (run.parent / 'downloads' / f'coresidency_{run.name}_report.txt').exists()
     return text
@@ -223,6 +305,9 @@
     am = (root_dir / 'am.log').read_text().splitlines()
     last_start = max(i for i, l in enumerate(am) if ' start ' in l)
     assert any('broadcast' in l and 'STOP' in l for l in am[last_start + 1:]), 'RobotCam not stopped after last start'
+    start_t = float(am[last_start].split()[0])  # am.log and root.log share time.monotonic()
+    assert any(l.split()[1] == 'forcestop:' and float(l.split()[0]) > start_t
+               for l in (root_dir / 'root.log').read_text().splitlines()), 'app not force-stopped after the last start'
     pid = int((root_dir / 'server.pid').read_text().split()[0])
     for _ in range(50):
         if not alive(pid):
@@ -240,7 +325,10 @@
     print('failed logcat query reported as failed, not as 0 kills: PASS')
 
 
-NO_HEAT = {'zone': 'z9', 'above_c': 80, 'reached_limit': False, 'time_to_limit_s': None, 'zone_reading': None}
+NO_HEAT = {'reached_limit': False, 'limit': None, 'reason': None, 'time_to_limit_s': None, 'reading': None}
+END_OK = {'capture_stopped': True, 'frames_seen': [], 'check_s': 3.1, 'force_stop_rc': 0, 'force_stop_output': '',
+          'pids_after_force_stop': []}
+LIMIT_READINGS = {'fast': [], 'dumps': [], 'cpuinfo_max_khz': POLICIES}
 
 
 def check_sampling_edges():
@@ -255,7 +343,7 @@
     samples = [dict(base, t=0, t_end=2), dict(base, t=5, t_end=13), dict(base, t=15, t_end=17),
                dict(base, t=175, t_end=183, mem_available_mib=100)]
     b = {'reads': [], 'selector_calls': [], 'duration_s': 180.0, 'samples': samples, 'heat_stop': NO_HEAT,
-         'thermal_start': {'z9': 40000}, 'thermal_end': {'z9': 41000}}
+         'thermal_start': {'z9': 40000}, 'thermal_end': {'z9': 41000}, **LIMIT_READINGS}
     s = cr.block_summary(b)
     assert s['late_samples'] == 1 and s['min_avail'] == 3000, s   # the sample finished after 180 s is dropped
     assert s['gaps'] == (2, 163.0), s['gaps']                      # 2 -> 13 and 17 -> 180
@@ -335,7 +423,9 @@
                'selector_calls': slow, 'duration_s': 180.0, 'planned_s': 180, 'heat_stop': NO_HEAT}]
     for b in blocks[:2]:
         b['heat_stop'] = NO_HEAT
-    good = {'sel': (9, 0, 7, 1.0, 2.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
+    for b in blocks:
+        b['camera_end'] = END_OK
+    good = {'max_unread': (0, 0.0), 'sel': (9, 0, 7, 1.0, 2.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
             'failed': {}, 'frames': 300, 'n320': 264, 'n640': 36, 'drift320': (70, 72), 'drift640': (260, 270),
             'drift_na': False}
     sums = [dict(good, sel=(0, 9, 0, None, None)),
@@ -377,13 +467,21 @@
         except cr.CoresLost:
             pass
         assert ctx['server'] is None and not alive(int(pid_file.read_text().split()[0])), 'server left after CoresLost'
-        run = {'smoke': True, 'block_s': 20, 'idle': {'z9': 36000}, 'sha256': {'coresidency.py': '0' * 64}}
+        run = {'smoke': True, 'block_s': 20, 'idle': {'z9': 36000, 'skin': 35.0}, 'sha256': {'coresidency.py': '0' * 64}}
         (tmp / 'run.json').write_text(json.dumps(run))
         sample = {'mem_available_mib': 1, 'swap_used_mib': 0, 'pss_kb': {}, 'pss_error': 'su rc 1'}
-        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(
-            {'sample': sample, 'server_status_kb': {}, 'load_s': 1.0, 'cmd': ['x']}))
+        b5 = {'sample': sample, 'server_status_kb': {}, 'load_s': 1.0, 'cmd': ['x'], 'camera_end': END_OK}
+        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(b5))
         text, bad = cr.report(tmp)
         assert bad and 'B5_mtp_ram_snapshot' in bad[0] and 'INCOMPLETE: B5_mtp_ram_snapshot' in text, (bad, text)
+        sample.pop('pss_error')  # B5's own end check: a failed force-stop is INCOMPLETE before the report is final
+        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(dict(b5, sample=sample)))
+        assert cr.report(tmp)[1] == []
+        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(
+            dict(b5, sample=sample, camera_end=dict(END_OK, force_stop_rc=1, force_stop_output='Error'))))
+        text, bad = cr.report(tmp)
+        assert bad == ["B5_mtp_ram_snapshot: RobotCam end check: am force-stop failed (rc 1: 'Error')"], bad
+        assert 'INCOMPLETE: B5_mtp_ram_snapshot: RobotCam end check' in text, text
         old = tmp / 'old_run'
         old.mkdir()
         (old / 'run.json').write_text(json.dumps(run))  # made by the frozen runner: no heat_stop
@@ -401,15 +499,14 @@
     sys.path.insert(0, str(HERE))
     import coresidency as cr
     cr.SELECTOR_S = 20
-    assert (cr.PAUSE_ZONE, cr.PAUSE_ABOVE_C) == ('z9', 80), 'read from main.Robot.run_mission / get_temp'
     ok_lmk = {'ok': True, 'logcat_rc': 0, 'raw_head': ''}
     alive_cam = {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': True}
-    good = {'sel': (1, 0, 1, 1962.0, 1962.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
+    good = {'max_unread': (0, 0.0), 'sel': (1, 0, 1, 1962.0, 1962.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
             'failed': {}, 'frames': 40, 'n320': 37, 'n640': 3, 'drift320': (101, 101), 'drift640': (493, 493),
             'drift_na': False}
     call = lambda t, end: {'t': t, 'started_s': t, 'ended_s': end, 'ms': 2000.0, 'correct': True}
     gemma = lambda **k: dict({'block': 'B3_mix_gemma', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk,
-                              'survived': alive_cam, 'heat_stop': NO_HEAT}, **k)
+                              'survived': alive_cam, 'heat_stop': NO_HEAT, 'camera_end': END_OK, **LIMIT_READINGS}, **k)
     # the smoke evidence: 20 s block ran 20.25 s, one call at slot 0 (old count ceil(20.25/20) = 2)
     smoke = gemma(duration_s=20.25, planned_s=20, selector_calls=[call(0.0, 1.97)])
     assert cr.problems([smoke], [good]) == [], cr.problems([smoke], [good])
@@ -421,8 +518,10 @@
         'ended after the block at slots []']
     print('selector slots = those that start inside the block (smoke 1, full 9): PASS')
 
-    # heat stop at 42 s: 3 slots (0, 20, 40); the call in flight at the stop is not an overrun; drift n/a, valid
-    heat = dict(NO_HEAT, reached_limit=True, time_to_limit_s=42.0, zone_reading={'z9': 81000})
+    # limit stop at 42 s: 3 slots (0, 20, 40); the call in flight at the stop is not an overrun; drift n/a, valid
+    reading = {'skin': 46.6, 'status': 4, 'bat_c': 38.0, 'cpu_c': 65.0, 'scaling_max': {'policy6': 1000000}}
+    heat = dict(NO_HEAT, reached_limit=True, limit='android_status', reason='Android thermal status 4 >= 4 (CRITICAL)',
+                time_to_limit_s=42.0, reading=reading)
     reads = [{'t': 0.5 * i, 'status': 'ok', 'detect_ms': 100.0, 'size': 640 if i % 10 == 0 else 320,
               'read_ms': 1, 'decode_ms': 5, 'age_s': 0.3, 'frame': i + 1} for i in range(84)]
     samples = [{'t': 5.0 * i, 't_end': 5.0 * i + 1, 'mem_available_mib': 3000, 'swap_used_mib': 0, 'pss_kb': {},
@@ -443,27 +542,190 @@
     # mix stopped at 3 s: no 640 frame was due yet
     early = dict(hot, duration_s=3.0, heat_stop=dict(heat, time_to_limit_s=3.0), selector_calls=[call(0.0, 2.0)])
     assert cr.problems([early], [dict(s, n640=0, sel=(1, 0, 1, 2000, 2000))]) == []
-    assert cr.problems([dict(early, heat_stop=NO_HEAT)], [dict(s, n640=0, drift_na=False)])  # no heat stop: missing
-    assert s['z9'][2] == 81.0, s['z9']  # the stop reading counts in the max even if no sample saw it
-    # the smoke case: B3 started at 81 degC and stops before its first frame and before any sample finished
+    assert cr.problems([dict(early, heat_stop=NO_HEAT)], [dict(s, n640=0, drift_na=False)])  # no limit stop: missing
+    # round-3 finding: a limit already met before the first frame is "not run: limit at start", INCOMPLETE, with the
+    # readings, and nothing else (no RobotCam survival or selector lines)
     now = gemma(duration_s=0.02, planned_s=20, reads=[], selector_calls=[], thermal_start={'z9': 81000, 'waited_s': 0},
                 thermal_end={'z9': 79000}, heat_stop=dict(heat, time_to_limit_s=0.01),
                 samples=[dict(samples[0], t=0.0, t_end=1.3)])
     s = cr.block_summary(now)
-    assert s['frames'] == 0 and s['min_avail'] is None and s['z9'][2] == 81.0, s
-    assert cr.problems([now], [s]) == [], cr.problems([now], [s])
-    # the same block without a heat stop, or with a failed read, stays INCOMPLETE
+    assert s['frames'] == 0 and s['min_avail'] is None, s
+    assert cr.problems([now], [s]) == [
+        "B3_mix_gemma: not run: limit at start (Android thermal status 4 >= 4 (CRITICAL); skin 46.6, status 4, "
+        "battery 38.0, CPU max 65.0 degC, scaling_max {'policy6': 1000000})"], cr.problems([now], [s])
+    # the same block without a limit stop, or with a failed read, stays INCOMPLETE on the usual rules
     assert len(cr.problems([dict(now, heat_stop=NO_HEAT)], [dict(s, drift_na=False)])) >= 4
     assert any('RobotCam' in p for p in cr.problems([now], [dict(s, failed={'missing': 1})]))
+    # skin loss: the fail-closed stop is INCOMPLETE and its selector slots are counted to the planned end
+    lost = dict(hot, heat_stop=dict(NO_HEAT, limit='fail_closed', reason='no VIRTUAL-SKIN and Android status reading '
+                                     'for 60 s', time_to_limit_s=61.2, reading=reading))
+    bad = cr.problems([lost], [dict(s, frames=120, n640=12, drift_na=False, sel=(4, 0, 4, 2000, 2000))])
+    assert bad[0] == 'B3_mix_gemma: stopped fail-closed at 61.2 s: no VIRTUAL-SKIN and Android status reading for 60 s'
+    print('limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop '
+          'INCOMPLETE: PASS')
+
+
+def check_cleanup1():
+    sys.path.insert(0, str(HERE))
+    import coresidency as cr
+    from PIL import Image
+    # block limits, each at its edge (times: monotonic s; the monitor began at 100)
+    fs = lambda t, cpu=60.0, bat=30.0: {'t': t, 'cpu_c': cpu, 'bat_c': bat, 'max': {}}
+    dm = lambda t, status=0, skin=35.0: {'t': t, 'status': status, 'skin': skin}
+    L = lambda now, fast, dumps: cr.block_limit(now, 100.0, fast, dumps)
+    ok = [fs(100 + i) for i in range(5)]
+    assert L(104.5, ok, [dm(104, status=3)]) is None, 'SEVERE (3) is no stop'
+    assert L(104.5, ok, [dm(104, status=4)]) == ('android_status', 'Android thermal status 4 >= 4 (CRITICAL)')
+    assert L(104.5, ok, [dm(104, status=4, skin=None)])[0] == 'android_status', 'status alone is enough'
+    assert L(104.5, ok, [dm(104, skin=48.0)]) is None, 'skin itself is no stop (Android acts on it)'
+    assert L(104.5, ok[:-1] + [fs(104, bat=44.9)], [dm(104)]) is None
+    assert L(104.5, ok[:-1] + [fs(104, bat=45.0)], [dm(104)]) == ('battery', 'battery 45.0 >= 45.0 degC')
+    assert L(104.5, ok[:-2] + [fs(103, bat=45.0), fs(104, bat=None)], [dm(104)])[0] == 'battery', 'latest readable'
+    hot = lambda t: fs(t, cpu=110.0)
+    assert L(104.5, [fs(100), fs(101), hot(102), hot(103), fs(104, cpu=109.9)], [dm(104)]) is None
+    assert L(104.5, [fs(100), fs(101), fs(102), hot(103), hot(104)], [dm(104)]) is None, '2 hot samples'
+    assert L(104.5, [fs(100), fs(101), hot(102), hot(103), hot(104)], [dm(104)])[0] == 'cpu_fault'
+    assert L(104.5, [fs(100), hot(101), hot(102), fs(103, cpu=None), hot(104)], [dm(104)])[0] == 'cpu_fault', \
+        'an unreadable sample neither breaks nor resets the run'
+    # skin/status loss: fail closed after 60 s, counted from the monitor start; either one alone keeps it fresh
+    assert L(159.9, [fs(159)], []) is None
+    assert L(160.1, [fs(160)], []) == ('fail_closed', 'no VIRTUAL-SKIN and Android status reading for 60 s')
+    assert L(170.0, [fs(170)], [dm(111)]) is None and L(171.5, [fs(171)], [dm(111)])[0] == 'fail_closed'
+    assert L(171.5, [fs(171)], [dm(111), dm(150, skin=None)])[0] == 'fail_closed', 'skin lost at 111'
+    assert L(171.5, [fs(171)], [dm(150, status=None), dm(151, skin=None)]) is None, 'from two dumps'
+    assert L(105.0, [fs(100)], [dm(104)]) is None
+    assert L(105.5, [fs(100)], [dm(105)]) == ('fail_closed', 'no CPU zone reading for 5 s')
+    assert L(105.5, [fs(100)] + [fs(101 + i, bat=None) for i in range(5)], [dm(105)]) == \
+        ('fail_closed', 'no battery temperature reading for 5 s')
+    print('block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), '
+          'skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS')
+
+    # a limit met before the first frame: frame_loop returns before any read
+    hit = {'limit': 'battery', 'reason': 'x', 'time_to_limit_s': 0.0}
+    reads = []
+    assert cr.frame_loop(None, 'mix', 's', time.monotonic(), 20, reads, lambda: hit) == (None, hit) and reads == []
+
+    # cooldown gate: reached; not reached in GATE_MAX_S -> warm start (not INCOMPLETE); smoke does not wait
+    real = cr.read_thermal, cr.read_dump, cr.GATE_MAX_S, cr.GATE_POLL_S
+    idle = {'z9': 30000, 'skin': 32.0}
+    try:
+        cr.GATE_MAX_S, cr.GATE_POLL_S = 0.6, 0.1
+        skins = iter([36.0, 34.0, 33.5])
+        cr.read_thermal = lambda path: {'at': 'x', 'z9': 33000, 'z10': 0, 'z11': 0}
+        cr.read_dump = lambda: {'skin': next(skins), 'status': 0}
+        g = cr.thermal_gate('log', idle, 'B1', False)
+        assert g['skin'] == 33.5 and not g['warm_start'], g   # 33.5 <= 32 + 1.5, z9 33 <= 30 + 4
+        cr.read_dump = lambda: {'skin': 33.6, 'status': 1}
+        g = cr.thermal_gate('log', idle, 'B1', False)
+        assert g['warm_start'] and g['skin'] == 33.6, g
+        cr.read_dump = lambda: {'skin': 33.0, 'status': 0}
+        cr.read_thermal = lambda path: {'at': 'x', 'z9': 34001, 'z10': 0, 'z11': 0}
+        assert cr.thermal_gate('log', idle, 'B1', False)['warm_start'], 'z9 still gates'
+        cr.read_dump = lambda: {'skin': None, 'status': None}
+        cr.read_thermal = lambda path: {'at': 'x', 'z9': 30000, 'z10': 0, 'z11': 0}
+        assert cr.thermal_gate('log', idle, 'B1', False)['warm_start'], 'no skin reading is not cool'
+        began = time.monotonic()
+        assert not cr.thermal_gate('log', idle, 'B1', True)['warm_start'] and time.monotonic() - began < 0.1
+    finally:
+        cr.read_thermal, cr.read_dump, cr.GATE_MAX_S, cr.GATE_POLL_S = real
+    print('cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS')
+
+    # RobotCam end check: capture stopped vs still advancing; force-stop failure
     with tempfile.TemporaryDirectory() as tmp:
-        log = Path(tmp) / 'thermal.log'
-        for z9, hit in ((80000, False), (80999, False), (81000, True), (95000, True)):
-            log.write_text(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} z9={z9} z10=90000 z11=90000\n')
-            assert (cr.heat_reached(str(log)) is not None) == hit, z9  # get_temp() // 1000 > 80, zone 9 only
-        log.write_text('2026-09-30T15:0')  # partial line: retried next frame
-        assert cr.heat_reached(str(log)) is None
-    print('heat stop: zone9 // 1000 > 80 from main.py; short or immediate stop valid, drift n/a, in-flight call '
-          'allowed, stop reading in the max: PASS')
+        cr.FRAME_DIR = tmp
+        stop = threading.Event()
+
+        def writer(n_max):
+            n = 0
+            while not stop.is_set() and n < n_max:
+                n += 1
+                boot_ms = int(time.clock_gettime(time.CLOCK_BOOTTIME) * 1000)
+                c = f'robotcam session=ab frame={n} capture_boot_ms={boot_ms} capture_wall_ms=0 clock=sensor'
+                Image.new('RGB', (64, 48)).save(Path(tmp) / 't.jpg', comment=c.encode())
+                os.replace(Path(tmp) / 't.jpg', Path(tmp) / 'frame.jpg')
+                stop.wait(0.5)
+        calls = []
+
+        def fake_root(cmd, tag, timeout=60):
+            calls.append((tag, cmd))
+            return (0, '') if tag == 'forcestop' else (0, '4321\n')
+        th = threading.Thread(target=writer, args=(2,))  # two more frames after STOP, then capture ends
+        th.start()
+        r = cr.camera_end_check(fake_root)
+        th.join()
+        assert r['capture_stopped'] and [x['frame'] for x in r['frames_seen']] == [1, 2] and r['check_s'] >= 3.4, r
+        assert calls == [('forcestop', 'am force-stop com.pixelrobot.robotcam'), ('pidof', 'pidof com.pixelrobot.robotcam')]
+        assert r['pids_after_force_stop'] == ['4321'] and cr.camera_end_failed(r) is None, 'a pid left is recorded only'
+        th = threading.Thread(target=writer, args=(1000,))  # still capturing
+        th.start()
+        r = cr.camera_end_check(fake_root, quiet_s=3, limit_s=5)
+        stop.set()
+        th.join()
+        assert not r['capture_stopped'] and len(r['frames_seen']) >= 9, r
+        assert cr.camera_end_failed(r).startswith('capture not shown stopped 5.'), r
+        assert [c[0] for c in calls[-2:]] == ['forcestop', 'pidof'], 'force-stop also when capture did not stop'
+        (Path(tmp) / 'frame.jpg').write_bytes(b'not a jpeg')  # unreadable reads are no evidence of a stop
+        r = cr.camera_end_check(fake_root, quiet_s=1, limit_s=2)
+        assert not r['capture_stopped'] and r['unreadable_reads'] >= 10 and not r['frames_seen'], r
+        assert 'unreadable reads' in cr.camera_end_failed(r)
+        os.unlink(Path(tmp) / 'frame.jpg')
+        r = cr.camera_end_check(lambda cmd, tag, timeout=60: (1, 'Error: no permission\n'), quiet_s=0.3)
+        assert r['capture_stopped'] and cr.camera_end_failed(r) == "am force-stop failed (rc 1: 'Error: no permission')"
+
+        def boom(cmd, tag, timeout=60):
+            raise subprocess.TimeoutExpired(cmd, timeout)
+        r = cr.camera_end_check(boom, quiet_s=0.3)
+        assert r['force_stop_rc'] is None and 'force-stop failed (rc None' in cr.camera_end_failed(r), r
+        b = {'block': 'B1_mix_nollm', 'gemma': False, 'mode': '640', 'heat_stop': NO_HEAT, 'camera_end': r,
+             'survived': {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': None},
+             'lmk': {'ok': True}}
+        good = {'max_unread': (0, 0.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'], 'failed': {}, 'frames': 300,
+                'n640': 300, 'drift640': (1, 1), 'drift_na': False}
+        bad = cr.problems([b], [good])
+        assert len(bad) == 1 and bad[0].startswith('B1_mix_nollm: RobotCam end check: am force-stop failed (rc None: ')\
+            and 'TimeoutExpired' in bad[0], bad
+        assert cr.problems([dict(b, camera_end=END_OK)], [good]) == []
+    print('RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure '
+          'INCOMPLETE; cached pid recorded only: PASS')
+
+    # capped time: in-block samples only, each counting the time since the previous one; unreadable flagged
+    mx = lambda t, p6=2850000, p0=1803000: {'t': t, 'max': {'policy0': p0, 'policy4': 2348000, 'policy6': p6}}
+    blk = {'reads': [], 'selector_calls': [], 'samples': [], 'heat_stop': NO_HEAT, 'dumps': [],
+           'thermal_start': {'z9': 40000}, 'thermal_end': {'z9': 41000}, 'cpuinfo_max_khz': POLICIES,
+           'duration_s': 10.0, 'fast': [mx(-1.0, p6=1000000), mx(1.0), mx(2.0, p6=2400000), mx(5.0, p6=2400000),
+                                        mx(6.0), mx(7.0, p0=None), mx(10.0), mx(11.0, p6=500000)]}
+    s = cr.block_summary(blk)
+    assert s['capped'] == (4.0, 40.0), s['capped']                      # 1-2 and 2-5 (a skipped slot), not -1 or 11
+    assert s['max_unread'] == (1, 1.0) and s['low_max'] == [1803000, 2348000, 2400000], s
+    good = {'max_unread': (0, 0.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'], 'failed': {}, 'frames': 20,
+            'n640': 20, 'drift640': (1, 1), 'drift_na': False}
+    b = dict(blk, block='B2_only640_nollm', gemma=False, mode='640', camera_end=END_OK, lmk={'ok': True},
+             survived={'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': None})
+    assert cr.problems([b], [dict(good, max_unread=s['max_unread'])]) == [
+        'B2_only640_nollm: 1 1 s sample(s) without every scaling_max_freq (1.0 s of the block not known capped or not)']
+    assert cr.problems([b], [good]) == []
+    print('capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable '
+          'scaling_max INCOMPLETE: PASS')
+
+    # discovery from a fake sysfs through the real persistent root shell (lowest zone of each type)
+    with tempfile.TemporaryDirectory() as tmp:
+        path = os.environ['PATH']
+        make_sysfs(Path(tmp), cr)
+        sh = cr.RootShell()
+        try:
+            lay = cr.discover(sh)
+            assert lay == {'cpu_zones': {'BIG': '9', 'MID': '10', 'LITTLE': '11'}, 'policies': POLICIES}, lay
+            x = cr.fast_sample(sh, cr.fast_keys(lay))
+            assert x['cpu_c'] == 60.0 and x['bat_c'] == 30.0 and x['max'] == POLICIES and 'error' not in x, x
+            os.unlink(Path(tmp) / 'sys/thermal/thermal_zone10/temp')
+            assert cr.fast_sample(sh, cr.fast_keys(lay))['cpu_c'] is None
+        finally:
+            sh.close()
+            os.environ['PATH'] = path
+    d = cr.parse_dump(dump_text(status=4, skin=46.6))
+    assert (d['status'], d['skin']) == (4, 46.6), d
+    assert cr.parse_dump(dump_text(skin=None))['skin'] is None and cr.parse_dump('Failure calling service')['status'] is None
+    print('layout discovery, 1 s sample and dump parsing: PASS')
 
 
 def check_fix1_cold():
@@ -513,6 +775,7 @@
 
 
 def main():
+    check_cleanup1()
     check_fix1_cadence_and_heat()
     check_fix1_cold()
     check_lmk_failure()
@@ -554,6 +817,20 @@
         check_stopped(root_dir)
         print('--resume after SIGTERM: PASS')
 
+        root_dir = Path(tmp) / 'c'  # the STOP broadcast fails at B1's end: the force-stop still follows
+        root_dir.mkdir()
+        p = subprocess.Popen([sys.executable, __file__, '--child', str(root_dir)], stdout=subprocess.PIPE,
+                             stderr=subprocess.STDOUT, text=True, env=dict(os.environ, FAKE_STOP_FAIL='1'))
+        out, _ = p.communicate(timeout=120)
+        assert p.returncode != 0 and 'CalledProcessError' in out and 'AssertionError' not in out, (p.returncode, out[-2000:])
+        assert not list(root_dir.glob('run_*_smoke/block_*.json')), 'a block with a failed STOP was kept'
+        am = (root_dir / 'am.log').read_text().splitlines()
+        assert sum(' start ' in l for l in am) == 1 and am[-1].split()[1] == 'broadcast', am
+        stop_t = float(am[-1].split()[0])
+        rl = [l.split() for l in (root_dir / 'root.log').read_text().splitlines()]
+        assert any(l[1] == 'forcestop:' and float(l[0]) > stop_t for l in rl), 'no force-stop after the failed STOP'
+        print('failed STOP broadcast: force-stop still runs: PASS')
+
 
 if __name__ == '__main__':
     if sys.argv[1:2] == ['--fake-server']:
diff -ru thermal_char/test_thermal_char.py ../cur/thermal_char/test_thermal_char.py
--- thermal_char/test_thermal_char.py	2026-09-30 22:07:41.724762011 +0000
+++ ../cur/thermal_char/test_thermal_char.py	2026-10-01 01:10:40.016767372 +0000
@@ -48,6 +48,8 @@
     (b / 'su').write_text('#!/bin/bash\nif [ "$1" = -c ]; then exec bash -c "$2"; fi\nexec bash\n')
     (b / 'dumpsys').write_text(f'#!/bin/bash\n[ "$1" = thermalservice ] || exit 1\ncat {T}/dump.txt\n')
     (b / 'pidof').write_text(f'#!/bin/bash\ncat {T}/robotcam_pids 2>/dev/null\n')
+    # only the force-stop goes through su; it ends the process STOP leaves cached
+    (b / 'am').write_text(f'#!/bin/bash\n[ "$1" = force-stop ] || exit 1\necho "$*" >>{T}/am_root.log\n: >{T}/robotcam_pids\n')
     for x in b.iterdir():
         x.chmod(0o755)
     th, cf, bat = sysd / 'class/thermal', sysd / 'cpufreq', sysd / 'battery'
@@ -178,8 +180,7 @@
         log(' '.join(args))
         if cam['run']:
             cam['run'].set()
-            cam['run'] = None
-            (T / 'robotcam_pids').write_text('')
+            cam['run'] = None  # the app process stays cached (pid kept) until the force-stop, as on the phone
         if args[0] == 'start':
             cam['run'], cam['since'] = threading.Event(), time.monotonic()
             (T / 'robotcam_pids').write_text('4321\n')
@@ -338,6 +339,16 @@
             ev = rep.split('EVENTS')[1].split('TIMELINE')[0]
             at = f'{rows[1]["t"] - L0:+.1f}'
             assert f'  {at} [load] policy6 scaling_max_freq 2850 -> 2400 MHz' in ev and f'(sample {at})' in ev, ev
+            run = json.loads((T / 'rep/run.json').read_text())
+            ok_cam = {'capture_stopped': True, 'frames_seen': [], 'check_s': 3.1, 'force_stop_rc': 0,
+                      'force_stop_output': '', 'pids_after_force_stop': ['4321']}  # a pid left is recorded, no failure
+            for cam, want in ((ok_cam, None), (dict(ok_cam, capture_stopped=False, frames_seen=[{}] * 30, check_s=15.0),
+                                               'capture not shown stopped 15.0 s after STOP (30 new frames, None unreadable reads)'),
+                              (dict(ok_cam, force_stop_rc=1, force_stop_output='Error'), 'am force-stop failed (rc 1')):
+                run['after_stop'] = {'robotcam': cam, 'llama_server_running': False, 'worker_threads_alive': []}
+                (T / 'rep/run.json').write_text(json.dumps(run))
+                bad = [b for b in tc.report(T / 'rep')[1] if 'after the load stop' in b]
+                assert (bad == []) if want is None else (len(bad) == 1 and want in bad[0]), (want, bad)
             r0, r1 = rows[0]['read_s'], rows[1]['read_s']
             assert f'read duration (stamped at completion): median {(r0 + r1) / 2:.3f} s, max {max(r0, r1):.3f} s' \
                 in rep, rep
@@ -429,6 +440,14 @@
         return False
 
 
+def check_after_stop(T, run):
+    a = run['after_stop']
+    rc = a.pop('robotcam')
+    assert a == {'llama_server_running': False, 'worker_threads_alive': []}, a
+    assert rc['capture_stopped'] and rc['check_s'] >= 3 and rc['force_stop_rc'] == 0, rc
+    assert rc['pids_after_force_stop'] == [] and 'force-stop com.pixelrobot.robotcam' in (T / 'am_root.log').read_text()
+
+
 def check(name, T, proc, output):
     frag, rc, complete = SCENARIOS[name]
     assert proc.returncode == rc, (name, proc.returncode, output[-1500:])
@@ -451,7 +470,7 @@
         assert am[-1].startswith('broadcast'), 'RobotCam not stopped last'  # a stop in camera_start comes after am start
         if name == 'startup':  # stopped while llama-server was loading: RobotCam never started
             assert not any(l.startswith('start ') for l in am), 'RobotCam started'
-        assert run['after_stop'] == {'robotcam_pids': [], 'llama_server_running': False, 'worker_threads_alive': []}
+        check_after_stop(T, run)
         print(f'ok: {name}: stop "{run["stop_reasons"][0][:70]}", exit {proc.returncode}')
         return
     for sec in ('STOP REASON: ', 'LAYOUT', 'EVENTS', 'TIMELINE every 3 s', 'WORK RATE per 2 s', 'SLOWDOWN', 'COOLDOWN'):
@@ -466,8 +485,7 @@
     assert not (T / 'robotcam_pids').read_text().strip()
     mods = (T / 'modules.txt').read_text().split() if name != 'sigterm' else []
     assert 'motors' not in mods and not any('serial' in m for m in mods), name
-    assert run['after_stop'] == {'robotcam_pids': [], 'llama_server_running': False, 'worker_threads_alive': []}, \
-        (name, run['after_stop'])
+    check_after_stop(T, run)
     samples = [json.loads(l) for l in (out / 'samples_1s.jsonl').read_text().splitlines()]
     frames = [json.loads(l) for l in (out / 'frames.jsonl').read_text().splitlines()]
     gens = [json.loads(l) for l in (out / 'gen.jsonl').read_text().splitlines()]
diff -ru thermal_char/thermal_char.py ../cur/thermal_char/thermal_char.py
--- thermal_char/thermal_char.py	2026-09-30 22:03:13.572761880 +0000
+++ ../cur/thermal_char/thermal_char.py	2026-10-01 00:50:33.008766782 +0000
@@ -16,8 +16,9 @@
 VIRTUAL-SKIN + status reading for 60 s (counted from load start), no CPU zone or battery temperature reading
 for 5 s (each limit uses only its own readings), no new RobotCam
 frame for 10 s, a llama-server failure, or cores 4-7 lost. The thermal stops also apply while llama-server
-and RobotCam are starting (then no load runs). After the load, stops RobotCam and llama-server and keeps
-logging 5 min (cooldown). --smoke: 60 s load, 30 s cooldown, same stops.
+and RobotCam are starting (then no load runs). After the load, stops RobotCam and llama-server, checks that capture
+stopped (no new frame for 3 s) and force-stops the app (it stays cached after STOP), and keeps logging 5 min
+(cooldown). --smoke: 60 s load, 30 s cooldown, same stops.
 
 Usage: thermal_char.py [--smoke] [--note TEXT]
 """
@@ -533,7 +534,10 @@
     if any(not s.get('cpus_ok') for s in load_samples):
         bad.append(f'cores 4-7 not allowed in {sum(not s.get("cpus_ok") for s in load_samples)} load samples')
     for k, v in (run.get('after_stop') or {}).items():
-        if v:
+        if k == 'robotcam':  # a cached app process is no failure: capture still advancing or a failed force-stop is
+            if why := cr.camera_end_failed(v):
+                bad.append(f'after the load stop: RobotCam {why}')
+        elif v:
             bad.append(f'after the load stop: {k} {v}')
     if L1 is not None and run.get('after_stop') and samples and samples[-1]['t'] - L1 < run['cool_s'] - 2 * SAMPLE_S:
         bad.append(f'cooldown logged {samples[-1]["t"] - L1:.0f} of {run["cool_s"]} s')
@@ -758,9 +762,8 @@
                 server.stop()
                 for t in workers:
                     t.join(timeout=30)
-                time.sleep(2)  # the STOP broadcast ends the service asynchronously
-                rc, pids = root_file('pidof com.pixelrobot.robotcam', 'pidof')
-                run['after_stop'] = {'robotcam_pids': pids.split(), 'llama_server_running': server.proc.poll() is None,
+                run['after_stop'] = {'robotcam': cr.camera_end_check(root_file),  # STOP ends capture asynchronously
+                                     'llama_server_running': server.proc.poll() is None,
                                      'worker_threads_alive': [t.name for t in workers if t.is_alive()]}
                 save()
                 print(f'[cooldown] logging {cool_s} s', flush=True)
@@ -788,8 +791,7 @@
             t.join(timeout=30)
         if run.get('baseline_end') is not None and 'after_stop' not in run:  # stopped or aborted in startup/load
             try:
-                time.sleep(2)
-                run['after_stop'] = {'robotcam_pids': root_file('pidof com.pixelrobot.robotcam', 'pidof')[1].split(),
+                run['after_stop'] = {'robotcam': cr.camera_end_check(root_file),
                                      'llama_server_running': bool(cr.LIVE) or bool(server and server.proc.poll() is None),
                                      'worker_threads_alive': [t.name for t in workers if t.is_alive()]}
             except (OSError, subprocess.SubprocessError) as e:
```
