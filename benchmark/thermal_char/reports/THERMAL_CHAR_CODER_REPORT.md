# THERMAL_CHAR — Coder report

Role: Coder, Claude Code CLI, claude-opus-5-5, effort medium, Ponytail lite. Reviewer: Codex CLI `codex exec`, gpt-6-sol, effort medium, `-s read-only`, Ponytail off, a fresh session each round. No fallback was used.

**Outcome: STOPPED after review round 3 with CHANGES REQUESTED** (the task allows at most 3 rounds). Round 3 raised one blocking finding, which is not fixed (see "Open blocking finding"). The candidate is **not approved**. Nothing was staged, committed or pushed, and no tracked file was changed.

## Base and tree

- HEAD `3584b5eddd3639c00c37b24e6e31f4d86fce2f2b` on main, checked at the start and before each round. `git status --short --untracked-files=no` stayed empty throughout.
- The only new folder is `benchmark/thermal_char/`. `benchmark/coresidency/` is unmodified: it is read and imported only.
- `.gitignore` line 8 (`benchmark/*`) ignores the new folder. A later commit needs `git add -f`, as the existing benchmark folders evidently had.

## Files (frozen state reviewed in round 3)

```
46c49c727fa02fbc71b4fd5a97c1919a53652bf802a335cfe4a3dea9c03a336d  benchmark/thermal_char/RUN.md
16787b0679c24b711eb9e370cdcc9a9942cf689718764874b8ec3e418f84d77b  benchmark/thermal_char/oneshot.sh
c4df2fa2799f019fd67ac68580c9296001e1e8ea63d51f75ab906184864d682e  benchmark/thermal_char/run_thermal_char.sh
e640e6f33e0c44585322f37394b68936a59b22adc32bde824b649d3d7a6ee2f6  benchmark/thermal_char/test_oneshot.sh
abe5462a5676c4dad1098705dec2ddf6c627ee39d2f6bdeeef6bde4940db1e08  benchmark/thermal_char/test_thermal_char.py
0954e0607f34a1a1fdb8822513b26dba409931286701adfdf7e93c105351813b  benchmark/thermal_char/thermal_char.py
```

| File | Purpose |
|---|---|
| `oneshot.sh` | Copy of `../coresidency/oneshot.sh` with minimal changes (diff below). It keeps every check: native Termux only, no agent running, not charging (at start and after the idle), screen timeout in the #123 form with read-back and restore, lock, TERM/HUP/INT forwarding, 5 min idle, thermal logger and cache-drop watcher. |
| `run_thermal_char.sh` | Copy of `run_coresidency.sh` that runs `thermal_char.py`. |
| `thermal_char.py` | The runner. |
| `test_thermal_char.py` | Offline test with fakes. |
| `test_oneshot.sh` | Copy of `../coresidency/test_oneshot.sh` pointed at the copied launcher, with 3 new checks. |
| `RUN.md` | Exact native commands for the smoke run and the full run. |

### Launcher diff

The co-residency launcher hardcodes its runner and lock folder, so it could not start another runner unchanged. It was copied with these changes only:

```diff
--- benchmark/coresidency/oneshot.sh	2026-09-30 13:36:37.943561148 +0000
+++ benchmark/thermal_char/oneshot.sh	2026-09-30 19:02:59.589818118 +0000
@@ -1,17 +1,17 @@
 #!/data/data/com.termux/files/usr/bin/bash
-# Co-residency benchmark (DECISIONS #124), motors off. Start from NATIVE Termux (prompt "~ $"), no agent running,
-# charger unplugged, Termux in front:
-#   bash ~/robot/benchmark/coresidency/oneshot.sh                  (full run)
-#   bash ~/robot/benchmark/coresidency/oneshot.sh --smoke          (20 s blocks, no cooldown gate)
-#   bash ~/robot/benchmark/coresidency/oneshot.sh --resume <dir>   (continue a run)
+# Thermal characterization run, motors off. Copy of ../coresidency/oneshot.sh with only the paths, the runner name
+# and the refusals of a concurrent co-residency/thermal run changed. Start from NATIVE Termux (prompt "~ $"), no agent
+# running, charger unplugged, Termux in front:
+#   bash ~/robot/benchmark/thermal_char/oneshot.sh --note "<room degC, mounting>"           (full run)
+#   bash ~/robot/benchmark/thermal_char/oneshot.sh --smoke --note "<room degC, mounting>"   (60 s load)
 # From ~/ladder/oneshot.sh (db1f3156...): thermal logger, checks, screen kept on, 5 min idle, root cache drops on
 # request (file handshake with the runner). Unlike it, the runner runs in native Termux, not in Debian: the robot
 # runs ONNX Runtime and llama-server natively. Root commands that call Android services use the DECISIONS #123
 # form: su -c "<cmd> </dev/null >/data/local/tmp/<file> 2>&1", then the file is read.
 H=/data/data/com.termux/files/home
-L=$H/coresidency
-R=$H/robot/benchmark/coresidency
-S=/data/local/tmp/coresidency_screen.txt
+L=$H/thermal_char
+R=$H/robot/benchmark/thermal_char
+S=/data/local/tmp/thermal_char_screen.txt
 mkdir -p "$L"
 CONSOLE=$L/oneshot_console_$(date -u +%Y%m%dT%H%M%SZ).log
 say() { echo "[oneshot] $*" | tee -a "$CONSOLE"; }
@@ -20,15 +20,18 @@
   echo "Run this from native Termux (~ \$), not from Debian."; exit 1
 fi
 su -c id >/dev/null 2>&1 || { echo "No root: su failed. Check Magisk."; exit 1; }
-[ -f "$R/run_coresidency.sh" ] || { echo "Missing $R/run_coresidency.sh"; exit 1; }
+[ -f "$R/run_thermal_char.sh" ] || { echo "Missing $R/run_thermal_char.sh"; exit 1; }
 if pgrep -fa 'claude|agy|node|codex' >/dev/null; then
   echo "An agent is still running:"; pgrep -fa 'claude|agy|node|codex'; echo "Quit it, then rerun."; exit 1
 fi
 if pgrep -fa 'llama-server|chat\.py' >/dev/null; then
   echo "llama-server or chat.py is already running:"; pgrep -fa 'llama-server|chat\.py'; echo "Stop it, then rerun."; exit 1
 fi
-if pgrep -fa 'coresidency\.py' >/dev/null; then
-  echo "A co-residency runner is already running:"; pgrep -fa 'coresidency\.py'; echo "Stop it, then rerun."; exit 1
+if pgrep -fa 'coresidency\.py|thermal_char\.py' >/dev/null; then
+  echo "A co-residency or thermal runner is already running:"; pgrep -fa 'coresidency\.py|thermal_char\.py'; echo "Stop it, then rerun."; exit 1
+fi
+if [ -d "$H/coresidency/.lock" ]; then  # the co-residency launcher uses the same RobotCam and port 8080
+  echo "The co-residency launcher holds $H/coresidency/.lock; wait for it or remove a stale lock."; exit 1
 fi
 check_battery() {  # checked at start and again after the idle, right before the runner
   BAT=$(su -c 'cat /sys/class/power_supply/battery/status' 2>/dev/null)
@@ -85,7 +88,7 @@
 trap 'on_signal 143' TERM; trap 'on_signal 129' HUP; trap 'on_signal 130' INT
 # one run at a time: runs share thermal.log, .stop_thermal, the handshake files, RobotCam and port 8080
 if ! mkdir "$L/.lock" 2>/dev/null; then
-  echo "Another co-residency launcher holds $L/.lock (pid $(cat "$L/.lock/pid" 2>/dev/null))."
+  echo "Another thermal launcher holds $L/.lock (pid $(cat "$L/.lock/pid" 2>/dev/null))."
   echo "If no run is active (for example after a crash), remove it: rm -r $L/.lock"; exit 1
 fi
 LOCKED=1
@@ -137,7 +140,7 @@
 # The runner runs as a job so TERM/HUP/INT sent to this launcher reach it: it stops RobotCam and llama-server
 # itself, and we wait for that before the EXIT trap restores the screen timeout.
 LAUNCHING=1
-env PYTHONUNBUFFERED=1 bash "$R/run_coresidency.sh" "$@" > >(tee -a "$CONSOLE") 2>&1 &
+env PYTHONUNBUFFERED=1 bash "$R/run_thermal_char.sh" "$@" > >(tee -a "$CONSOLE") 2>&1 &
 RUNNER=$!
 [ -z "$PENDING" ] || kill -TERM "$RUNNER" 2>/dev/null
 while kill -0 "$RUNNER" 2>/dev/null; do wait "$RUNNER"; done
--- benchmark/coresidency/run_coresidency.sh	2026-09-30 11:21:33.091557190 +0000
+++ benchmark/thermal_char/run_thermal_char.sh	2026-09-30 19:02:59.609818118 +0000
@@ -1,9 +1,9 @@
 #!/data/data/com.termux/files/usr/bin/bash
 # Native Termux only; started by oneshot.sh in this folder.
-# Usage: run_coresidency.sh [--smoke] [--resume RUN_DIR]
+# Usage: run_thermal_char.sh [--smoke] [--note TEXT]
 set -euo pipefail
 if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
   echo 'Run from native Termux, not Debian/proot.' >&2
   exit 2
 fi
-exec python -u /data/data/com.termux/files/home/robot/benchmark/coresidency/coresidency.py "$@"
+exec python -u /data/data/com.termux/files/home/robot/benchmark/thermal_char/thermal_char.py "$@"
```

## Design choices

- **Reuse:** `thermal_char.py` imports `coresidency.py` as a module. It reuses `camera_start`/`camera_stop` (RobotCam mode B rate 2 with retries), `Server` (server_manager `setup_q4` binary and flags in the same order, deferring a signal until the process is registered, a LIVE registry, killpg stop), `allowed_cpus`/`check_cores`/`wait_cores`, `exit_on_signal`, `require_native`, `read_frame` and `Detector`. No `motors.py` import and no USB/serial; the test asserts that neither `motors` nor any serial module is in `sys.modules`.
- **Layout:** discovered at start by globbing, never assumed. For every thermal zone: type and every `trip_point_N_temp`/`_type`. For every `cpufreq/policy*`: `related_cpus`, `cpuinfo_max/min_freq`, available frequencies, and `scaling_max_freq` at start. For every cooling device: type and `max_state`. The run refuses to start if a BIG/MID/LITTLE zone type or a policy's `cpuinfo_max_freq` is missing.
- **1 s samples:** one persistent `su` shell fed through a pipe, with no new su per sample. Each sample is a single script using shell builtins only (`read -r v <file`), so no process is spawned per file; an unreadable file gives an empty value (like zone6–8 on this phone). Each sample reads every zone temp, `scaling_cur_freq`/`scaling_max_freq` per policy, every cooling device `cur_state`, and battery temp (power_supply, tenths of degC), current, voltage and status. W = -(current_now × voltage_now)/1e12, as camera_heat.py computes it. After a timeout, the shell discards the late output of the timed-out command.
- **5 s dumps:** `dumpsys thermalservice` in the #123 form (`su -c "{ cmd; } </dev/null >/data/local/tmp/thermal_char_thermalservice.txt 2>&1; cat …"`). Parsed for `Thermal Status: N` and the "Current temperatures from HAL" section; NaN values are dropped and the names in the cached section are recorded too. Skin = `VIRTUAL-SKIN` from the HAL section only. The report lists every HAL name found and says whether VIRTUAL-SKIN was present. Raw text is kept for the first dump and for every dump whose text differs from the previous one. A dump is timestamped at completion, with its start kept (round-2 fix).
- **Load:** `detect_person.Detector` (both sessions resident) runs at 640 on every new frame. Frames go through `read_frame`'s fail-closed checks and the frame number must advance. llama-server runs with `setup_q4` flags and a streamed `/completion` back-to-back: one fixed raw-continuation prompt, n_predict 256, temperature 0, cache_prompt false. Tokens are timestamped as they stream. An interrupted request keeps its streamed tokens (round-1 fix).
- **Work rate:** per 10 s window, the report gives frames detected, the 640 detect ms median, and tok/s two ways: streamed tokens / window, and llama-server timings (sum predicted_n / sum predicted_ms of requests ending in the window).
- **Stops (first of):**
  - VIRTUAL-SKIN ≥ 48.0 °C;
  - battery ≥ 45.0 °C (power_supply temp);
  - Android status ≥ 5 (EMERGENCY);
  - the max over BIG/MID/LITTLE ≥ 110 °C in 3 consecutive samples in which all three zones were readable;
  - 20 min of load.

  Each limit uses only its own readings (round-1 fix).
- **Fail-closed stops (the run is marked INCOMPLETE and exits 1):**
  - no VIRTUAL-SKIN and status reading for 60 s, counted from max(last reading, load start);
  - no CPU-zone reading, or no battery-temperature reading, for 5 s;
  - no new RobotCam frame for 10 s;
  - a llama-server request failure or exit, or a detector exception;
  - cores 4–7 lost.

  A threshold already met when the 10 s baseline ends prevents the load. All stops also apply while llama-server and RobotCam are starting (round-2 fix): they are checked on every poll those two make, and a trip is recorded as `during startup:`.
- **After the stop:** RobotCam STOP, llama-server stop and worker join. `pidof com.pixelrobot.robotcam` and the server state are recorded in `run.json` `after_stop`; a leftover makes the run INCOMPLETE. Then 5 min of cooldown logging.
- **SIGTERM/SIGHUP:** SystemExit via `coresidency.exit_on_signal`, recorded as `aborted: …`. The `finally` stops every LIVE server, stops RobotCam, closes the root shell, writes the report and exits non-zero.
- **Output:** `~/thermal_char/run_<UTC>[_smoke]/` holds `samples_1s.jsonl`, `thermalservice_5s.jsonl`, `thermalservice_raw.jsonl`, `frames.jsonl`, `gen.jsonl`, `llama-server.log`, `run.json` (hashes of the runner, coresidency, reader, detector, server_manager and both ONNX models; GGUF size; server command, sample script and dump command; layout; thresholds; note; stop reasons; after_stop) and `report.txt`. The report is also copied to `~/storage/downloads/thermal_char_run_<UTC>[_smoke]_report.txt`. `--note` goes into run.json and the report.
- **report.txt sections:**
  - STOP REASON (at the top and at the end), and INCOMPLETE lines;
  - LAYOUT: policies with CPUs and cpuinfo_max, CPU zones with their trips, HAL names, counts and the largest sample gap;
  - EVENTS: each policy's scaling_max_freq steps, with the first drop below cpuinfo_max flagged; every Android status change; every cooling-device change. Each event comes with skin, status, battery °C/W/status, cur/max freq per policy, all zone temps and all HAL temps;
  - TIMELINE every 30 s (skin, battery, BIG/MID/LITTLE, quiet_therm, max MHz per policy, status, 640 ms, tok/s both ways, W);
  - WORK RATE per 10 s;
  - SLOWDOWN, first vs last minute;
  - COOLDOWN: time for skin and BIG to return within 2 °C of their start values, or "not reached". Start values are the last readings before llama-server starts;
  - Notes. Measurements only; no thresholds are recommended.
- **`--smoke`:** 60 s of load, 30 s of cooldown, the same stops.
- **Not done (non-blocking reviewer note, all 3 rounds):** no GGUF SHA-256. Hashing 2.8 GB right after the launcher's idle would warm the phone before the baseline; the size is recorded instead. Left for Local AI to decide.

## Open blocking finding (round 3, not fixed)

Each 1 s sample's `t` is taken **before** the root shell reads the sensors (`take_sample`, thermal_char.py:217). If a read is slow, the report can place frequency and cooling events, and the readings shown beside a status change, earlier than when they were observed. The fix would mirror the round-2 dump fix: stamp the sample on completion and keep its start. It was **not** applied, because any change after round 3 needs a new review round and the task allows at most 3. Local AI and the human decide whether to authorize a fourth round.

## Check output (Debian/proot)

```
bash -n benchmark/thermal_char/oneshot.sh ok
bash -n benchmark/thermal_char/run_thermal_char.sh ok
bash -n benchmark/thermal_char/test_oneshot.sh ok
/usr/bin/python3 3.13.5 py_compile ok
/data/data/com.termux/files/usr/bin/python 3.13.13 py_compile ok
--- /data/data/com.termux/files/usr/bin/python test_thermal_char.py — final files (exit 0):
ok: parse_dump and every stop condition at its edge
ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, missing CPU zone refused
ok: interrupted request keeps its 5 streamed tokens; without a load stop it is a load failure
ok: a slow thermalservice dump is stamped at completion (start kept)
ok: duration: stop "planned load duration 8 s reached" after 8.1 s of load, exit 0
ok: skin: stop "VIRTUAL-SKIN 48.0 >= 48.0 degC" after 2.7 s of load, exit 0
ok: battery: stop "battery 45.0 >= 45.0 degC" after 3.0 s of load, exit 0
ok: status: stop "Android thermal status 5 >= 5 (EMERGENCY)" after 3.2 s of load, exit 0
ok: cpu: stop "CPU zone >= 110 degC in 3 consecutive 1 s samples (max [110.0, 110.0, " after 3.7 s of load, exit 0
ok: noskin: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.1 s of load, exit 1
ok: skinlost: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.1 s of load, exit 1
ok: sensorfail: stop "fail closed: no battery temperature reading for 6 s" after 6.1 s of load, exit 1
ok: noframes: stop "load failed: no new RobotCam frame for 2 s" after 4.9 s of load, exit 1
ok: serverdie: stop "load failed: llama-server request failed: RuntimeError: stream ended w" after 2.3 s of load, exit 1
ok: cores: stop "cores 4-7 lost (allowed [0, 1, 2, 3, 4, 5])" after 2.4 s of load, exit 1
ok: startup: stop "during startup: VIRTUAL-SKIN 48.5 >= 48.0 degC", exit 1
ok: detectfail: stop "load failed: frame/detect loop failed: ValueError: fake detector failu" after 3.2 s of load, exit 1
ok: sigterm: stop "aborted: SystemExit: 143" after 1.5 s of load, exit 143
ALL OK
--- same test, two further runs before the final docstring/RUN.md-only edit: both ALL OK (exit 0)
--- bash test_oneshot.sh (exit 0):
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run
ok: refuses with a thermal runner running
ok: refuses while the co-residency launcher holds its lock
ok: note with spaces reaches the runner
```

The offline test covers:
- unit checks: dump parsing, including a status change, a missing skin sensor, NaN values and a failed service call; every stop condition at its edge (47.9/48.0, 44.9/45.0, status 4/5, 2 vs 3 hot samples, the 60 s skin rule, per-input sensor fail-closed);
- the persistent root shell, including timeout recovery;
- discovery on a fake sysfs, and refusal without BIG;
- sample parsing with an unreadable zone;
- an interrupted streamed request (kept, and counted as a load failure unless the load had stopped);
- a slow dumpsys stamped at completion;
- 15 end-to-end scenarios with fake su/dumpsys/sysfs/frames/streaming server/detector: duration (checking the EVENTS, TIMELINE, WORK RATE, SLOWDOWN and COOLDOWN contents), skin, battery, status, cpu, noskin, skinlost, sensorfail, noframes, serverdie, cores, startup, detectfail and sigterm.

Each scenario checks the stop reason, exit code, INCOMPLETE marking, the note, the Downloads copy, that no fake llama-server is left, that RobotCam was stopped last, that `after_stop` is clean and that motors were not imported.

## Unverified (no native run was made; no native results are claimed)

- Magisk `su` as a persistent piped shell, and mksh `read` from sysfs files.
- The real `dumpsys thermalservice` format on this phone, and whether VIRTUAL-SKIN appears in "Current temperatures from HAL". The skin-freshness fail-closed stop covers the case where it does not.
- The b10194 llama-server stream format (a final chunk carrying `stop` and `timings`), and whether each streamed chunk is exactly one token.
- The real cooling-device count and change rate (event volume), and 1 s sampling duration with ~33 zones.
- The launcher on the phone: only its fake-based test ran.

## Environment notes

- During this session, seven `security_reminder_hook.py` processes (Claude Code security-guidance plugin hooks) were stuck at ~97% CPU each for over 25 minutes. They slowed proot, stalled 1 s reads by up to ~4.5 s, and caused test timing flakiness (the test timings were widened accordingly). I did not kill them. PIDs at the time: 18466 18468 18472 18473 18474 18475 18486. Worth checking before the native run: the launcher refuses if `claude|node` processes are running anyway.
- A first test run interrupted by a session break left 4 orphaned test children (fakes only, no hardware). They were killed by PID and their temp directory removed.

## Review rounds

- Round 1: **CHANGES REQUESTED**. Two blocking findings: the CPU stop required "complete" samples; streamed tokens of an interrupted request were dropped. Both fixed.
- Round 2: **CHANGES REQUESTED**. Two blocking findings: no stops during startup; dumps stamped before dumpsys ran. Both fixed.
- Round 3: **CHANGES REQUESTED**. One blocking finding (1 s sample stamped before the read). Not fixed; stopped per the 3-round limit.

### Round 1 review (verbatim)

```
**BLOCKING** — [thermal_char.py](/termux-home/robot/benchmark/thermal_char/thermal_char.py:209) makes a sample “complete” only when battery power and every policy’s maximum frequency are readable. The CPU heat stop at [line 297](/termux-home/robot/benchmark/thermal_char/thermal_char.py:297) then requires three *complete* samples. If an unrelated power or frequency read fails every third sample, a CPU zone can stay above 110 °C indefinitely without meeting that stop or the five second missing-sample stop. Evaluate each heat limit from its own required sensor readings.

**BLOCKING** — [thermal_char.py](/termux-home/robot/benchmark/thermal_char/thermal_char.py:345) retains streamed token arrivals only until a 256-token request finishes. Stopping the load kills the server and discards arrivals from the request in progress ([lines 366–378](/termux-home/robot/benchmark/thermal_char/thermal_char.py:366)). The report can therefore show zero streamed tokens per second for a valid thermal-stop run that generated tokens, reporting wrong work rates as valid. Persist arrivals as they occur, including for an interrupted request.

**NON-BLOCKING** — [run.json creation](/termux-home/robot/benchmark/thermal_char/thermal_char.py:631) records the GGUF size but no GGUF hash, limiting reproducibility of the reported model setup. The supplied checks are offline; native sensor and thermalservice behavior remains unverified, as the task requires.

VERDICT: CHANGES REQUESTED.
```

### Round 2 review (verbatim)

```
**BLOCKING** — [thermal_char.py](/termux-home/robot/benchmark/thermal_char/thermal_char.py:668) starts llama-server, then RobotCam, before entering the stop loop at line 679. Server startup may wait up to 180 seconds. During that interval, the loggers run but no skin, battery, Android status, or CPU limit can stop the load. Apply the thermal stops throughout startup.

**BLOCKING** — [thermal_char.py](/termux-home/robot/benchmark/thermal_char/thermal_char.py:252) timestamps each thermalservice dump *before* running `dumpsys`, then uses that timestamp for status events and their accompanying sensor readings. A slow dump can place a status change and its reported “at that moment” readings seconds before they were observed, while presenting them as valid measurements. Use the completion time or report the observation interval, and align the accompanying readings with it.

**NON-BLOCKING** — `run.json` records the GGUF size but no GGUF hash, limiting reproducibility. The supplied checks are offline; I did not run hardware or create files during this review.

The round 1 CPU stop and interrupted-token fixes are present. I inspected the candidate and the shared camera, detector, server, and launcher paths.

VERDICT: CHANGES REQUESTED.
```

### Final review — round 3 (verbatim)

```
**BLOCKING** — [thermal_char.py](/termux-home/robot/benchmark/thermal_char/thermal_char.py:217) stamps each one-second sample before the root shell reads the sensors. A read can take several seconds, yet the report uses that start time for frequency and cooling events and for the sensor values shown beside Android status changes ([line 465](/termux-home/robot/benchmark/thermal_char/thermal_char.py:465)). It can therefore present readings obtained after a status change as values measured at that event. Record the sample’s completion time and observation interval, as the thermalservice logger now does.

**NON-BLOCKING** — `run.json` records GGUF size but no GGUF hash, limiting reproducibility. The supplied verification is offline; native behavior remains unverified.

The round 2 startup-stop and thermalservice timestamp fixes are present. I inspected the on-disk candidate and shared camera, detector, server, and launcher paths. No files were changed and no hardware was run.

VERDICT: CHANGES REQUESTED.
```

Reviewer stderr diagnostics were kept separately in the session scratchpad (review_r{1,2,3}.err) and are not part of the reviews.
