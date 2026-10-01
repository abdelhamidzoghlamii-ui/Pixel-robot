# CORESIDENCY FIX 1 — Coder report

**Role:** Coder. Claude Code CLI, `claude-opus-5-5`, medium effort, Ponytail lite.
**Reviewer:** Codex CLI 0.158.0, `gpt-6-sol`, medium effort, read-only sandbox, Ponytail off, a fresh `codex exec` session per round. Command: `codex exec -m gpt-6-sol -c model_reasoning_effort="medium" -s read-only -C /termux-home/robot - < request.md > stdout.txt 2> stderr.txt`
**Base:** `main` at `3584b5eddd3639c00c37b24e6e31f4d86fce2f2b`. I checked HEAD, the clean tracked tree and all six frozen hashes at the start, and all matched. HEAD and the clean tracked tree still hold at the end. Nothing was staged, committed or pushed.

## Outcome: STOPPED. Round 3 verdict is REQUEST CHANGES (one MAJOR finding is still open)

The task allows 3 review rounds. Round 3 was not APPROVE or APPROVE WITH NOTES, so I stopped. I made **no edits after round 3**, and the files on disk equal what round 3 reviewed (`review3/frozen.sha`; `sha256sum` diff is empty).

**Open finding (round 3, MAJOR, not fixed):** An immediate heat stop still makes the run INCOMPLETE. In `run_block`, when the stop comes before the first processed frame, `last` is `None`, so `survived.robotcam_new_frame_at_end` is `False`. `problems()` then flags the RobotCam rule even though the `at_start` exemption applies. My immediate-stop unit test passed `survived=True` by hand, so it did not exercise this path. The native smoke run can reach this path: its B3 started at 81 °C with no gate.
Suggested fix, not applied: when `last is None`, the survival baseline becomes "any ok frame of the session newer than `cam['frame']`" (the frame `camera_start` saw). The test should then build that block through `run_block` with a hot log from t = 0. This change needs a new review round, which only a new instruction can allow.

This only affects **smoke runs without the thermal gate**. The full run gates each block to z9 ≤ idle + 4 °C, so a stop before the first frame is unlikely there. It would then show as INCOMPLETE, which is a false negative. It does not report wrong numbers as valid.

## Causes (with evidence)

### 1. Selector cadence
- Evidence: `block_B3_mix_gemma.json` has `duration_s 20.25` and `selector_calls [{t 0.0, started_s 0.0, ended_s 1.97, ms 1962.0}]`. B4 has `20.24` and one call ending at 2.25 s.
- Cause: the frozen runner computed `slots = ceil(b['duration_s'] / 20)`. `duration_s` is measured after `frame_loop` returns, so it is always slightly over the planned length: ceil(20.25/20) = **2**. `selector_loop` only starts slots with `k*20 < planned` (k = 0 only for 20 s), so 2 successful calls could never happen.
- **180 s block:** the old code expected ceil(180.x/20) = **10**, but only 9 slots (0…160) can start. **The real run would also have failed every Gemma block.** After the fix it expects **9**, and the smoke run expects **1**.
- Fix: `slots = ceil(end / 20)`. `end` is the new block field `planned_s`. After a heat stop it is `time_to_limit_s − 1` (after round 2).

### 2. Heat stop
- The live pause is `main.Robot.run_mission`, which `run_mission.py` calls: `if get_temp() > 80:` pause 5 s. Here `main.get_temp()` = `su -c "cat /sys/class/thermal/thermal_zone9/temp"` // 1000, in whole °C. `thermal_guard.py` (WARN 82 / CRIT 86 over max(z9, z10, z11)) has no caller on the robot's code path.
- The runner reads the zone and the threshold from `main.py` with two regexes at import. It exits if either is not found, and records `main.py`'s sha256 and `heat_stop {zone, above_c}` in `run.json`. Before every frame read it checks the newest thermal-log line (oneshot's root logger, every 5 s). On `z9 // 1000 > 80` it ends the frame loop.
- The block JSON gets `planned_s` and `heat_stop {zone, above_c, reached_limit, time_to_limit_s, zone_reading}`. `stop.set()` now runs right after the frame loop, so no selector slot starts after a stop.
- In a heat-stopped block:
  - the selector call still in flight at the stop is not an overrun;
  - drift is n/a when the block ran under 60 s;
  - the mix-640 rule is skipped before 5 s;
  - after round 2: a stop before the first frame skips the frame and per-size rules, a stop under 5 s skips "no usable samples" and an empty battery status, and the stop reading counts in the z9 max.
- The report has a new row, `time to limit s` (a number or `-`). The thermal gate is unchanged.
- Documented ceiling: `time_to_limit_s` is when the runner saw the limit, up to about 5 s plus one detection after the zone crossed it.

### 3. Cold load
- Evidence:
  - `loads.json`: `"cached_mib_before": 2413, "cached_mib_after": 1223, "fallback_reason": "handshake answer 'ok', cached 2413 -> 1223 MiB"`.
  - Console: `[oneshot] cache drop: ok`.
  - The check needed `after < 0.5 * before` = 1206.5 MiB, and **missed by 17 MiB**.
  - Earlier ladder full-cold drops left 567, 673, 829 and 987 MiB (`/termux-home/ladder/*/results.json`).
- Cause: this was **a bug in the runner's check, not a failed drop**. The drop freed 1190 MiB. But `Cached` also counts pages `drop_caches` cannot free (shmem, pages mapped by running processes). That floor does not scale with the "before" value, so a ratio test fails a real drop whenever "before" is low.
- Fix: the load is labelled full-cold only if the handshake answered `ok` **and** `mincore(2)` finds **0** cached pages in every file the cold load reads: the GGUF, `llama-server` and the `lib*.so` beside it (its RUNPATH). One cached page is enough to fall back to weights-cold (with the fadvise, as before). A mincore failure also falls back, and so does a file the kernel will not report on, because the kernel then reports every page resident. `Cached` before and after are still recorded. This test is stricter about what "cold" means; it is not a looser threshold.
- **Not verified natively:** I cannot show that the smoke run's drop left 0 pages of these files cached. The next native run's `resident_pages_after` will show it.

## Review rounds
| Round | Verdict | Findings → action |
|---|---|---|
| 1 | REQUEST CHANGES | MAJOR: `--resume` of the old smoke run crashed in `report()` because old blocks lack `heat_stop` and `planned_s`. **Fixed:** `--resume` refuses runs made by the older runner, and a test covers it. MINOR: the stop latency can exceed 5 s. **Documented** in a code comment and the report notes. |
| 2 | REQUEST CHANGES | MAJOR: an immediate heat stop was INCOMPLETE because of the frames, samples and selector rules. **Fixed** with the at_start and no_sample exemptions, and the selector slots now need to be due 1 s before the stop. MAJOR: the z9 max could leave out the stop reading. **Fixed.** Tests added for both. |
| 3 | **REQUEST CHANGES** | MAJOR: the RobotCam survival rule still fails an immediate stop (see above). **Not fixed**, because the round limit was reached. |

The reviewer sandbox had some of its own shell commands exit with `182`, as in the previous task: 4, 7 and 6 per round, next to 16, 12 and 11 that succeeded. There were no authentication, quota or model errors. The requests contained the complete numbered files, the diff and the check output. The rounds' requests, stdout and stderr are in `~/storage/downloads/coresidency_fix1_review_artifacts/`.

## Unverified
- Nothing ran natively. There is no native evidence for the heat stop on real zone readings, for `mincore` on Android/bionic through `ctypes.CDLL(None)` (the offline test ran it with Termux python inside proot), for the real `resident_pages_after`, or for the cadence on the phone.
- The open round-3 finding: an immediate heat stop in a smoke run will still report INCOMPLETE through the RobotCam survival rule.
- `SMOKE.md` is unchanged, because the smoke command did not change. It still says to check that the cold mode is `full-cold`.
- Lazier alternative, stated once: for the cold check, a simple absolute floor on `Cached` would have been ~1 line, but it would be another arbitrary threshold. The mincore check is about 20 lines.

## Files (final, as reviewed in round 3)
```
3e3cd4e21f476d32e5784ebf51b65b9076cb5cd1b6b233688033d3044cdbabae  SMOKE.md
65c572e01826525e5f26881e9d5167145beb430a07e41f6ff15b37c58d1cf60d  coresidency.py
efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2  oneshot.sh
dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0  run_coresidency.sh
6ea51d6ead34926759c1b5b654e71fa169a4d13ffed0e410400fda3d30b2dd17  test_coresidency.py
bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19  test_oneshot.sh
 2882 SMOKE.md
49674 coresidency.py
 7351 oneshot.sh
  426 run_coresidency.sh
29971 test_coresidency.py
 8056 test_oneshot.sh
98360 total
```
Unchanged from the frozen candidate: `oneshot.sh`, `run_coresidency.sh`, `test_oneshot.sh`, `SMOKE.md`. Changed: `coresidency.py` (was `ef622037…`) and `test_coresidency.py` (was `e076915d…`).

`git rev-parse HEAD` / `git status --short --untracked-files=no`:
```
3584b5eddd3639c00c37b24e6e31f4d86fce2f2b
(end)
```

## Diff (frozen candidate → final)
```diff
--- frozen/coresidency.py
+++ current/coresidency.py
@@ -12,6 +12,7 @@
                     letter-scoring call every 20 s in its own thread (s1o, variant s1o_b1609dp_q40 prompts)
   B4_only640_gemma  B2 + the same
   B5_mtp_ram_snapshot  (not timed) Gemma with the conv_mtp drafter flags, one short request, RAM snapshot
+B1-B4 end early at the robot's live thermal pause (zone and threshold read from main.py): a result, not a failure.
 
 Every 5 s: MemAvailable/swap, PSS (root dumpsys meminfo) of the runner, llama-server, RobotCam app and
 camera provider, zone9/10/11 from the thermal log, battery power (camera_heat.py's method). Per block:
@@ -64,6 +65,15 @@
 SAMPLE_S, SELECTOR_S, DRIFT_S = 5, 20, 30
 BLOCKS = [('B1_mix_nollm', 'mix', False), ('B2_only640_nollm', '640', False),
           ('B3_mix_gemma', 'mix', True), ('B4_only640_gemma', '640', True)]
+LARGE_S = 5  # main.LARGE_FRAME_INTERVAL_S: drive mode runs one 640 frame every 5 s
+# Heat stop: the robot's live thermal pause, main.Robot.run_mission `if get_temp() > 80:` with get_temp() reading
+# thermal_zone9 in whole degC (millidegC // 1000). Both values are read from main.py, never restated here.
+_MAIN = (ROBOT / 'main.py').read_text()
+_zone = re.search(r'def get_temp\(\):\n.*?thermal_zone(\d+)/temp', _MAIN, re.S)
+_limit = re.search(r'if get_temp\(\) > (\d+):', _MAIN)
+if not (_zone and _limit) or _zone.group(1) not in ('9', '10', '11'):
+    raise SystemExit('main.py: the live thermal pause (get_temp zone, get_temp() > N) was not found')
+PAUSE_ZONE, PAUSE_ABOVE_C = f'z{_zone.group(1)}', int(_limit.group(1))
 DEFAULT_INSTRUCTION = 'Choose the single best next action for the robot.'  # ladder.py
 WARMUP = ('The robot is idle in the hallway.', {'wait': '', 'explore': ''}, 'Pick one.')  # ladder_worker.py
 MTP_PROMPT = 'In one short sentence, what does a home robot do?'
@@ -267,16 +277,51 @@
         request.unlink(missing_ok=True)
         done.unlink(missing_ok=True)
     after = meminfo_mib()['cached_mib']
-    if answer == 'ok' and after < 0.5 * before:
-        return {'mode': 'full-cold (page cache dropped by the handshake watcher)', 'cached_mib_before': before,
-                'cached_mib_after': after}
+    # Cached keeps what drop_caches cannot free (shmem, mapped pages), so a Cached ratio is no test of the drop;
+    # the test is that no page of what the cold load reads (GGUF, llama-server, its build libs) is still cached.
+    try:
+        resident = resident_pages(cold_files())
+    except (OSError, AttributeError, ValueError) as e:
+        resident = f'{type(e).__name__}: {e}'
+    state = {'cached_mib_before': before, 'cached_mib_after': after, 'resident_pages_after': resident}
+    if answer == 'ok' and resident == 0:
+        return {'mode': 'full-cold (page cache dropped by the handshake watcher)', **state}
     fd = os.open(MODEL, os.O_RDONLY)
     try:
         os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
     finally:
         os.close(fd)
-    return {'mode': 'weights-cold (posix_fadvise DONTNEED on the GGUF)', 'cached_mib_before': before,
-            'cached_mib_after': after, 'fallback_reason': f'handshake answer {answer!r}, cached {before} -> {after} MiB'}
+    return {'mode': 'weights-cold (posix_fadvise DONTNEED on the GGUF)', **state,
+            'fallback_reason': f'handshake answer {answer!r}, {resident} load-file pages still cached, '
+                               f'Cached {before} -> {after} MiB'}
+
+
+def cold_files():
+    """What a llama-server start reads from disk: the GGUF, the binary and the libraries beside it (RUNPATH)."""
+    binary = Path(server_manager.LLAMA_SERVER)
+    return sorted({Path(MODEL), binary, *(p.resolve() for p in binary.parent.glob('lib*.so*'))})
+
+
+def resident_pages(paths):
+    """Page-cache pages of these files, by mincore(2) on a read-only shared mapping (mapping reads nothing).
+    The kernel reports every page as resident for a file the caller neither owns nor may write: fails closed."""
+    import ctypes
+    import mmap
+    import numpy as np
+    libc = ctypes.CDLL(None, use_errno=True)
+    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
+    total = 0
+    for p in paths:
+        size = os.path.getsize(p)
+        with open(p, 'rb') as fh, mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ) as m:
+            view = np.frombuffer(m, dtype=np.uint8)
+            vec = ctypes.create_string_buffer((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE)
+            rc = libc.mincore(view.ctypes.data, size, vec)
+            del view  # the mapping cannot close while exported
+            if rc != 0:
+                raise OSError(ctypes.get_errno(), f'mincore {p}')
+            total += sum(b & 1 for b in vec.raw)
+    return total
 
 
 # ---------------------------------------------------------------- llama-server and selector
@@ -451,11 +496,26 @@
 robotcam_reader.io = _Stamp
 
 
-def frame_loop(detector, mode, session, t0, duration, reads):
-    """Reads every new frame and detects on it; returns the last frame number processed."""
-    policy = SizePolicy(5, 1/3)  # main.LARGE_FRAME_INTERVAL_S, PERSON_HEIGHT_FRACTION; drive mode
+def heat_reached(thermal_log):
+    """The live pause test on the newest thermal-log line: the reading if it is reached, else None. A line that
+    cannot be parsed is retried next frame (the sampler records it as a thermal error); a stale log still stops."""
+    try:
+        t = read_thermal(thermal_log)
+    except (OSError, ValueError, IndexError, KeyError):
+        return None
+    return t if t[PAUSE_ZONE] // 1000 > PAUSE_ABOVE_C else None
+
+
+def frame_loop(detector, mode, session, t0, duration, reads, thermal_log):
+    """Reads every new frame and detects on it until the block ends or the heat stop is reached.
+    Returns (last frame number processed, heat stop record or None)."""
+    policy = SizePolicy(LARGE_S, 1/3)  # main.LARGE_FRAME_INTERVAL_S, PERSON_HEIGHT_FRACTION; drive mode
     last = None
     while time.monotonic() < t0 + duration:
+        # ponytail: checked between frames on the 5 s thermal log, so the stop (time_to_limit_s = when the runner
+        # saw it) can come up to ~5 s plus one detection after the zone crossed; a live su read would cut that
+        if (hot := heat_reached(thermal_log)) is not None:
+            return last, {'time_to_limit_s': round(time.monotonic() - t0, 2), 'zone_reading': hot}
         check_cores('during the block')
         _Stamp.at = None
         a = time.perf_counter()
@@ -480,7 +540,7 @@
         row.update(size=size, detect_ms=round((time.perf_counter() - c) * 1000, 2), n_detections=len(detections))
         # rate 2: the next frame appears about 0.5 s after this one was first seen (a repeat retries in 20 ms)
         time.sleep(max(0.0, a + 0.5 - time.perf_counter()))
-    return last
+    return last, None
 
 
 # ---------------------------------------------------------------- blocks
@@ -505,8 +565,9 @@
     for t in threads:
         t.start()
     try:
-        last = frame_loop(ctx['detector'], mode, cam['session'], t0, duration, reads)
+        last, heat = frame_loop(ctx['detector'], mode, cam['session'], t0, duration, reads, ctx['thermal_log'])
         elapsed = time.monotonic() - t0
+        stop.set()  # a heat stop ends the selector's cadence here too (no slot starts after the block)
         deadline = time.monotonic() + 2  # survival: a newer frame within 2 s (rate 2)
         while True:
             end = read_frame(FRAME_DIR, session=cam['session'])
@@ -523,7 +584,9 @@
         raise RuntimeError(f'{name}: a sampler/selector thread still running 120 s after the block; run stopped')
     check_cores('after ' + name)
     therm_end = read_thermal(ctx['thermal_log'])
-    return {'block': name, 'mode': mode, 'gemma': gemma, 'duration_s': round(elapsed, 2), 'camera': cam,
+    return {'block': name, 'mode': mode, 'gemma': gemma, 'duration_s': round(elapsed, 2), 'planned_s': duration,
+            'heat_stop': {'zone': PAUSE_ZONE, 'above_c': PAUSE_ABOVE_C, 'reached_limit': heat is not None,
+                          **(heat or {'time_to_limit_s': None, 'zone_reading': None})}, 'camera': cam,
             'thermal_start': therm_start, 'thermal_end': therm_end,
             'survived': {'robotcam_process': bool(robotcam_pid), 'robotcam_pids': robotcam_pid,
                          'robotcam_new_frame_at_end': end['status'] == 'ok' and last is not None and end['frame'] > last,
@@ -609,12 +672,15 @@
             failed[r['status']] = failed.get(r['status'], 0) + 1
     last_t = b['duration_s']
     s = {'frames': len(done), 'failed': failed, 'repeats': sum(r['status'] == 'repeat' for r in b['reads'])}
+    # a heat stop before both drift windows fit: drift is n/a, not missing
+    s['drift_na'] = b['heat_stop']['reached_limit'] and last_t < 2 * DRIFT_S
     for size in (320, 640):
         ms = [r['detect_ms'] for r in done if r['size'] == size]
         s[f'n{size}'] = len(ms)
         s[f'det{size}'] = (med(ms), p95(ms))
-        s[f'drift{size}'] = (med([r['detect_ms'] for r in done if r['size'] == size and r['t'] < DRIFT_S]),
-                             med([r['detect_ms'] for r in done if r['size'] == size and r['t'] >= last_t - DRIFT_S]))
+        s[f'drift{size}'] = (None, None) if s['drift_na'] else (
+            med([r['detect_ms'] for r in done if r['size'] == size and r['t'] < DRIFT_S]),
+            med([r['detect_ms'] for r in done if r['size'] == size and r['t'] >= last_t - DRIFT_S]))
     s['read'] = med([r['read_ms'] for r in done])
     s['decode'] = med([r['decode_ms'] for r in done])
     s['age'] = med([r['age_s'] for r in done])
@@ -627,7 +693,8 @@
     s['max_swap'] = max((x['swap_used_mib'] for x in sm), default=None)
     for name in ('runner', 'llama_server', 'robotcam_app', 'camera_provider'):
         s['pss_' + name] = max((x['pss_kb'][name] for x in sm if name in x.get('pss_kb', {})), default=None)
-    z9 = [b['thermal_start']['z9'], b['thermal_end']['z9']] + [x['z9'] for x in sm if 'z9' in x]
+    z9 = [b['thermal_start']['z9'], b['thermal_end']['z9']] + [x['z9'] for x in sm if 'z9' in x] + \
+         ([b['heat_stop']['zone_reading']['z9']] if b['heat_stop']['reached_limit'] else [])  # the stop reading
     s['z9'] = (b['thermal_start']['z9'] / 1000, b['thermal_end']['z9'] / 1000, max(z9) / 1000)
     w = [x['battery_w'] for x in sm if 'battery_w' in x]
     s['w'] = statistics.mean(w) if w else None
@@ -644,30 +711,44 @@
     """Why the run cannot count as a complete measurement (blocks are still kept: a kill is evidence)."""
     out = []
     for b, s in zip(blocks, sums):
-        if b['gemma'] and s['sel'][0] == 0:
-            out.append(f'{b["block"]}: no successful selector call ({s["sel"][1]} failed)')
-        elif b['gemma']:  # the Gemma workload is one call per 20 s slot, each inside the block
+        heat = b['heat_stop']
+        # a heat stop is a result: rules that need time only apply as far as the block ran (at_start: stopped
+        # before its first frame; no_sample: before a second sample was due, so the first may finish after it)
+        at_start = heat['reached_limit'] and s['frames'] == 0 and not s['failed']
+        no_sample = heat['reached_limit'] and b['duration_s'] < SAMPLE_S
+        if b['gemma']:  # the Gemma workload is one call per 20 s slot, each inside the block
             calls = b['selector_calls']
-            slots = math.ceil(b['duration_s'] / SELECTOR_S)
+            # slots that can start inside the block: k * SELECTOR_S < planned length (selector_loop's bound; the
+            # block itself runs a little past it), or before the heat stop, which also stops the selector; a slot
+            # due in the last second before a heat stop is not required (the stop may beat the thread to it)
+            end = heat['time_to_limit_s'] - 1 if heat['reached_limit'] else b['planned_s']
+            slots = max(0, math.ceil(end / SELECTOR_S))
             late = [c['t'] for c in calls if c['started_s'] - c['t'] > 5]
-            over = [c['t'] for c in calls if c['ended_s'] > b['duration_s']]
-            if s['sel'][0] < slots or late or over:
+            # a call in flight at a heat stop ends after it by construction: not an overrun
+            over = [] if heat['reached_limit'] else [c['t'] for c in calls if c['ended_s'] > b['duration_s']]
+            if slots and s['sel'][0] == 0:
+                out.append(f'{b["block"]}: no successful selector call ({s["sel"][1]} failed)')
+            elif s['sel'][0] < slots or late or over:
                 out.append(f'{b["block"]}: selector cadence missed: {s["sel"][0]}/{slots} successful calls, '
                            f'started >5 s late at slots {late}, ended after the block at slots {over}')
-        if s['sample_errors'] or s['min_avail'] is None:
+        if s['sample_errors'] or (s['min_avail'] is None and not no_sample):
             out.append(f'{b["block"]}: {s["sample_errors"]} sample(s) with root/PSS/thermal errors, '
                        f'{0 if s["min_avail"] is None else "some"} usable samples')
         sv = b['survived']
-        if s['failed'] or s['frames'] == 0 or not (sv['robotcam_new_frame_at_end'] and sv['robotcam_process']):
+        if s['failed'] or (s['frames'] == 0 and not at_start) or \
+                not (sv['robotcam_new_frame_at_end'] and sv['robotcam_process']):
             out.append(f'{b["block"]}: RobotCam: {s["frames"]} frames, failed reads {s["failed"] or "none"}, '
                        f'new frame at end {sv["robotcam_new_frame_at_end"]}, process at end {sv["robotcam_process"]}')
         for size in (320, 640) if b['mode'] == 'mix' else (640,):
-            if s[f'n{size}'] == 0 or None in s[f'drift{size}']:
+            if at_start or (size == 640 and b['mode'] == 'mix' and heat['reached_limit'] and
+                            heat['time_to_limit_s'] < LARGE_S):
+                continue  # stopped before this size's first frame was due
+            if s[f'n{size}'] == 0 or (None in s[f'drift{size}'] and not s['drift_na']):
                 out.append(f'{b["block"]}: {s[f"n{size}"]} detections at {size}, drift first/last {DRIFT_S} s '
                            f'{s[f"drift{size}"]} (a required measurement is missing)')
         if b['gemma'] and b['survived']['llama_server'] is not True:
             out.append(f'{b["block"]}: llama-server did not survive the block (see LMK lines)')
-        if s['bat_status'] != ['Discharging']:
+        if s['bat_status'] != ['Discharging'] and not (no_sample and s['bat_status'] == []):
             out.append(f'{b["block"]}: battery status {s["bat_status"]} (power needs Discharging throughout)')
         if not b['lmk']['ok']:
             out.append(f'{b["block"]}: LMK logcat query failed (rc {b["lmk"]["logcat_rc"]}): {b["lmk"]["raw_head"][:120]!r}')
@@ -702,6 +783,7 @@
         ('survived RobotCam / llama', None),
         ('zone9 start/end/max degC', lambda s: '/'.join(f'{v:.1f}' for v in s['z9'])),
         ('gate wait s', None),
+        ('time to limit s', None),
         ('mean battery W', lambda s: f(s['w'], '.2f')),
         ('battery status', lambda s: ','.join(s['bat_status']) or 'n/a'),
         ('sample errors', lambda s: str(s['sample_errors'])),
@@ -712,7 +794,9 @@
                                                       else f'QUERY FAILED rc {b["lmk"]["logcat_rc"]}'),
              'survived RobotCam / llama': lambda b: (f'{"yes" if b["survived"]["robotcam_new_frame_at_end"] and b["survived"]["robotcam_process"] else "NO"} / '
                                                      f'{ {True: "yes", False: "NO", None: "-"}[b["survived"]["llama_server"]]}'),
-             'gate wait s': lambda b: str(b['thermal_start']['waited_s'])}
+             'gate wait s': lambda b: str(b['thermal_start']['waited_s']),
+             'time to limit s': lambda b: (f'{b["heat_stop"]["time_to_limit_s"]:.1f}' if b['heat_stop']['reached_limit']
+                                           else '-')}
     run = json.loads((out / 'run.json').read_text())
     lines = [f'Co-residency benchmark (DECISIONS #124), run {out.name}{"  [SMOKE: not a measurement]" if run["smoke"] else ""}',
              f'block length {run["block_s"]} s; idle z9 {run["idle"]["z9"] / 1000:.1f} degC; runner sha256 {run["sha256"]["coresidency.py"][:12]}']
@@ -742,7 +826,10 @@
                   'BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from '
                   'battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines '
                   'matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", '
-                  'or killinfo. Every sample runs one su call (dumpsys) in all blocks alike.']
+                  'or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Heat stop = the robot\'s '
+                  f'live pause ({PAUSE_ZONE} > {PAUSE_ABOVE_C} degC, main.Robot.run_mission), checked between frames '
+                  'on the 5 s thermal log; time to limit = when the runner saw it (up to ~5 s plus one detection late); '
+                  f'the block ends there and counts as run; drift is n/a if it ran under {2 * DRIFT_S} s.']
     return '\n'.join(lines), bad
 
 
@@ -762,6 +849,8 @@
         run = json.loads((out / 'run.json').read_text())
         if run['smoke'] != a.smoke:
             raise SystemExit(f'--resume: {out} was a {"smoke" if run["smoke"] else "full"} run')
+        if 'heat_stop' not in run:  # its blocks lack heat_stop/planned_s and were judged by the old cadence count
+            raise SystemExit(f'--resume: {out} was made by an older runner (no heat stop); start a new run')
     else:
         out = OUT_ROOT / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}{"_smoke" if a.smoke else ""}'
         out.mkdir(parents=True)
@@ -776,12 +865,14 @@
         print(f'thermal idle reading: z9 {ctx["idle"]["z9"] / 1000:.1f} degC; output {out}', flush=True)
         files = {'coresidency.py': __file__, 'robotcam_reader.py': robotcam_reader.__file__,
                  'detect_person.py': ROBOT / 'detect_person.py', 'detector_size_policy.py': ROBOT / 'detector_size_policy.py',
-                 'server_manager.py': server_manager.__file__, 'adapters.py': V3 / 'adapters.py',
+                 'server_manager.py': server_manager.__file__, 'main.py': ROBOT / 'main.py',
+                 'adapters.py': V3 / 'adapters.py',
                  's1/schema.py': S1O_SRC / 's1/schema.py', 'cases': CASES,
                  'yolo11s_320.onnx': ROBOT / 'yolo11s_320.onnx', 'yolo11s_640.onnx': ROBOT / 'yolo11s_640.onnx'}
         meta = {'started': utc(), 'smoke': a.smoke, 'block_s': ctx['duration'], 'idle': ctx['idle'],
                 'sha256': {k: sha256(p) for k, p in files.items()},
                 'model_bytes': {p: os.path.getsize(p) for p in (MODEL, DRAFT)},
+                'heat_stop': {'zone': PAUSE_ZONE, 'above_c': PAUSE_ABOVE_C},
                 'server_cmd': server_cmd(), 'mtp_cmd': server_cmd(MTP_ARGS), 'python': sys.version,
                 'cpus_allowed': sorted(allowed_cpus())}
         if a.resume:
--- frozen/test_coresidency.py
+++ current/test_coresidency.py
@@ -127,11 +127,23 @@
         return 0, 'uid=0(root)\n'
     cr.root = fake_root
 
+    hot = {'since': None}  # B4 reaches the live thermal pause 5 s after it is entered
+    real_run_block = cr.run_block
+
+    def run_block(name, *a):
+        hot['since'] = time.monotonic() if name == 'B4_only640_gemma' else None
+        try:
+            return real_run_block(name, *a)
+        finally:
+            hot['since'] = None
+    cr.run_block = run_block
+
     def thermal():
         while True:
             stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
+            z9 = 90000 if hot['since'] is not None and time.monotonic() > hot['since'] + 5 else 36000
             with open(root_dir / 'thermal.log', 'a') as f:
-                f.write(f'{stamp} z9=36000 z10=35000 z11=34000\n')
+                f.write(f'{stamp} z9={z9} z10=35000 z11=34000\n')
             time.sleep(1)
 
     def watcher():
@@ -163,7 +175,16 @@
     for n in names:
         b = json.loads((run / f'block_{n}.json').read_text())
         done = [r for r in b['reads'] if 'detect_ms' in r]
-        assert len(done) >= 10, (n, len(done))
+        heat = b['heat_stop']
+        assert heat['zone'] == 'z9' and heat['above_c'] == 80, heat
+        if n == 'B4_only640_gemma':  # the fake log goes to z9 90 degC 5 s in: the block ends there, still valid
+            assert heat['reached_limit'] and 2 < heat['time_to_limit_s'] < 7.5, heat
+            assert heat['zone_reading']['z9'] == 90000 and b['duration_s'] < b['planned_s'], (heat, b['duration_s'])
+            assert all(r['t'] <= heat['time_to_limit_s'] for r in b['reads']), 'read after the heat stop'
+            assert len(done) >= 4, (n, len(done))
+        else:
+            assert not heat['reached_limit'] and heat['time_to_limit_s'] is None, heat
+            assert len(done) >= 10, (n, len(done))
         frames = [r['frame'] for r in done]
         assert frames == sorted(set(frames)), f'{n}: frame numbers must advance'
         assert all('read_ms' in r and 'decode_ms' in r and 'age_s' in r for r in done), n
@@ -191,6 +212,9 @@
                   'peak PSS camera provider MiB', 'LMK log lines', 'zone9 start/end/max', 'mean battery W',
                   'Gemma load', 'MTP snapshot'):
         assert label in text, label
+    row = next(l for l in text.splitlines() if l.startswith('time to limit s')).split()[4:]
+    assert row[:3] == ['-', '-', '-'] and 2 < float(row[3]) < 7.5, row
+    assert 'INCOMPLETE' not in text, text
     assert (run.parent / 'downloads' / f'coresidency_{run.name}_report.txt').exists()
     return text
 
@@ -216,6 +240,9 @@
     print('failed logcat query reported as failed, not as 0 kills: PASS')
 
 
+NO_HEAT = {'zone': 'z9', 'above_c': 80, 'reached_limit': False, 'time_to_limit_s': None, 'zone_reading': None}
+
+
 def check_sampling_edges():
     sys.path.insert(0, str(HERE))
     import coresidency as cr
@@ -227,7 +254,7 @@
     base = {'t': 0, 'mem_available_mib': 3000, 'swap_used_mib': 0, 'pss_kb': {}, 'z9': 40000, 'battery_w': 2.0}
     samples = [dict(base, t=0, t_end=2), dict(base, t=5, t_end=13), dict(base, t=15, t_end=17),
                dict(base, t=175, t_end=183, mem_available_mib=100)]
-    b = {'reads': [], 'selector_calls': [], 'duration_s': 180.0, 'samples': samples,
+    b = {'reads': [], 'selector_calls': [], 'duration_s': 180.0, 'samples': samples, 'heat_stop': NO_HEAT,
          'thermal_start': {'z9': 40000}, 'thermal_end': {'z9': 41000}}
     s = cr.block_summary(b)
     assert s['late_samples'] == 1 and s['min_avail'] == 3000, s   # the sample finished after 180 s is dropped
@@ -296,17 +323,21 @@
     ok_lmk = {'ok': True, 'logcat_rc': 0, 'raw_head': ''}
     alive_cam = {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': True}
     slow = [{'t': 0.0, 'started_s': 0.0, 'ended_s': 170.0, 'ms': 1.7e5}, {'t': 20.0, 'started_s': 170.0, 'ended_s': 185.0, 'ms': 15000}]
-    blocks = [{'block': 'B3_mix_gemma', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk, 'survived': alive_cam},
+    blocks = [{'block': 'B3_mix_gemma', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk, 'survived': alive_cam,
+               'planned_s': 180, 'duration_s': 180.3, 'selector_calls': []},
               {'block': 'B1_mix_nollm', 'mode': 'mix', 'gemma': False,
                'survived': dict(alive_cam, llama_server=None, robotcam_process=False),
                'lmk': {'ok': False, 'logcat_rc': 1, 'raw_head': 'logcat_rc=1\nlogcat: Invalid time'}},
               {'block': 'B4_only640_gemma', 'mode': '640', 'gemma': True, 'lmk': ok_lmk,
                'survived': {'robotcam_new_frame_at_end': False, 'robotcam_process': True, 'llama_server': False},
-               'selector_calls': [], 'duration_s': 180.0},
+               'selector_calls': [], 'duration_s': 180.0, 'planned_s': 180, 'heat_stop': NO_HEAT},
               {'block': 'B3_slow_selector', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk, 'survived': alive_cam,
-               'selector_calls': slow, 'duration_s': 180.0}]
+               'selector_calls': slow, 'duration_s': 180.0, 'planned_s': 180, 'heat_stop': NO_HEAT}]
+    for b in blocks[:2]:
+        b['heat_stop'] = NO_HEAT
     good = {'sel': (9, 0, 7, 1.0, 2.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
-            'failed': {}, 'frames': 300, 'n320': 264, 'n640': 36, 'drift320': (70, 72), 'drift640': (260, 270)}
+            'failed': {}, 'frames': 300, 'n320': 264, 'n640': 36, 'drift320': (70, 72), 'drift640': (260, 270),
+            'drift_na': False}
     sums = [dict(good, sel=(0, 9, 0, None, None)),
             dict(good, sel=(0, 0, 0, None, None), sample_errors=3, bat_status=['Charging', 'Discharging']),
             dict(good, failed={'missing': 40}, frames=1, n320=0, n640=1, drift640=(9000, None), sel=(9, 0, 7, 1.0, 2.0)),
@@ -353,10 +384,137 @@
             {'sample': sample, 'server_status_kb': {}, 'load_s': 1.0, 'cmd': ['x']}))
         text, bad = cr.report(tmp)
         assert bad and 'B5_mtp_ram_snapshot' in bad[0] and 'INCOMPLETE: B5_mtp_ram_snapshot' in text, (bad, text)
-    print('cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE: PASS')
+        old = tmp / 'old_run'
+        old.mkdir()
+        (old / 'run.json').write_text(json.dumps(run))  # made by the frozen runner: no heat_stop
+        cr.require_native = lambda: None
+        try:
+            cr.main(['--smoke', '--resume', str(old)])
+            raise AssertionError('--resume of an old-format run was accepted')
+        except SystemExit as e:
+            assert 'older runner' in str(e), e
+    print('cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; '
+          'old-format run refused by --resume: PASS')
+
+
+def check_fix1_cadence_and_heat():
+    sys.path.insert(0, str(HERE))
+    import coresidency as cr
+    cr.SELECTOR_S = 20
+    assert (cr.PAUSE_ZONE, cr.PAUSE_ABOVE_C) == ('z9', 80), 'read from main.Robot.run_mission / get_temp'
+    ok_lmk = {'ok': True, 'logcat_rc': 0, 'raw_head': ''}
+    alive_cam = {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': True}
+    good = {'sel': (1, 0, 1, 1962.0, 1962.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
+            'failed': {}, 'frames': 40, 'n320': 37, 'n640': 3, 'drift320': (101, 101), 'drift640': (493, 493),
+            'drift_na': False}
+    call = lambda t, end: {'t': t, 'started_s': t, 'ended_s': end, 'ms': 2000.0, 'correct': True}
+    gemma = lambda **k: dict({'block': 'B3_mix_gemma', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk,
+                              'survived': alive_cam, 'heat_stop': NO_HEAT}, **k)
+    # the smoke evidence: 20 s block ran 20.25 s, one call at slot 0 (old count ceil(20.25/20) = 2)
+    smoke = gemma(duration_s=20.25, planned_s=20, selector_calls=[call(0.0, 1.97)])
+    assert cr.problems([smoke], [good]) == [], cr.problems([smoke], [good])
+    # full run: slots 0..160 (9); the old count ceil(180.4/20) = 10 could never be met
+    full = gemma(duration_s=180.4, planned_s=180, selector_calls=[call(20.0 * k, 20.0 * k + 2) for k in range(9)])
+    assert cr.problems([full], [dict(good, sel=(9, 0, 9, 2000.0, 2000.0))]) == []
+    assert cr.problems([full], [dict(good, sel=(8, 0, 8, 2000.0, 2000.0))]) == [
+        'B3_mix_gemma: selector cadence missed: 8/9 successful calls, started >5 s late at slots [], '
+        'ended after the block at slots []']
+    print('selector slots = those that start inside the block (smoke 1, full 9): PASS')
+
+    # heat stop at 42 s: 3 slots (0, 20, 40); the call in flight at the stop is not an overrun; drift n/a, valid
+    heat = dict(NO_HEAT, reached_limit=True, time_to_limit_s=42.0, zone_reading={'z9': 81000})
+    reads = [{'t': 0.5 * i, 'status': 'ok', 'detect_ms': 100.0, 'size': 640 if i % 10 == 0 else 320,
+              'read_ms': 1, 'decode_ms': 5, 'age_s': 0.3, 'frame': i + 1} for i in range(84)]
+    samples = [{'t': 5.0 * i, 't_end': 5.0 * i + 1, 'mem_available_mib': 3000, 'swap_used_mib': 0, 'pss_kb': {},
+                'z9': 60000, 'battery_w': 5.0, 'battery_status': 'Discharging'} for i in range(9)]
+    hot = gemma(duration_s=42.01, planned_s=180, heat_stop=heat, reads=reads, samples=samples,
+                thermal_start={'z9': 36000, 'waited_s': 0}, thermal_end={'z9': 81000},
+                selector_calls=[call(0.0, 2.0), call(20.0, 22.0), call(40.0, 43.5)])
+    s = cr.block_summary(hot)
+    assert s['drift_na'] and s['drift320'] == (None, None) and s['n640'] == 9, s
+    assert cr.problems([hot], [s]) == [], cr.problems([hot], [s])
+    assert cr.problems([dict(hot, selector_calls=hot['selector_calls'][:2])], [dict(s, sel=(2, 0, 2, 2000, 2000))])[0] \
+        .startswith('B3_mix_gemma: selector cadence missed: 2/3 successful calls')
+    # 70 s: both drift windows fit, so a missing one is still INCOMPLETE
+    long = dict(hot, duration_s=70.0, heat_stop=dict(heat, time_to_limit_s=70.0))
+    assert not cr.block_summary(long)['drift_na']
+    assert any('drift' in p for p in cr.problems([long], [dict(s, drift_na=False, drift640=(100, None),
+                                                               sel=(4, 0, 4, 2000, 2000))]))
+    # mix stopped at 3 s: no 640 frame was due yet
+    early = dict(hot, duration_s=3.0, heat_stop=dict(heat, time_to_limit_s=3.0), selector_calls=[call(0.0, 2.0)])
+    assert cr.problems([early], [dict(s, n640=0, sel=(1, 0, 1, 2000, 2000))]) == []
+    assert cr.problems([dict(early, heat_stop=NO_HEAT)], [dict(s, n640=0, drift_na=False)])  # no heat stop: missing
+    assert s['z9'][2] == 81.0, s['z9']  # the stop reading counts in the max even if no sample saw it
+    # the smoke case: B3 started at 81 degC and stops before its first frame and before any sample finished
+    now = gemma(duration_s=0.02, planned_s=20, reads=[], selector_calls=[], thermal_start={'z9': 81000, 'waited_s': 0},
+                thermal_end={'z9': 79000}, heat_stop=dict(heat, time_to_limit_s=0.01),
+                samples=[dict(samples[0], t=0.0, t_end=1.3)])
+    s = cr.block_summary(now)
+    assert s['frames'] == 0 and s['min_avail'] is None and s['z9'][2] == 81.0, s
+    assert cr.problems([now], [s]) == [], cr.problems([now], [s])
+    # the same block without a heat stop, or with a failed read, stays INCOMPLETE
+    assert len(cr.problems([dict(now, heat_stop=NO_HEAT)], [dict(s, drift_na=False)])) >= 4
+    assert any('RobotCam' in p for p in cr.problems([now], [dict(s, failed={'missing': 1})]))
+    with tempfile.TemporaryDirectory() as tmp:
+        log = Path(tmp) / 'thermal.log'
+        for z9, hit in ((80000, False), (80999, False), (81000, True), (95000, True)):
+            log.write_text(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} z9={z9} z10=90000 z11=90000\n')
+            assert (cr.heat_reached(str(log)) is not None) == hit, z9  # get_temp() // 1000 > 80, zone 9 only
+        log.write_text('2026-09-30T15:0')  # partial line: retried next frame
+        assert cr.heat_reached(str(log)) is None
+    print('heat stop: zone9 // 1000 > 80 from main.py; short or immediate stop valid, drift n/a, in-flight call '
+          'allowed, stop reading in the max: PASS')
+
+
+def check_fix1_cold():
+    sys.path.insert(0, str(HERE))
+    import coresidency as cr
+    with tempfile.TemporaryDirectory() as tmp:
+        tmp = Path(tmp)
+        f = tmp / 'weights.gguf'
+        with open(f, 'wb') as fh:
+            fh.write(os.urandom(1 << 20))
+            fh.flush()
+            os.fsync(fh.fileno())
+        f.read_bytes()
+        assert cr.resident_pages([f]) > 0
+        fd = os.open(f, os.O_RDONLY)
+        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
+        os.close(fd)
+        assert cr.resident_pages([f]) == 0, cr.resident_pages([f])
+        cr.OUT_ROOT, cr.MODEL, cr.HANDSHAKE_S = tmp, str(f), 5
+        answer = {'text': 'ok'}
+
+        def watcher(stop):
+            while not stop.is_set():
+                if (tmp / '.drop_request').exists() and not (tmp / '.drop_done').exists():
+                    (tmp / '.drop_done').write_text(answer['text'] + '\n')
+                time.sleep(0.05)
+        stop = threading.Event()
+        threading.Thread(target=watcher, args=(stop,), daemon=True).start()
+        cached = iter([2413, 1223] * 4)  # the smoke run's Cached values: 50.7 %, the old check's fallback
+        cr.meminfo_mib = lambda: {'cached_mib': next(cached)}
+        cr.cold_files = lambda: [f]
+        st = cr.make_cold()
+        assert st['mode'].startswith('full-cold') and st['resident_pages_after'] == 0, st
+        f.read_bytes()  # a load file still cached after an 'ok' answer: fallback
+        st = cr.make_cold()
+        assert st['mode'].startswith('weights-cold') and st['resident_pages_after'] > 0, st
+        assert "handshake answer 'ok'" in st['fallback_reason'] and cr.resident_pages([f]) == 0, st
+        answer['text'] = 'failed'
+        st = cr.make_cold()
+        assert st['mode'].startswith('weights-cold') and "answer 'failed'" in st['fallback_reason'], st
+        cr.cold_files = lambda: [tmp / 'missing']
+        answer['text'] = 'ok'
+        st = cr.make_cold()
+        assert st['mode'].startswith('weights-cold') and 'FileNotFoundError' in st['resident_pages_after'], st
+        stop.set()
+    print('cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS')
 
 
 def main():
+    check_fix1_cadence_and_heat()
+    check_fix1_cold()
     check_lmk_failure()
     check_sampling_edges()
     check_load_interrupt_and_selector_failure()
```

## Check output (verbatim, final candidate; Debian/proot, native Termux python run inside proot)
```
$ git -C /termux-home/robot rev-parse HEAD; git status --short --untracked-files=no
3584b5eddd3639c00c37b24e6e31f4d86fce2f2b
(end git status)
$ sha256sum *
3e3cd4e21f476d32e5784ebf51b65b9076cb5cd1b6b233688033d3044cdbabae  SMOKE.md
65c572e01826525e5f26881e9d5167145beb430a07e41f6ff15b37c58d1cf60d  coresidency.py
efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2  oneshot.sh
dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0  run_coresidency.sh
6ea51d6ead34926759c1b5b654e71fa169a4d13ffed0e410400fda3d30b2dd17  test_coresidency.py
bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19  test_oneshot.sh
$ bash -n oneshot.sh
exit 0
$ bash -n run_coresidency.sh
exit 0
$ bash -n test_oneshot.sh
exit 0
$ python3 -m py_compile coresidency.py test_coresidency.py   # Debian
exit 0
$ /data/data/com.termux/files/usr/bin/python -m py_compile coresidency.py test_coresidency.py   # native Termux python in proot
exit 0
$ bash test_oneshot.sh
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
exit 0
$ /data/data/com.termux/files/usr/bin/python test_coresidency.py   # onnxruntime platform warnings filtered
selector slots = those that start inside the block (smoke 1, full 9): PASS
heat stop: zone9 // 1000 > 80 from main.py; short or immediate stop valid, drift n/a, in-flight call allowed, stop reading in the max: PASS
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
time to limit s                                    -                   -                   -                 5.6
Gemma load (spawn to /health ok): cold 0.91 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 0.68 s
Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Heat stop = the robot's live pause (z9 > 80 degC, main.Robot.run_mission), checked between frames on the 5 s thermal log; time to limit = when the runner saw it (up to ~5 s plus one detection late); the block ends there and counts as run; drift is n/a if it ran under 60 s.
full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
exit 0
$ find /termux-home/robot -name "*.pyc" -newer SMOKE.md -not -path "*/coresidency/*"
(end)
```

## Final review (round 3) verbatim
```
**MAJOR — An immediate heat stop still makes the run INCOMPLETE.** [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:592) sets `robotcam_new_frame_at_end` to false when the stop occurs before the first processed frame, because `last` is `None`. [problems()](/termux-home/robot/benchmark/coresidency/coresidency.py:738) then rejects the block despite its `at_start` exception. The smoke evidence has B3 starting at z9 81 °C, so this is a realistic path. The immediate-stop test supplies `survived=True` rather than using the value `run_block()` would produce. The survival check needs an achievable baseline for this case, and the test needs to exercise it.

The cadence correction expects **9 calls for 180 s**, versus **10 before** when measured duration exceeded 180 s; the old full run would also have failed. The smoke cache values explain the old ratio fallback: 1223 MiB was 17 MiB above the 50% cutoff. I inspected the supplied checks; I did not rerun tests or run anything native.

REQUEST CHANGES
```

## Round 1 and 2 reviews verbatim
### Round 1
```
**MAJOR — Resuming the existing smoke run crashes during reporting.** [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:675) now requires `heat_stop` in every block JSON, and line 719 requires `planned_s` for Gemma blocks. The completed blocks in `/termux-home/coresidency/run_20260930T150019Z_smoke/` have neither field. `--resume` keeps those blocks, then `report()` raises `KeyError` instead of producing a report. Add compatibility handling or explicitly reject old-format runs before proceeding.

**MINOR — The heat-stop delay can exceed the documented five seconds.** [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:515) checks the thermal log only before a frame read and detection. If a detection runs across a hot log update, the block stops after that detection finishes. The report’s “time to limit” is therefore the detection time, not necessarily the threshold-crossing time.

The cadence correction is sound: **180 s expected 10 calls before, 9 after**; the old full run would also have failed. The smoke evidence supports why the old Cached ratio rejected the drop. The supplied offline checks pass; the revised candidate has no native result.

REQUEST CHANGES
```
### Round 2
```
**MAJOR — An immediate heat stop is reported as INCOMPLETE.** [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:713) still requires a successful selector call, usable sample, and frames; lines 735–741 require detections. The native smoke log shows B3 starting at z9 81 °C. With the new check, it can stop before its first frame or completed sample, then fail those requirements despite having no time to produce them. The offline heat test starts five seconds into B4 and misses this case.

**MAJOR — The reported maximum temperature can omit the limit reading.** [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:696) computes “zone9 start/end/max” from the start, end, and completed samples, but excludes the reading that triggered `heat_stop`. If that reading is 81 °C and the next log reading is lower, the report can show a maximum below the limit while treating the block as valid. Include the recorded stop reading in the maximum.

The cadence fix gives **9 expected calls for 180 s**, versus **10 before** when measured duration ran slightly long; the old full run would also have failed. The smoke cache figures explain the old ratio fallback, and the new page check remains conservative. The supplied offline checks pass; there is no new native result.

REQUEST CHANGES
```
