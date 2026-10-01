# THERMAL_CHAR fix 1 — Coder report (round-3 blocking finding)

Coder: Claude Code CLI, claude-opus-5-5, medium, Ponytail lite. Reviewer: Codex CLI `codex exec -m gpt-6-sol -c model_reasoning_effort="medium" -s read-only` (fresh session). No fallback used.

## Pre-checks
- HEAD 3584b5eddd3639c00c37b24e6e31f4d86fce2f2b, tracked tree clean (before and after). `benchmark/thermal_char/` is gitignored (`.gitignore:8 benchmark/*`).
- All six frozen SHA-256 values matched THERMAL_CHAR_CODER_REPORT.md before any edit.

## Change
- `take_sample` (the only place a 1 s sample gets its time): `t` is now taken when the root-shell read completes, with `t_start` (read start) and `read_s` (= completion minus start) next to it. `utc` is also taken at completion, as in `dump_loop`. Every consumer (stop_reasons, the events and "at that moment" readings, timeline, cooldown, work-rate alignment, gap count) reads `s['t']`, so all of them now use the completion time. A failed read (RuntimeError) is still an incomplete sample, and it is also stamped at completion.
- samples_1s.jsonl now carries `t_start` and `read_s` per sample.
- Report LAYOUT: new line `1 s sample read duration (stamped at completion): median X s, max Y s`.
- Test: a new unit check uses a slow fake shell (0.4 s per read). The load start is placed inside the second read. The check asserts that the second sample's scaling_max_freq change is reported as `[load]` at its completion time, including the `(sample …)` reading. It also checks `t`, `t_start` and `read_s`, and the LAYOUT median/max line.
- One existing test assertion was changed, as a direct consequence of the fix. In the `sigterm` scenario, the sample being read when SIGTERM arrives now completes after load_stop and is correctly stamped after it. The report therefore shows a short COOLDOWN section (about 0.6 s under load here) instead of "no cooldown logged". The old assertion (`'no cooldown logged' in rep`) assumed start stamping. It is restated with the same intent: no read started after the load stop (`all t_start <= L1`). The run is still marked aborted/incomplete.
- Mutation check: I set `t` back to the start time and kept the new fields. The new unit test then fails at the event-placement assertion.
- Nothing else changed. The GGUF hash stays out (Local AI decision). RUN.md, oneshot.sh, run_thermal_char.sh and test_oneshot.sh are byte-identical to the frozen versions. The `__pycache__` created by my checks was removed.

## security_reminder_hook finding
`pgrep -fa security_reminder_hook`, run before the timing-sensitive tests, found 7 orphaned processes. They are the same PIDs as in the previous report: 18466 18468 18472 18473 18474 18475 18486. Each is `bash …/security-guidance/hooks/sg-python.sh …/security_reminder_hook.py` with PPID 1, state R, about 69% CPU each, 2h38m elapsed, and each has one defunct bash child. Load average was about 9. I did not kill them. The full offline test still passed in two consecutive runs, but these processes should be cleared before the native run.

## New SHA-256
```
46c49c727fa02fbc71b4fd5a97c1919a53652bf802a335cfe4a3dea9c03a336d  benchmark/thermal_char/RUN.md
16787b0679c24b711eb9e370cdcc9a9942cf689718764874b8ec3e418f84d77b  benchmark/thermal_char/oneshot.sh
c4df2fa2799f019fd67ac68580c9296001e1e8ea63d51f75ab906184864d682e  benchmark/thermal_char/run_thermal_char.sh
e640e6f33e0c44585322f37394b68936a59b22adc32bde824b649d3d7a6ee2f6  benchmark/thermal_char/test_oneshot.sh
a610c2bdd1490bbfee08f3fb7303d9aa34d69503e8c4c2f72161ec19616971db  benchmark/thermal_char/test_thermal_char.py
cbf73d12b5bcebb80db1d3c09f5b1e72f0231df915c490669c925f1bd9365d1d  benchmark/thermal_char/thermal_char.py
```

## Diff (against the frozen candidate)
```diff
--- a/benchmark/thermal_char/thermal_char.py	2026-09-30 20:38:18.817820910 +0000
+++ b/benchmark/thermal_char/thermal_char.py	2026-09-30 22:03:13.572761880 +0000
@@ -215,9 +215,19 @@
 
 
 def take_sample(shell, keys, script, layout, t0):
-    s = {'t': round(time.monotonic() - t0, 3), 'utc': cr.utc()}
+    start = time.monotonic()
     try:
-        s.update(parse_sample(keys, shell.run(script), layout))
+        lines, err = shell.run(script), None
+    except RuntimeError as e:
+        lines, err = None, e
+    done = time.monotonic()
+    # t = when the read was in hand (it ran t_start..t): stops, events, timeline, cooldown and gaps use it
+    s = {'t': round(done - t0, 3), 't_start': round(start - t0, 3), 'read_s': round(done - start, 3),
+         'utc': cr.utc()}
+    try:
+        if err:
+            raise err
+        s.update(parse_sample(keys, lines, layout))
     except RuntimeError as e:
         s.update(complete=False, error=str(e))
     s['cpus_ok'] = set(range(4, 8)) <= cr.allowed_cpus()
@@ -528,6 +538,7 @@
     if L1 is not None and run.get('after_stop') and samples and samples[-1]['t'] - L1 < run['cool_s'] - 2 * SAMPLE_S:
         bad.append(f'cooldown logged {samples[-1]["t"] - L1:.0f} of {run["cool_s"]} s')
     incomplete = [s for s in samples if not s.get('complete')]
+    reads = [s['read_s'] for s in samples if 'read_s' in s]
     failed_reads = [r for r in frames if r['status'] != 'ok']
 
     lines = [f'Thermal characterization run {out.name}{"  [SMOKE: functional check, not a measurement]" if run["smoke"] else ""}',
@@ -554,7 +565,9 @@
               f'  thermalservice dumps {len(dumps)} (with skin {len(good_skin)}, with status {len(good_status)}), '
               f'raw kept {sum(d.get("raw_kept", False) for d in dumps)}; 1 s samples {len(samples)} '
               f'(incomplete {len(incomplete)}, largest gap {f(max((b["t"] - a["t"] for a, b in zip(samples, samples[1:])), default=None))} s); '
-              f'failed RobotCam reads {len(failed_reads)}', '',
+              f'failed RobotCam reads {len(failed_reads)}',
+              f'  1 s sample read duration (stamped at completion): median {f(med(reads), ".3f")} s, '
+              f'max {f(max(reads, default=None), ".3f")} s', '',
               f'EVENTS (t = s from load start; {len(ev)} events)']
     for t, text in ev:
         lines += [f'  {rel(t)} [{phase(t)}] {text}', f'      {readings(t)}']
--- a/benchmark/thermal_char/test_thermal_char.py	2026-09-30 20:32:04.593820728 +0000
+++ b/benchmark/thermal_char/test_thermal_char.py	2026-09-30 22:07:41.724762011 +0000
@@ -316,6 +316,31 @@
             except RuntimeError as e:
                 assert 'no answer' in str(e)
             assert sh.run('echo next') == ['next'], 'output of a timed-out command leaked into the next one'
+
+            # a slow read: the sample, and so its event, is stamped when the read completed (start kept)
+            class Slow:
+                def run(self, script, timeout=5):
+                    time.sleep(0.4)
+                    return sh.run(script, timeout)
+            script, t0 = tc.sample_script(keys), time.monotonic()
+            rows = [tc.take_sample(Slow(), keys, script, layout, t0)]
+            (cf / 'policy6/scaling_max_freq').write_text('2400000\n')
+            rows.append(tc.take_sample(Slow(), keys, script, layout, t0))
+            (cf / 'policy6/scaling_max_freq').write_text('2850000\n')
+            assert all(r['complete'] and r['read_s'] >= 0.39 and abs(r['t'] - r['t_start'] - r['read_s']) <= 0.002
+                       for r in rows), rows
+            L0 = rows[1]['t_start'] + 0.2  # load starts during the second read: its change was seen after L0
+            (T / 'rep').mkdir()
+            (T / 'rep/run.json').write_text(json.dumps({'layout': layout, 'load_start': L0, 'smoke': False,
+                                                        'sha256': {'thermal_char.py': 'x'}, 'load_s': 1, 'cool_s': 1}))
+            (T / 'rep/samples_1s.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
+            rep = tc.report(T / 'rep')[0]
+            ev = rep.split('EVENTS')[1].split('TIMELINE')[0]
+            at = f'{rows[1]["t"] - L0:+.1f}'
+            assert f'  {at} [load] policy6 scaling_max_freq 2850 -> 2400 MHz' in ev and f'(sample {at})' in ev, ev
+            r0, r1 = rows[0]['read_s'], rows[1]['read_s']
+            assert f'read duration (stamped at completion): median {(r0 + r1) / 2:.3f} s, max {max(r0, r1):.3f} s' \
+                in rep, rep
             os.unlink(bat / 'temp')
             s = tc.take_sample(sh, keys, tc.sample_script(keys), layout, time.monotonic())
             assert not s['complete'] and s['bat_c'] is None, s
@@ -328,7 +353,8 @@
         finally:
             sh.close()
         assert sh.p.poll() is not None, 'root shell left running'
-    print('ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, missing CPU zone refused')
+    print('ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, slow read stamped at '
+          'completion, missing CPU zone refused')
 
     # a request cut off by the load stop keeps its streamed tokens; one cut off without a stop is a load failure
     with tempfile.TemporaryDirectory() as T:
@@ -487,8 +513,8 @@
         assert 'VIRTUAL-SKIN in HAL temperatures: NO' in rep and 'skin: no start value' in rep, load_s
     elif name == 'skinlost':  # last skin reading before load start: the 5 s count from load start
         assert 5 <= load_s < 6.5 and 'VIRTUAL-SKIN in HAL temperatures: yes' in rep, load_s
-    if name == 'sigterm':
-        assert 'no cooldown logged' in rep, rep.split('COOLDOWN')[1][:300]
+    if name == 'sigterm':  # no cooldown: no read started after the stop (one in flight completes after it)
+        assert all(s['t_start'] <= L1 for s in samples), [s['t_start'] - L1 for s in samples[-3:]]
     elif name == 'sensorfail':
         assert 4.5 < load_s < 8, load_s  # removed 0.5 s after camera start, stop 6 s after the last reading
     raw = [json.loads(l) for l in (out / 'thermalservice_raw.jsonl').read_text().splitlines()]
```

## Check output
```
$ pgrep -fa security_reminder_hook
18466 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
18468 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
18472 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
18473 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
18474 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
18475 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
18486 bash /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/sg-python.sh /root/.claude/plugins/synced/ff503365-328d-4646-a874-24093b664704_515f2405-15b3-4635-8dc7-798458853f36/security-guidance/hooks/security_reminder_hook.py
(exit 0)

$ bash -n oneshot.sh
(exit 0)
$ bash -n run_thermal_char.sh
(exit 0)
$ bash -n test_oneshot.sh
(exit 0)
$ python3 -m py_compile (Debian)
Python 3.13.5
(exit 0)
/data/data/com.termux/files/usr/bin/python3
--- full offline test (native python), first run, before the sigterm assertion change (only sigterm failed; cause above):
ok: parse_dump and every stop condition at its edge
ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, slow read stamped at completion, missing CPU zone refused
ok: interrupted request keeps its 5 streamed tokens; without a load stop it is a load failure
ok: a slow thermalservice dump is stamped at completion (start kept)
ok: duration: stop "planned load duration 8 s reached" after 8.0 s of load, exit 0
ok: skin: stop "VIRTUAL-SKIN 48.0 >= 48.0 degC" after 3.5 s of load, exit 0
ok: battery: stop "battery 45.0 >= 45.0 degC" after 3.3 s of load, exit 0
ok: status: stop "Android thermal status 5 >= 5 (EMERGENCY)" after 2.9 s of load, exit 0
ok: cpu: stop "CPU zone >= 110 degC in 3 consecutive 1 s samples (max [110.0, 110.0, " after 4.2 s of load, exit 0
ok: noskin: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.3 s of load, exit 1
ok: skinlost: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.0 s of load, exit 1
ok: sensorfail: stop "fail closed: no battery temperature reading for 6 s" after 6.3 s of load, exit 1
ok: noframes: stop "load failed: no new RobotCam frame for 2 s" after 4.8 s of load, exit 1
ok: serverdie: stop "load failed: llama-server request failed: RuntimeError: stream ended w" after 2.9 s of load, exit 1
ok: cores: stop "cores 4-7 lost (allowed [0, 1, 2, 3, 4, 5])" after 2.7 s of load, exit 1
ok: startup: stop "during startup: VIRTUAL-SKIN 48.5 >= 48.0 degC", exit 1
ok: detectfail: stop "load failed: frame/detect loop failed: ValueError: fake detector failu" after 3.1 s of load, exit 1
FAIL: sigterm: AssertionError:  (start = latest reading before llama-server started, i.e. idle baseline; time from the load stop until within 2 degC of it)
FAILED
--- final files, run 1:
$ bash -n *.sh
bash -n oneshot.sh ok
bash -n run_thermal_char.sh ok
bash -n test_oneshot.sh ok
$ python3 -m py_compile
Python 3.13.5
(exit 0)
$ /data/data/com.termux/files/usr/bin/python -m py_compile
Python 3.13.13
(exit 0)
--- /data/data/com.termux/files/usr/bin/python test_thermal_char.py
  warnings.warn(
ok: parse_dump and every stop condition at its edge
ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, slow read stamped at completion, missing CPU zone refused
ok: interrupted request keeps its 5 streamed tokens; without a load stop it is a load failure
ok: a slow thermalservice dump is stamped at completion (start kept)
ok: duration: stop "planned load duration 8 s reached" after 8.2 s of load, exit 0
ok: skin: stop "VIRTUAL-SKIN 48.0 >= 48.0 degC" after 2.9 s of load, exit 0
ok: battery: stop "battery 45.0 >= 45.0 degC" after 3.0 s of load, exit 0
ok: status: stop "Android thermal status 5 >= 5 (EMERGENCY)" after 3.2 s of load, exit 0
ok: cpu: stop "CPU zone >= 110 degC in 3 consecutive 1 s samples (max [110.0, 110.0, " after 3.2 s of load, exit 0
ok: noskin: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.2 s of load, exit 1
ok: skinlost: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.0 s of load, exit 1
ok: sensorfail: stop "fail closed: no battery temperature reading for 6 s" after 6.0 s of load, exit 1
ok: noframes: stop "load failed: no new RobotCam frame for 2 s" after 4.6 s of load, exit 1
ok: serverdie: stop "load failed: llama-server request failed: RuntimeError: stream ended w" after 2.7 s of load, exit 1
ok: cores: stop "cores 4-7 lost (allowed [0, 1, 2, 3, 4, 5])" after 3.0 s of load, exit 1
ok: startup: stop "during startup: VIRTUAL-SKIN 48.5 >= 48.0 degC", exit 1
ok: detectfail: stop "load failed: frame/detect loop failed: ValueError: fake detector failu" after 3.1 s of load, exit 1
ok: sigterm: stop "aborted: SystemExit: 143" after 1.2 s of load, exit 143
ALL OK
(exit 0)
--- final files, run 2:
$ bash -n *.sh
bash -n oneshot.sh ok
bash -n run_thermal_char.sh ok
bash -n test_oneshot.sh ok
$ python3 -m py_compile
Python 3.13.5
(exit 0)
$ /data/data/com.termux/files/usr/bin/python -m py_compile
Python 3.13.13
(exit 0)
--- /data/data/com.termux/files/usr/bin/python test_thermal_char.py
  warnings.warn(
ok: parse_dump and every stop condition at its edge
ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, slow read stamped at completion, missing CPU zone refused
ok: interrupted request keeps its 5 streamed tokens; without a load stop it is a load failure
ok: a slow thermalservice dump is stamped at completion (start kept)
ok: duration: stop "planned load duration 8 s reached" after 8.2 s of load, exit 0
ok: skin: stop "VIRTUAL-SKIN 48.0 >= 48.0 degC" after 2.5 s of load, exit 0
ok: battery: stop "battery 45.0 >= 45.0 degC" after 4.0 s of load, exit 0
ok: status: stop "Android thermal status 5 >= 5 (EMERGENCY)" after 2.7 s of load, exit 0
ok: cpu: stop "CPU zone >= 110 degC in 3 consecutive 1 s samples (max [110.0, 110.0, " after 3.3 s of load, exit 0
ok: noskin: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.1 s of load, exit 1
ok: skinlost: stop "fail closed: no VIRTUAL-SKIN and Android status reading for 5 s" after 5.1 s of load, exit 1
ok: sensorfail: stop "fail closed: no battery temperature reading for 6 s" after 6.3 s of load, exit 1
ok: noframes: stop "load failed: no new RobotCam frame for 2 s" after 4.6 s of load, exit 1
ok: serverdie: stop "load failed: llama-server request failed: RuntimeError: stream ended w" after 2.6 s of load, exit 1
ok: cores: stop "cores 4-7 lost (allowed [0, 1, 2, 3, 4, 5])" after 2.9 s of load, exit 1
ok: startup: stop "during startup: VIRTUAL-SKIN 48.5 >= 48.0 degC", exit 1
ok: detectfail: stop "load failed: frame/detect loop failed: ValueError: fake detector failu" after 3.1 s of load, exit 1
ok: sigterm: stop "aborted: SystemExit: 143" after 1.6 s of load, exit 143
ALL OK
(exit 0)
--- bash test_oneshot.sh:
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
(exit 0)
```

## Review
- Round 1 (the only round): **APPROVE WITH NOTES**. No changes were made after the review; the hashes were re-verified and match.
- On the note: the same narrow window existed with the old assertion. Under start stamping, a read starting between the stop and the logger stop event would also have produced a sample after L1 and failed `'no cooldown logged'`. The window lasts a few statements (`except` → `finally: log_stop.set()`), against a 1 s sampling period.

### Final review — round 1 (verbatim)
```
No blocking findings. The sample’s `t` is recorded after the root-shell read, `t_start` and `read_s` preserve its interval, and the report’s sample consumers use `t`. The slow-read test checks event placement; the supplied offline runs pass. I verified the candidate file hashes but did not rerun the tests.

Non-blocking: the SIGTERM test’s `t_start <= load_stop` assertion could be timing-sensitive if the sampling thread starts a read between recording the stop and setting the logger stop event.

VERDICT: APPROVE WITH NOTES
```

Reviewer stderr diagnostics are kept in the session scratchpad (review_r1.err) and are not part of the review.

Unverified: no native/hardware run was made. No staging, commit or push was done.
