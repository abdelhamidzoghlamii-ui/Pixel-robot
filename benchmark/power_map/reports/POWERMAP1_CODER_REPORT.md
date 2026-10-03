# POWERMAP1 Coder report

## Profiles as used

```text
EXECUTION PROFILE
ROLE: Coder
PLATFORM: Codex CLI
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: lite
WHY: Bounded extension of a proven, reviewed benchmark runner; no robot or motor code touched.
FALLBACK: Claude Code CLI | claude-opus-5-5 | effort medium | Ponytail lite

REVIEWER PROFILE (you invoke it; never review your own work)
ROLE: Reviewer
PLATFORM: Codex CLI (codex exec, fresh session every round)
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off
WHY: Independent correctness and safety review in a separate fresh session.
FALLBACK: AGY CLI | gemini-3.1-pro-high | effort N/A | Ponytail off

```

No fallback was used. Coder: supplied gpt-6.1-sol / medium / Ponytail lite profile. Reviewer: fresh Codex CLI gpt-6.1-sol / medium / Ponytail off for every round.

## Candidate and gate

Base: `7c77f1439544634ede8b8baf92aa03e3b35b24b6` on main. Initial HEAD matched and `git status --short --branch` was `## main...origin/main`, with a clean tree. No staging, commit or push; no real root/Android/camera/server/motor/USB/serial activity. Only offline mock harnesses ran.

Final status:

```text
## main...origin/main
 M .gitignore
?? benchmark/power_map/RUN.md
?? benchmark/power_map/oneshot.sh
?? benchmark/power_map/power_map.py
?? benchmark/power_map/run_power_map.sh
?? benchmark/power_map/test_power_map.py
```

## New/changed files and SHA-256

| File | Change | SHA-256 |
| --- | --- | --- |
| `.gitignore` | changed | `13e1fafe6fce375fcd8d66a7f2e51234a03cf7caecf633395b83f111a2c09c3e` |
| `benchmark/power_map/RUN.md` | new | `33a060a4b50dc18cb03afa6c398212e59ac9b0384069f42b3d9110e47ba509f5` |
| `benchmark/power_map/oneshot.sh` | new | `874dfc850c47becfe5226dd0c184e5ffcb9b8a474b6848259ed193a27ebc49dd` |
| `benchmark/power_map/power_map.py` | new | `4756ce7dc7ba4e358578288c0d2509345fc4e2fe02d45f077f65355792e29539` |
| `benchmark/power_map/run_power_map.sh` | new | `444c2cb71f1497842f7b0f94c8d554102443a80e5b82736e8af111bf194774ba` |
| `benchmark/power_map/test_power_map.py` | new | `aa04f88d25b23af4813bae107c76c86c140645860d9a5b51a8402a1ba64fa196` |

The `.gitignore` exception exposes the new folder, which was otherwise hidden by `benchmark/*`. Archived folders and robot sources have no diff.

## Reused versus replaced

Imported unchanged from coresidency: native/proot refusal, cpuset guard and wait, root command wrapper, persistent RootShell, sysfs discovery/fast keys/fast sample/monitor loop (1 s), dumpsys parsing/read (5 s), block limits/latest readings, memory/PSS/battery sampler (5 s), camera stop/end check/force-stop/survival inputs, LMK collection, Server/server command/health/cleanup, selector construction/cases/letter scoring/20 s loop/warm-up, summary medians/P95 and common accounting. SizePolicy and default Detector are imported unchanged from robot code. No motors.py import, USB or serial access.

Local replacements: camera_start takes rate 1/2 (same retry/session validation); frame_loop takes rate/cadence and paces 1/rate seconds; skin gate drops the irrelevant zone9 term and bounds waiting at eight minutes (same idle+1.5 C threshold). Block orchestration is adapted from coresidency to keep one resident Gemma, allow per-block detector settings and add per-thread observations. INCOMPLETE checking is adapted only for required detector sizes/cadence, with added skin-slope/missing-block/ORT verification checks. Report and main implement power-map block lists, explicit per-policy capped time, least-squares skin slope, status changes, settings and resume. Launcher is copied with paths/wrapper changed, cache-drop watcher removed, cross-benchmark runner refusal retained/expanded, and CPU-core logging labelled; #123 screen timeout, root logger, Discharging/agent checks and five-minute idle retained.

R3/CONFIRM pin only session workers with per-thread sched_setaffinity plus a dedicated detector calling thread (ORT includes its caller in intra-op work). Foreground main thread affinity is unchanged. Default uses Detector() exactly as deployed. ORT version, options and observed default worker-pool count are recorded. Session workers are identified by newly created TIDs; explicit settings refuse unexpected worker counts. Per-second samples read processor, CPU ticks and Cpus_allowed_list for all runner threads. After review round 1, stale pre-pin processor values are excluded until CPU ticks advance; idle workers still have affinity verified and are reported as no sampled CPU tick advance. Sub-tick work/first-sample execution may be missed, and stat sampling is not a complete scheduler trace.

## Tests

All numbers in these outputs are synthetic/offline harness data, not hardware results. Archived tests were not edited. Existing test_oneshot.sh was also run against a temporary copied harness with its names/paths adapted to the new launcher; its source is `/termux-home/powermap_launcher_offline/test_oneshot.sh`. No blanket permission bypasses were used.

### Round 1: python benchmark/power_map/test_power_map.py

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block lists and CLI setting validation: PASS
5/10/off size sequences and frame pacing at rates 1/2; limit before first read: PASS
separate capped policies, skipped gaps, unknown readings, excluded edges/tail; least-squares skin slope: PASS
synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS
native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS
fake integration: interruption/resume preserves completed block, resident Gemma, report copy, camera cleanup: PASS
POWERMAP offline checks: PASS
```

### python benchmark/coresidency/test_coresidency.py

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS
  [B1] waiting: skin 36.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 34.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 1 s
cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS
RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure INCOMPLETE; cached pid recorded only: PASS
capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable scaling_max INCOMPLETE: PASS
layout discovery, 1 s sample and dump parsing: PASS
selector slots = those that start inside the block (smoke 1, full 9): PASS
limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop INCOMPLETE: PASS
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpwkjz9q1v/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpwkjz9q1v/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpwkjz9q1v/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpwkjz9q1v/.drop_done
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
Co-residency benchmark (DECISIONS #124), run run_20261001T205037Z_smoke  [SMOKE: not a measurement]
block length 8 s; idle z9 90.0 degC, idle skin 35.0 degC; runner sha256 85a6e4cb003b

                                        B1_mix_nollm    B2_only640_nollm        B3_mix_gemma    B4_only640_gemma
frames processed                                  16                  16                  16                   8
failed reads by status                          none                none                none                none
repeat reads (not failures)                        4                   4                   4                   0
detect320 ms median/P95                 10/10 (n 15)       n/a/n/a (n 0)        10/12 (n 15)       n/a/n/a (n 0)
detect640 ms median/P95                  30/30 (n 1)        30/31 (n 16)         31/31 (n 1)         30/30 (n 8)
drift320 first/last 30s                     10 -> 10          n/a -> n/a            10 -> 10          n/a -> n/a
drift640 first/last 30s                     30 -> 30            30 -> 30            31 -> 31          n/a -> n/a
read/decode ms median                       3.5/15.0            3.7/16.7            3.4/14.4             1.4/6.1
frame age s median                              0.08                0.07                0.07                0.10
selector ms median/P95                             -                   -             273/275             238/248
selector calls ok/err/correct                  0/0/0               0/0/0               3/0/1               2/0/1
min MemAvailable MiB                            3315                3383                3405                3403
max swap used MiB                               2002                2102                2099                2106
peak PSS runner MiB                              121                 121                 121                 121
peak PSS llama-server MiB                        n/a                 n/a                 121                 121
peak PSS RobotCam app MiB                         37                  37                  37                  37
peak PSS camera provider MiB                     264                 264                 264                 264
LMK log lines (kill lines)                     3 (2)               3 (2)               3 (2)               3 (2)
survived RobotCam / llama                    yes / -             yes / -           yes / yes           yes / yes
zone9 start/end/max degC              90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0
VIRTUAL-SKIN start/end/max degC       35.0/35.0/35.0      35.0/35.0/35.0      35.0/35.0/35.0      35.0/46.6/46.6
Android status max                                 3                   3                   3                   4
policy capped s (% of block)                  0 (0%)              0 (0%)              0 (0%)             3 (75%)
lowest scaling_max MHz 0/4/6          1803/2348/2850      1803/2348/2850      1803/2348/2850      1803/2348/2400
gate wait s                                        0                   0                   0                   0
stop limit                                         -                   -                   -      android_status
time to limit s                                    -                   -                   -                 4.0
mean battery W                                  1.64                1.64                1.64                1.64
battery status                           Discharging         Discharging         Discharging         Discharging
sample errors                                      0                   0                   0                   0
sample gaps >3s (max s)                      0 (2.0)             0 (2.0)             0 (2.0)             0 (2.0)
samples past block end (dropped)                   0                   0                   0                   0

Gemma load (spawn to /health ok): cold 0.95 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 0.68 s; gate wait 0 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmptk7sbq7h/a/server.pid

MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 3405 MiB, swap used 2106 MiB, PSS llama-server 121 MiB, VmHWM 23 MiB, load 0.70 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmptk7sbq7h/a/server.pid --model-draft /data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3

Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between frames: Android thermal status >= 4 (CRITICAL), battery >= 45.0 degC, CPU zone >= 110 degC in 3 consecutive 1 s samples (fault stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading interval late); the block ends there and counts as run; drift is n/a if it ran under 60 s. Skin/status = `dumpsys thermalservice` every 1 s; capped = time (each in-block 1 s sample counts the time since the previous one) with any policy's scaling_max_freq below its cpuinfo_max_freq, % of the block length. Gate: skin <= idle + 1.5 and z9 <= idle + 4 degC; warm start = not reached in 15 min.

full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
failed STOP broadcast: force-stop still runs: PASS
```

### bash benchmark/coresidency/test_oneshot.sh

```text
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
```

### New launcher against archived offline harness

```text
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
```

### Round 2 targeted stale-processor regression

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
stale pre-pin processor excluded for idle workers; active in/out-of-target CPUs verified: PASS
synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS
native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS
```

### Round 2: python benchmark/power_map/test_power_map.py

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block lists and CLI setting validation: PASS
5/10/off size sequences and frame pacing at rates 1/2; limit before first read: PASS
separate capped policies, skipped gaps, unknown readings, excluded edges/tail; least-squares skin slope: PASS
stale pre-pin processor excluded for idle workers; active in/out-of-target CPUs verified: PASS
synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS
native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS
fake integration: interruption/resume preserves completed block, resident Gemma, report copy, camera cleanup: PASS
POWERMAP offline checks: PASS
```

`git diff --check` and `bash -n benchmark/power_map/oneshot.sh benchmark/power_map/run_power_map.sh` passed. All listed test commands exited zero when completed.

## Independent review rounds (verbatim)

### Round 1: REQUEST CHANGES

Invocation (fresh session; edits frozen):

```bash
codex exec -m gpt-6.1-sol -c model_reasoning_effort="medium" -s read-only -C /termux-home/robot - < /termux-home/powermap_review_request_1.md
```

Request: `/termux-home/powermap_review_request_1.md`; final stdout: `/termux-home/powermap_review_1.txt`; diagnostics: `/termux-home/powermap_review_1.stderr`.

```text
Blocking finding:

- **[P2] ORT verification can reject correctly pinned blocks.** In [power_map.py](/termux-home/robot/benchmark/power_map/power_map.py), `build_detector()` creates workers before pinning them. `thread_sample()` then samples their last scheduled CPU immediately when the block starts, and `observed_workers()` permanently accumulates those values. An idle worker can still show its pre-pinning CPU from session construction. For `little`, that can be CPU 4–7, causing INCOMPLETE even though its allowed CPUs are correctly `0-3` and it never executes outside that set during the block. Cadence `off` makes this particularly relevant for unused 640-session workers. Distinguish stale processor values from execution during the block, and add an offline regression check for this case.

Nonblocking notes:

- The supplied offline checks passed. Real ORT pinning and rate-1 camera pacing remain explicitly unverified until human smoke.
- HEAD matches the frozen base; status shows only `.gitignore` and the new folder. Initial required instruction reads succeeded, but subsequent reads failed with a sandbox mount error. The remaining review used the complete supplied source and evidence.
- No files were changed, hardware invoked, or additional reviewer launched. This review grants no commit permission.

REQUEST CHANGES
```

### Round 2: APPROVE WITH NOTES

Invocation (fresh session; edits frozen):

```bash
codex exec -m gpt-6.1-sol -c model_reasoning_effort="medium" -s read-only -C /termux-home/robot - < /termux-home/powermap_review_request_2.md
```

Request: `/termux-home/powermap_review_request_2.md`; final stdout: `/termux-home/powermap_review_2.txt`; diagnostics: `/termux-home/powermap_review_2.stderr`.

```text
No blocking findings in the supplied frozen candidate.

The prior blocking finding is resolved: `observed_workers()` now records a processor value only when that thread’s CPU ticks advance between samples. Idle workers’ stale pre-pinning CPU values no longer cause false INCOMPLETE results. The added regression checks cover idle workers, execution inside the target CPUs, and execution outside them.

Nonblocking notes:

- The supplied final power-map suite and regression checks passed. The supplied coresidency and launcher checks also passed. I did not independently rerun them.
- Real ORT pinning and rate-1 camera pacing remain unverified until human smoke. The runner and documentation explicitly disclose this. Thread observations remain sampled evidence, with initial-interval and sub-tick limitations stated in the report.
- HEAD matches `7c77f1439544634ede8b8baf92aa03e3b35b24b6`; status shows only `.gitignore` and `benchmark/power_map/`. I read `AGENTS.md`, HANDOFF, WORKFLOW, REVIEWER and prototype instructions. Subsequent reads of STATUS, the decision index and full decisions failed twice because of a sandbox mount error; the remaining review used the complete supplied source and offline evidence.

No files were changed, staged or committed. No hardware or additional reviewer was invoked. This review grants no commit or push permission.

APPROVE WITH NOTES
```

## Open findings and unverified behavior

Round 1 returned REQUEST CHANGES for stale pre-pinning processor values; that blocking finding was fixed and regression-tested. Round 2 returned APPROVE WITH NOTES with no blocking code findings. Both nested codex invocations exited 0 and produced full reviews, but required later repository reads were blocked by sandbox mount errors. Under the task rule against treating permission-blocked review as a pass, final review acceptance is recorded as INCOMPLETE; the returned verdict is preserved verbatim, not promoted to an accepted pass. No sandbox bypass or fallback was attempted. Stop for the human. Real pinning and rate-1 pacing remain unverified.

Real ORT pinning behavior, runtime worker identification, default thread count, sampled CPU execution and rate-1 RobotCam pacing are untested on the phone until the human smoke run. No real power, skin, cap-duty-cycle, timing, memory or camera/server survival measurements were made. Worker-count assertions and observed affinities may fail on the phone; such a failure is not valid measurement evidence. Long CONFIRM stability and real interruption cleanup also require phone testing. Offline observations do not set the robot's required detection rate or establish uncapped operation indefinitely.

Human commands and expected times are in `benchmark/power_map/RUN.md`. No human run was started. Stop here for the human gate; reviewer approval is not commit/push authorization.
