# CAMPOWER2 coder report

## Profiles

EXECUTION PROFILE
ROLE: Coder
PLATFORM: Codex CLI v0.160.0
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: lite
WHY: Bounded follow-up fixes to reviewed CAMPOWER1 RobotCam switches and benchmark, driven by two phone smoke runs.
FALLBACK: Claude Code CLI | claude-opus-5-5 | effort medium | Ponytail lite — not used.

REVIEWER PROFILE
ROLE: Reviewer
PLATFORM: Codex CLI (codex exec, fresh session every round)
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off (explicitly instructed)
WHY: Independent review of Camera2 manual-mode changes and measurement validity.
FALLBACK: AGY CLI | gemini-3.1-pro-high | effort N/A | Ponytail off — not used.

## Corrected base check

Main HEAD and origin/main: `7c77f1439544634ede8b8baf92aa03e3b35b24b6`.
Experiment HEAD and origin/robotcam-camera-power initially: `89b5eb366bec61d4c60660948298344be52086fe`.
Experiment branch `robotcam-camera-power`, initially clean. Main status exactly:

```text
 M .gitignore
?? benchmark/duty_cycle/
?? benchmark/power_map/
```

`benchmark/camera_power/` is present and **ignored**, as explicitly corrected by Local AI after the initial base-gate stop. `.gitignore:8` is `benchmark/*`. It was not edited. Its SHA-256: `2e82b2fc97e6c00ca7f0f014e34e413e57ab21afef169470719533021ec5f01b`.
All seven benchmark hashes matched the CAMPOWER1 report before any CAMPOWER2 edits:

```text
5a9106c249ce84adf4ac9c5eb16ae8e1738fefe460ca97a6ba9fd0b71dcbe036  benchmark/camera_power/README.md  MATCH
c0e43cf8cd94ad6ca1fdf2307af3caa0ad5095b2b05ff4e69825353f8b809e66  benchmark/camera_power/RESEARCH.md  MATCH
d30dd04c4f5c0e096b257bba7e2eecb15092e50c0ed02345b18c8d17334c004a  benchmark/camera_power/camera_power.py  MATCH
0540acfb9a8ddedf710755aaa2df64424582ec0577ea6886705046e3e9c1bdfe  benchmark/camera_power/launcher.py  MATCH
cb557055e51b0aae7811cfe98ce99e8e463bccb6d7db3e273fd6bda3f0499e58  benchmark/camera_power/oneshot.sh  MATCH
e2fdaaa2d4c54a1b4090bc43b63ff3c0cfd18805c1822d5bc4387a4f2cf9557f  benchmark/camera_power/run_camera_power.sh  MATCH
c398f90ff173cc2d2a35458e55a9b50705d89d8914a127db04d77c4d7499ce8e  benchmark/camera_power/test_camera_power.py  MATCH
```

Read the CAMPOWER1 report and retained task packet, shared workflow/Coder/Reviewer instructions, prototype and app instructions, relevant #122/#127 context, current app status and decision index. RobotCam is the prototype camera helper, not the proposed native robot app. No canonical status or decisions were edited.

## Findings and changes (evidence labels)

### 1. What frames_failed actually counted

**Documented repository:** CAMPOWER1 summed every polling result other than `ok` or `repeat`, including startup polls. Polling sleeps 0.1 s after each decode/metrics pass, rather than observing individual failed sensor captures. `robotcam_reader.py` returns `missing` for unreadable/absent JPEG, capture predating block start, or age >2 s; `bad` for decode/comment/clock failure; `other_session` for changed session. Repeats are tested only after the reader's age validation. Atomic rename provides a whole old/new JPEG; these logs do not show partial-read/decode failures. Pairing failure leaves a good JPEG `ok` with null diagnostics; it is not a frame failure.

**Measured smoke evidence, not power measurements:** every ordinary block's rejected polls precede its first good frame. Lit first baseline: 16 missing polls before first good at 1.991 s, 19 distinct good frames and 124 repeat polls; no post-start rejection. Dark first baseline: 17 startup polls, first good at 2.077 s. Mode A: six startup polls at about 0.98 s in both runs. Both manual200 runs have 19 startup polls, zero post-start rejections.

**Changed:** report `startup_wait_polls`, post-start `frames_failed`, raw status counts, repeat polls and unpaired sidecars separately. `frames_failed` explicitly counts rejected **polls**, not distinct sensor/capture failures. All raw reads remain evidence. No-good-frame blocks still fail validity checks. Read/age/session contract is unchanged; a stale repeat still fails before it could be classified as repeat.

### 2. manual500 is a real freshness gap

**Measured smoke:** dark: six good frames, 105 missing polls = 29 startup + 76 post-start; first 3.345 s. Lit: six good frames, 101 missing polls = 30 startup + 71 post-start; first 3.409 s. Lit observed gaps: 2.497, 2.876, 3.078, 2.900, 3.039 s. Publication frame numbers 1–6 are consecutive. Paired RESULT frame duration is 500000000 ns. First lit frame age 0.794 s, subsequent good ages roughly 0.65–0.79 s. Retained inferred-age evidence shows post-start missing polls when the preceding JPEG would exceed the unchanged 2 s age limit. This is not a reader timeout (there is none), a sidecar-pair failure, bad decode, or a count of 101 failed sensor exposures.

**Documented source:** B submits a still on a one-second tick; ticks are skipped while `stillInFlight` is true. It clears only when the capture completes/fails. Slow repeating preview work must finish before a higher-priority capture proceeds. [Android CameraCaptureSession](https://developer.android.com/reference/android/hardware/camera2/CameraCaptureSession#capture(android.hardware.camera2.CaptureRequest,%20android.hardware.camera2.CameraCaptureSession.CaptureCallback,%20android.os.Handler)).

**Inference:** the long preview/still pipeline plus tick skipping explains the ~3 s publication cadence. Logs do not contain queue depth, submission/completion timing or CaptureFailure events, so the exact internal HAL delay is not proven. Requested 500 ms itself is supported and observed in RESULT.

**Changed:** when frame_ms is nonzero, B repeats its chosen capture template directly to the publish YUV surface; A repeats PREVIEW there. This applies during convergence too. Manual B has no small preview surface and no periodic still queue/in-flight gating. Only the opt-in path changes surfaces/requests. Manual A/B use sensor timestamps when realtime (arrival clock otherwise) and an anchored publish deadline with half-frame tolerance, avoiding `now + period` skipping every other 1 Hz frame due to jitter. frame_ms accepts 100–1000, with existing MANUAL_SENSOR/AE OFF/max duration checks.

**Offline model:** all integer durations 100–1000 pass an average ~1/s selection model with +/-4 ms jitter and a 30 s AE recovery transition. 200/500/1000 selections are paced around one second. Durations that do not divide one second necessarily quantize per-frame intervals; the model's worst gap is 1.966 s at 983 ms. Strictly identical one-second capture spacing cannot be promised for non-divisor streams; buffer/pipeline delivery latency may also exceed reader freshness. This is modeled evidence, not HAL proof. No requested A/B combination is proven unsupported by supplied phone evidence; actual 1000 ms publication freshness remains unverified.

**Benchmark validity:** new-frame poll gaps and observed publish Hz are reported. Gaps over `1/rate + min(frame_ms/1000, 0.5) + 0.35 s` mark INCOMPLETE in addition to reader failures. Timestamp polling includes decode/metrics overhead. This tolerance targets the requested default variants; optional non-divisor experiments may be INCOMPLETE due to quantization and need inspection. The original manual500 smoke blocks remain INCOMPLETE under the corrected classification.

### 3. Manual exposure and post-RAW boost

**Measured smoke:** dark manual200 luma is near black while automatic baselines are around 4; lit manual200 luma is around 131. Copied exposure/ISO appears in RESULT, with small device rounding. The original schema did not record boost.

**Guess:** loss of post-RAW boost may explain dark manual brightness; existing logs cannot confirm the boost value or exclude other processing/3A causes.

**Documented Android:** post-RAW gain affects YUV/JPEG and is controlled explicitly with AE OFF; RESULT reports applied gain. [CaptureRequest post-RAW boost](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest.html#CONTROL_POST_RAW_SENSITIVITY_BOOST).

**Changed:** copy boost from the same converged result as exposure/ISO when the request key is advertised and the result is non-null. On unavailable devices no override is added. Requested and RESULT `post_raw_boost` fields are recorded (null means absent/unreported). Exposure is `min(converged_exposure, frame_ms * 1000000)`: never increased and never beyond requested duration. Keep AE re-convergence after 30 s; normal AF/AWB template behaviour retained. `power.schema_version=2` lets the benchmark reject an older APK. Latest RESULT still may describe another sensor frame than the JPEG; sidecar pairing does not establish sensor-result pairing.

## Files and SHA-256 / preservation proof

Only three app branch files changed since 89b5eb3: README.md, CameraPower.kt and CameraService.kt. Local benchmark changes: README.md, camera_power.py, test_camera_power.py; remaining four local benchmark files unchanged. No benchmark file was staged. No helper, reader, robot/motor/USB code, .gitignore or canonical document changes.

Complete structural verification and file hashes:

```text
Protected JPEG/comment/EXIF/clock/YUV/size/writer/stop/generation methods byte-identical to origin/main: PASS
onFrame with disabled manual branch equals origin/main; original A pacing and B direct publication preserved: PASS
startSession with disabled manual branch equals 89b5eb3, and origin/main except observation callbacks/template selector: PASS
Default B surface guard is true; sizes, ImageReader queues, period/ticks and captureStill unchanged since CAMPOWER1: PASS
Default request identical to origin/main after removing no-op power.apply; default apply sets no keys, observe cannot replace requests: PASS
robotcam_reader.py byte-identical in both worktrees and origin/main: PASS
.gitignore SHA-256 unchanged since corrected base verification: PASS
No Gradle/Kotlin compiler or Android SDK available; offline source checks and pacing model do not compile Kotlin or validate HAL.
c22c87362a4b50e392f8f0132e00f2975c81c4797e74d0c516cb353611c869bd /termux-home/robotcam-exp/android/robotcam/README.md
9583f7ce0823ff7615a5d603018854a2008db35351664c772dcbf49b97370164 /termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraService.kt
d362dbac3679bb3abd2555987097466ee99573c89ec3b28bdba5a6fdfbf29ae0 /termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraPower.kt
76a5447c90e5f739d0ff31a308c33e43fa0a6107b98788490f481ebe18c19790 /termux-home/robot/benchmark/camera_power/README.md
c0e43cf8cd94ad6ca1fdf2307af3caa0ad5095b2b05ff4e69825353f8b809e66 /termux-home/robot/benchmark/camera_power/RESEARCH.md
fdb6380e500305ae31eef49c2b9ece8599fcf68081f369efd4d82f514db59f5f /termux-home/robot/benchmark/camera_power/camera_power.py
0540acfb9a8ddedf710755aaa2df64424582ec0577ea6886705046e3e9c1bdfe /termux-home/robot/benchmark/camera_power/launcher.py
cb557055e51b0aae7811cfe98ce99e8e463bccb6d7db3e273fd6bda3f0499e58 /termux-home/robot/benchmark/camera_power/oneshot.sh
e2fdaaa2d4c54a1b4090bc43b63ff3c0cfd18805c1822d5bc4387a4f2cf9557f /termux-home/robot/benchmark/camera_power/run_camera_power.sh
9792ec266cd823436178d8b7bd72272852d05f9d61b4ea431a76eb176141e6e8 /termux-home/robot/benchmark/camera_power/test_camera_power.py

```

The disabled-manual branches compare against origin/main; protected JPEG/EXIF/clock/YUV/size/writer/cleanup/generation methods are byte-identical. Default B uses the same two surfaces and periodic STILL_CAPTURE; default A uses the original publish timing. power.apply adds no keys with default options, and observe returns without request replacement. Default capture-template selector resolves to STILL_CAPTURE. Diagnostics/callbacks add the CAMPOWER1 observation overhead and new sidecar fields; this proves request/contract preservation, not identical elapsed execution or power.

## Tests (full output pasted)

All six commands below exited 0. These are offline tests with fake camera/root/Android/detector/server/sysfs interfaces; no actual hardware benchmark, camera, launcher, su, am, USB/serial or motor command was executed. The fake server is an HTTP test process, not a model workload. The ONNX Runtime Android warning arises from helper imports. New Camera2 checks are static source assertions and a Python arithmetic model, not execution of Kotlin or a HAL simulator. Local Gradle/Kotlin compiler/Android SDK are unavailable; CI is the compile check.

### /data/data/com.termux/files/usr/bin/python benchmark/camera_power/test_camera_power.py

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block order/duration/baseline no extras/all six extras: PASS
startup waits / post-start rejection / repeats / unpaired sidecars: PASS
existing dark/lit smoke regression: ordinary failures -> startup waits; manual500 remains INCOMPLETE: PASS
Android source contracts: manual streaming A/B, boost copy, exposure clamp, 30 s recovery: PASS
manual pacing model 100..1000 ms with jitter/re-convergence: PASS (HAL unverified)
luma/Laplacian/power gaps/skin slope/CPU math: PASS
old APK/wrong options/proot refusal; sidecar pair/malformed; atomic resume: PASS
mocked block monitors/frames/images/metrics/cleanup: PASS
mocked block monitors/frames/images/metrics/cleanup on core loss: PASS
limit + stalled/failed accounting: STOP/end-check/force-stop precedes drain; CPU error retained: PASS
fail-closed after startup: evidence retained, no next variant, resume refuses, final cleanup: PASS
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
inherited launcher offline fake-only suite: PASS
CAMPOWER2 offline checks: PASS (no hardware)

Exit: 0
```

### /data/data/com.termux/files/usr/bin/python benchmark/duty_cycle/test_duty_cycle.py

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
PARTS/CYCLE/smoke block lists and active-only 20 s selector clock: PASS
restart passes request CLOCK_BOOTTIME; actual reader rejects pre-restart capture: PASS
energy above baseline and per-call interpolation/missing coverage; proc stat parser: PASS
CAM/YOLO/YOLO_NOSPIN/Active rate 1: exactly one read and JPEG decode per second, work included in period: PASS
LOAD spawn/health/call CPU split, GGUF cold/warm, stop on interruption, worker joined: PASS
[CONT] selector drain 0.051s; RuntimeError: failed limit query
[CONT] selector drain 0.051s; SystemExit: 143
[CONT] selector drain 0.051s; CoresLost: body core loss
[CONT] selector drain 122.000s; RuntimeError: selector drain exceeded 120 s; request cancelled
[CONT] selector drain 122.000s; SystemExit: 143
[CONT] selector drain 122.000s; SystemExit: 129
[CONT] selector drain 122.000s; SystemExit: 130
[CONT] selector drain 122.000s; KeyboardInterrupt: 
SystemExit TERM/HUP/INT and KeyboardInterrupt override retained CoresLost in final join, after joining: PASS
selector drain: arbitrary error, signal, unwinding CoresLost and timeout cancellation all join before propagation: PASS
both ORT spinning options retained; rejection produces INCOMPLETE without fallback: PASS
synthetic report/invalid battery; atomic resume preserves finished blocks and redoes HEATCOOL; proot refusal: PASS
core loss during pending selector drain: request finished before retry; discarded-block drain time recorded: PASS
mocked main + real block loops: interrupt/resume preserves CONT, redoes HEATCOOL, no COOL gate, final cleanup: PASS
pending selector + core loss + actual SIGTERM: call finished, exit 143, no retry, RobotCam/server stopped: PASS
DUTY1 offline checks: PASS

Exit: 0
```

### /data/data/com.termux/files/usr/bin/python benchmark/power_map/test_power_map.py

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

Exit: 0
```

### /data/data/com.termux/files/usr/bin/python benchmark/coresidency/test_coresidency.py

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
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpjxc5ui3g/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpjxc5ui3g/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpjxc5ui3g/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpjxc5ui3g/.drop_done
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
Co-residency benchmark (DECISIONS #124), run run_20261003T070822Z_smoke  [SMOKE: not a measurement]
block length 8 s; idle z9 90.0 degC, idle skin 35.0 degC; runner sha256 85a6e4cb003b

                                        B1_mix_nollm    B2_only640_nollm        B3_mix_gemma    B4_only640_gemma
frames processed                                  16                  16                  16                   8
failed reads by status                          none                none                none                none
repeat reads (not failures)                        4                   4                   4                   0
detect320 ms median/P95                 10/12 (n 15)       n/a/n/a (n 0)        10/10 (n 15)       n/a/n/a (n 0)
detect640 ms median/P95                  30/30 (n 1)        30/33 (n 16)         30/30 (n 1)         30/30 (n 8)
drift320 first/last 30s                     10 -> 10          n/a -> n/a            10 -> 10          n/a -> n/a
drift640 first/last 30s                     30 -> 30            30 -> 30            30 -> 30          n/a -> n/a
read/decode ms median                       3.7/16.6            3.1/13.0            3.2/12.2            5.0/12.1
frame age s median                              0.07                0.07                0.06                0.13
selector ms median/P95                             -                   -             252/284             278/293
selector calls ok/err/correct                  0/0/0               0/0/0               3/0/1               2/0/1
min MemAvailable MiB                            3140                3132                3095                3099
max swap used MiB                               2213                2213                2207                2206
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

Gemma load (spawn to /health ok): cold 0.90 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 1.00 s; gate wait 0 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmprb1ee58_/a/server.pid

MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 2969 MiB, swap used 2060 MiB, PSS llama-server 121 MiB, VmHWM 24 MiB, load 0.80 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmprb1ee58_/a/server.pid --model-draft /data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3

Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between frames: Android thermal status >= 4 (CRITICAL), battery >= 45.0 degC, CPU zone >= 110 degC in 3 consecutive 1 s samples (fault stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading interval late); the block ends there and counts as run; drift is n/a if it ran under 60 s. Skin/status = `dumpsys thermalservice` every 1 s; capped = time (each in-block 1 s sample counts the time since the previous one) with any policy's scaling_max_freq below its cpuinfo_max_freq, % of the block length. Gate: skin <= idle + 1.5 and z9 <= idle + 4 degC; warm start = not reached in 15 min.

full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
failed STOP broadcast: force-stop still runs: PASS

Exit: 0
```

### bash benchmark/duty_cycle/test_oneshot.sh

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

Exit: 0
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

Exit: 0
```

## Independent review rounds (verbatim)

Each invocation: `codex exec -m gpt-6.1-sol -c 'model_reasoning_effort="medium"' -s read-only -C /termux-home/robot -o /tmp/campower2_review_N.final - < /tmp/campower2_review_request_N.md`, stdout/stderr retained separately. Candidate frozen by SHA-256 manifest during each round. Packets include the full authorized task and correction, base, diff since 89b5eb3 AND origin/main, all seven full camera benchmark files, log excerpts with evidence labels, source/helper context and full offline test outputs. Sandbox mount errors with substantive verdict count by explicit task instruction; no permission bypass was used.

### Round 1

```text
Review completed with Ponytail off. **No blocking defect found in the supplied frozen candidate.**

The initial instruction reads succeeded. Subsequent Git, hash, source-location, and history reads failed with:

> error building bubblewrap command: app-server socket directory has an unsupported host mount

I used the complete inline source/diffs, evidence, and offline outputs after that failure. I could not independently confirm the worktree state or hashes, read the CAMPOWER1 report, or rerun tests. No files were changed; no hardware, benchmark, launcher, root, Android command, or additional reviewer was run.

The candidate satisfies the bounded follow-up at source-review level:

- **Failure classification:** `poll_counts()` correctly separates rejected startup polls from post-start rejected polls. Repeats and missing sidecar pairing remain separate. The supplied dark/lit evidence supports the ordinary failures being exclusively startup polls. Manual_500 still has genuine freshness failures: 76 dark and 71 lit post-start rejected polls, six accepted frames, and roughly three-second publication gaps. These are poll counts, not distinct failed captures.
- **Manual cadence:** the opt-in B path removes the small preview surface and periodic still queue, repeating directly to the publication surface instead. A uses repeating PREVIEW. Both paths use anchored selection; the default path remains behind `frameMs == 0`. This addresses the suspected scheduling mechanism, but the exact HAL cause and effectiveness remain unverified. Android documents that captures take priority over further repeating requests while current processing must finish; that supports possible pipeline delay, not proof of the specific three-second stall. [CameraCaptureSession reference](https://developer.android.com/reference/android/hardware/camera2/CameraCaptureSession)
- **Exposure and boost:** `CameraPower.observe()` copies reported boost when the request key is available, clamps exposure to the requested duration without increasing it, and retains the 30-second return to AE convergence. Requested and RESULT boost are recorded. Copying post-RAW boost alongside sensor sensitivity is consistent with Android’s brightness-preservation guidance. The dark-room diagnosis remains a hypothesis because the old logs omitted boost. [CaptureRequest reference](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest#CONTROL_POST_RAW_SENSITIVITY_BOOST)
- **Benchmark selection:** the eight default blocks match the requested order, totaling 16 minutes at 120 seconds each. Optional blocks remain selectable by name. The printed 21–85-minute wall estimate correctly includes idle and bounded gates, with overhead explicitly additional.
- **Default preservation:** from the complete origin/main diff, default `apply()` sets no request keys and default `observe()` cannot replace requests. B retains its preview surface, still template, tick scheduling, and direct publication; A retains its original pacing. Frame-size selection, JPEG comment/EXIF, timestamps, reader age limits, writer locking, and cleanup/generation behavior remain intact. Diagnostic callbacks and sidecar serialization add observation overhead, as disclosed.
- **Offline evidence:** the supplied outputs report exit zero for camera_power, duty_cycle, power_map, coresidency, and both launcher suites. Camera tests cover classification, old-build refusal, block selection, mocked cleanup/fail-closed behavior, source contracts, and the timing model. They establish neither Kotlin compilation nor HAL behavior.

Two nonblocking notes remain:

1. **The README states the inferred cause too strongly.** The CAMPOWER2 paragraph in [RobotCam README](/termux-home/robotcam-exp/android/robotcam/README.md) says long preview frames “previously delayed those captures.” The evidence establishes slow publication and stale reads; pipeline delay is an inference. The implementation limitations correctly make this distinction, and that qualification should accompany the README claim.

2. **Some valid non-divisor durations can trip the benchmark’s gap rule.** [CameraService.kt](/termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraService.kt:383) allows quantized intervals, while [camera_power.py](/termux-home/robot/benchmark/camera_power/camera_power.py:162) caps permitted gaps at 1.85 seconds for longer durations. The supplied model reports a 1.966-second transition gap at 983 ms. This does not affect the default 200/500/1000-ms selection, but completion is not guaranteed for every supported duration. The model’s average cadence must not be presented as guaranteed reader freshness.

Kotlin compilation, actual publication cadence through re-convergence, low-light brightness, image quality, and power savings remain explicitly unverified. Those disclosed hardware limitations are not independently blocking under this review procedure.

**APPROVE WITH NOTES**
```

## Commit/push and Actions

Final reviewed commit: `90ef439ecb0ede51788298b96482c1e44c000134`
Message: `RobotCam: manual frame-rate fixes (test only)`
Identity: Git automatic `root <root@localhost.localdomain>`; not amended.

Push exit 0:

```text
To https://github.com/abdelhamidzoghlamii-ui/Pixel-robot.git
   89b5eb3..90ef439  robotcam-camera-power -> robotcam-camera-power
```

Experiment HEAD equals origin/robotcam-camera-power, worktree clean. Main HEAD/origin/main still 7c77f1439544634ede8b8baf92aa03e3b35b24b6; main dirty state unchanged. No benchmark staged.

New [Actions run](https://github.com/abdelhamidzoghlamii-ui/Pixel-robot/actions/runs/37105818442) — completed, conclusion success. Artifact: `robotcam-debug-apk` is available.

Latest queried status: completed, conclusion success. Artifact count 1. Artifact details are recorded below.

Artifact [robotcam-debug-apk](https://github.com/abdelhamidzoghlamii-ui/Pixel-robot/actions/runs/37105818442/artifacts/11267084384), expired=false.

GitHub Actions successful build supplies Kotlin/APK compilation verification; no APK was installed or run.

Actions query result:

```json
{
  "conclusion": "success",
  "databaseId": 37105818442,
  "headSha": "90ef439ecb0ede51788298b96482c1e44c000134",
  "status": "completed",
  "url": "https://github.com/abdelhamidzoghlamii-ui/Pixel-robot/actions/runs/37105818442",
  "workflowName": "robotcam debug APK",
  "artifact_count": 1,
  "artifacts": [
    {
      "name": "robotcam-debug-apk",
      "id": 11267084384,
      "expired": false
    }
  ]
}
```

## Human install and run

Use the new commit's **robotcam debug APK** Actions run on branch `robotcam-camera-power`. Wait for successful build, download **robotcam-debug-apk**, unzip and install app-debug.apk on the Pixel. The CI debug signing key changes between runs: a signature mismatch usually requires uninstall/reinstall, then deleting the old `~/storage/downloads/robotcam` output folder once from native Termux to restore ownership. Open RobotCam to grant Camera/Notifications and stop it. No install/uninstall/delete was performed by the Coder.

Quit all agents, exit Debian/proot, stop robot-chat/chat.py, llama-server and other runners. Motors off, no USB/serial, unplug charger, native Termux foreground and screen on; fix framing/lighting. These commands are for the human only:

```bash
bash ~/robot/benchmark/camera_power/oneshot.sh --smoke
bash ~/robot/benchmark/camera_power/oneshot.sh
bash ~/robot/benchmark/camera_power/oneshot.sh --resume ~/camera_power/run_<UTC>
# smoke resume additionally requires --smoke
# optional switches by name; not in default list:
bash ~/robot/benchmark/camera_power/oneshot.sh --smoke --blocks dump
bash ~/robot/benchmark/camera_power/oneshot.sh --blocks baseline_first,preview,record,fast,off,focus_1m,baseline_last
```

Full default: baseline_first, manual_200 (B), manual_500 (B), manual_1000 (B), mode_A_manual_1000, mode_A_manual_500, mode_A auto, baseline_last; all rate 1. **8 x 120 s = 16 min block time**, <=24 min. Printed wall estimate **21–85 min plus startup/stop/query overhead**, including 5 min idle and 0–8 min skin gate per block. Smoke 8 x20 s, 2 min40 s block time / 7 min40 s including idle plus overhead, no gate wait, explicitly not a measurement. No template/processing/focus/camera-ID/dump blocks by default. Optional --blocks preserves supplied order; other_camera requires --camera-id for an independently exposed BACK ID (provided dump has only back 0/front 1). --camera-id alone adds no block.

Resume requires identical smoke/list/options/helper hashes. Do not resume CAMPOWER1 runs after these changes; preserve old logs. Completed INCOMPLETE blocks remain evidence; moving a block JSON aside explicitly is required to redo it. Saved fail-closed stops refuse further variants/resume until monitoring is repaired and evidence moved aside. The inherited shutdown-before-accounting rule is retained.

## Unverified items / limits

- Successful CI verifies Kotlin/APK compilation; HAL delivery freshness at 100/200/500/1000 ms in both A/B, 30 s recovery on hardware, requested-vs-RESULT boost and low-light brightness, AF/AWB behaviour, motion/YOLO image quality and power savings.
- Exact HAL queue depth/stall mechanism and CaptureFailure events were not in supplied logs; queue explanation is inference. Boost dark-room explanation remains a guess. Smoke CPU numbers are observations, not valid power measurements or a robot-option decision.
- Non-divisor frame periods cannot yield identical one-second capture spacing; modeled selection averages ~1/s with quantized intervals and may approach the reader's freshness ceiling after delivery latency. Source timing models do not prove phone behaviour.
- 10 Hz full JPEG decoding includes repeat work in battery numbers. Compare variants within the same run; do not attribute DUTY1 differences solely to camera changes. Final CPU accounting still races with teardown and can include stop work or miss final app ticks; small differences remain imprecise.
- Fuel-gauge averaging, startup accounting/skew, original helper gate behaviour and screen-off operation remain the disclosed CAMPOWER1 limits. No hardware action was taken; all installed APKs and physical camera state are unchanged by the Coder.

Stopped for the human after the authorized review/publish/report work. No main push, force-push, merge or PR; nothing from benchmark staged.
