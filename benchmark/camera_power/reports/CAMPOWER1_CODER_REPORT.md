# CAMPOWER1 coder report

## Profiles as used

EXECUTION PROFILE
ROLE: Coder
PLATFORM: Codex CLI
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: lite (Ponytail 4.10.1 skill read)
WHY: Bounded Camera2 option switches in RobotCam (default unchanged) plus a camera-only benchmark; needs judgement on Camera2 behaviour.
FALLBACK: Claude Code CLI | claude-opus-5-5 | effort medium | Ponytail lite — not used.

REVIEWER PROFILE — three separate fresh `codex exec` sessions
ROLE: Reviewer
PLATFORM: Codex CLI v0.159.3
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off, explicitly requested and used
WHY: Independent review of Android camera code and measurement validity.
FALLBACK: AGY CLI | gemini-3.1-pro-high | effort N/A | Ponytail off — not used.

Each reviewer used `-s read-only -C /termux-home/robot`, model and effort as above.
Edits were frozen during each review. No self-review substituted for independent
review. Only the two blocking findings were fixed between rounds.

## Base and isolation

Initial HEAD and origin/main both verified as
`7c77f1439544634ede8b8baf92aa03e3b35b24b6`.
Initial status exactly:

```text
 M .gitignore
?? benchmark/duty_cycle/
?? benchmark/power_map/
```

Created `/termux-home/robotcam-exp`, new branch `robotcam-camera-power` from
origin/main. All app/workflow changes occurred there. No robot code, reader,
motor code, USB/serial, canonical status or decisions were changed. The three
pre-existing dirty paths and their outputs were not edited/staged/committed.
`benchmark/camera_power/` is local and excluded from Git by the existing
`benchmark/*` rule; .gitignore was not edited to expose it. Nothing from benchmark
was staged. No PR, merge, main push or force-push was performed.

No actual `su`, `am`, RobotCam, launcher or benchmark was run on hardware.
Offline shell checks ran only mocked commands/temp sysfs; native Termux Python
was used solely for offline tests because Debian Python lacks Pillow.

## Part 1: research written before coding

# CAMPOWER1 research — 2026-10-02, before implementation

Labels apply to each claim: **documented** = Android API or repository source;
**measured** = supplied phone evidence; **guess** = hypothesis, including every
saving estimate. Estimates are reductions from the +1.62 W camera increment,
not promises and not additive. No camera/hardware commands were run by the coder.

**Measured:** DUTY1 PARTS, 120 s: base 0.63 W, camera 2.25 W (+1.62 W),
provider 112.6 CPU-s, app 17.7 CPU-s; YOLO adds 0.70 W. #122: 1/s 2.30 W,
2/s 2.29 W. Today's supplied sidecar: B, 640x480, AE 15–15 fps.
**Documented (repo):** B repeats PREVIEW to a small dropped YUV surface and
submits STILL_CAPTURE once per period. A publishes from one repeating YUV stream.
**Guess:** most avoidable cost is sensor/ISP/HAL/provider streaming work rather
than the 1/s JPEG encoder; CPU accounting does not identify a particular HAL stage.

## Options

| Option | Change and mechanism | Expected saving (guess) | Indoor driving risk (guess) | Runtime? |
|---|---|---|---|---|
| Manual 2–5 fps | **Documented:** AE OFF permits manual exposure/ISO/frame duration; copy converged result exposure and ISO, request 200–500 ms frame duration; verify RESULT duration. **Guess:** fewer sensor/ISP frames may cut provider work. Keep exposure unchanged, rather than extending it to the full period. | 0–1.0 W; HAL may clamp or preserve its internal cadence | Frozen exposure across room transitions, LED flicker, fewer motion samples; periodic AE recovery briefly restores 15 fps; dark-room convergence may fail. Startup includes convergence. | Yes, opt-in `frame_ms` 200–500, AE ON warm-up and re-convergence every 30 s; if unsupported, reject. CONTROL_MODE stays template default so AWB/AF can adapt. CONTROL_MODE_OFF is an alternative requiring manual WB/focus too, deferred. |
| PREVIEW/RECORD for B shots | **Documented:** PREVIEW prioritizes cadence; STILL_CAPTURE prioritizes quality; Android's recording constant is TEMPLATE_RECORD (there is no TEMPLATE_VIDEO_RECORD). **Guess:** lighter per-shot postprocessing or fewer pipeline mode changes. | 0–0.4 W | More noise/less sharpening may affect small YOLO targets; no guaranteed energy saving | Yes, `capture_template` still(default)/preview/record |
| No still | **Documented (repo):** A is one PREVIEW stream and publishes selected frames. **Guess:** eliminates still/preview transitions and second output. | 0–0.5 W | Stream frames may have lower processing quality; fresh-frame age can improve or worsen | Already `mode A`, rate 1; benchmark it |
| Processing OFF/FAST | **Documented:** NR, edge, hot-pixel, aberration, shading and tone-map controls exist; availability varies. OFF disables supported stages; tone-map has FAST rather than a generic OFF. **Guess:** less ISP/software processing. | 0–0.5 W | Noise, hot pixels, vignetting, reduced edge contrast, colour fringes; Laplacian can rise from noise rather than detail, so inspect saved frames | Yes, `processing` default/fast/off; choose only advertised modes; tonemap FAST in both variants; face detection OFF where advertised |
| OIS/video stabilization OFF | **Documented:** separate lens OIS and video stabilization controls. **Guess:** reduced actuator/processing use, potentially no effect when defaults are already off. | 0–0.15 W | More hand/robot shake and motion blur; OIS helps low light | Bundled in opt-in processing profiles, OFF only if advertised |
| Fixed focus | **Documented:** AF OFF plus LENS_FOCUS_DISTANCE in diopters; characteristic minimum-focus-distance bounds the range. **Guess:** saves AF computation and lens motion. | 0–0.1 W | Fixed infinity loses close targets; calibration/approximate diopters, depth of field and dim-room blur matter | Yes, `focus_diopters` explicit finite 0…device limit; default remains automatic. Copying converged AF is a further alternative, deferred |
| AE/AWB locks | **Documented:** supported lock keys freeze automatic settings. **Guess:** less 3A work, HAL may continue statistics anyway. | 0–0.1 W | Bright/dark room changes and mixed lighting cause wrong exposure/colour; should periodically unlock | Possible, deferred: manual exposure already tests AE suppression; independent locks add another state machine |
| Stream sizes/layout | **Documented:** output configuration and supported YUV sizes/minimum durations constrain requests. **Guess:** smaller-only outputs can permit binning and less scaling, but do not guarantee the sensor readout mode. Current B already chooses small same-aspect preview; A removes it. Smaller publish output sacrifices YOLO detail. | 0–0.5 W | Detail loss and changed field of view/crop; preview size alone can have no effect | Mode A exists; publish-size/preview-size switches deferred pending dump |
| Session parameters | **Documented:** supported session keys can be supplied at session creation; changing them can require reconfiguration. **Guess:** expressing FPS up front may avoid reconfiguration, but does not bypass minimum AE range or guarantee steady-state savings. | 0–0.1 W steady state | Setup delays/unsupported combinations | Possible at restart, deferred |
| Other back cameras / ultra-wide | **Documented:** enumerated IDs have individual characteristics; logical IDs need not expose every physical lens as an independently openable ID. Dump facing, focal lengths, focus limit, YUV sizes/fps ranges and manual support. **Guess:** a fixed-focus ultra-wide might use a cheaper pipeline. No ultra-wide ID, sizes or ranges are claimed known on this phone yet. | Could save 0–0.8 W or cost more | Wider geometry/crop/distortion, fewer pixels per person, changed calibration/YOLO area-distance meaning, often worse low-light detail | `camera_id` explicitly selects an enumerated BACK ID; no automatic ultra-wide substitution; dump first |
| Camera off between shots | **Measured:** about 2 s start time supplied. **Guess:** saves idle stream power only at slow intervals substantially longer than startup; bursts may increase startup energy. | At slow rates up to much of 1.62 W, unknown net | Blind intervals and convergence; unsuitable for uncalibrated driving cadence | Option only, deliberately not built |
| Own idea: suppress logical-camera lens switching | **Guess:** choosing an exposed single BACK camera may avoid auxiliary sensor use, but cannot prove that a logical device powers one lens only. | 0–0.5 W, possibly zero | Geometry and low-light changes | Test exposed IDs; physical-output routing deferred |
| Own idea: lower JPEG quality / avoid extra copies | **Documented (repo):** JPEG quality 85, encode only published frames, ~2–10 ms historically. **Guess:** only a small share of this continuous-stream cost. | 0–0.05 W | Compression artifacts affect detection | Possible, deferred to preserve JPEG behaviour and isolate camera pipeline |
| Own idea: lower frame duration with AE priority / vendor controls | **Guess:** unsupported undocumented controls are not a reliable test route. Advertised AE ranges are the safe automatic limit; newer API features depend on device/API support. | Unknown | Exposure instability and portability | Not built; no vendor tags or unadvertised FPS ranges |

## Sources read

- **Documented:** [CaptureRequest](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest), especially SENSOR_FRAME_DURATION, CONTROL_AE_MODE/LOCK, processing and stabilization keys. Manual duration is bounded by sensor/stream limits; RESULT is the evidence of application. Total source-derived description here is under 200 words; savings and robot risks are our guesses.
- **Documented:** [CameraCharacteristics](https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics): capability, supported mode, size, FPS, focus and session-key discovery. No phone characteristic data has been retrieved during this task.
- **Documented:** [CameraDevice](https://developer.android.com/reference/android/hardware/camera2/CameraDevice): PREVIEW, STILL_CAPTURE and RECORD templates.
- **Documented (repo):** CameraService.kt, StartActivity.kt, ControlReceiver.kt,
  robotcam_reader.py, android/robotcam/README.md and FINDINGS.md, camera_heat/README.md,
  DECISIONS #122/#127, duty_cycle and power_map helpers.

## Bounded implementation choice

Six new extras: capture_template, processing, focus_diopters, frame_ms, camera_id,
dump_characteristics. Processing bundles supported lighter stages and stabilization;
the benchmark isolates each extra, with 200/500 ms separate blocks. Default requests,
surfaces, size choice and publish pacing remain unchanged. Diagnostics callbacks
observe the existing requests; they do not claim their results match a JPEG unless
the sensor timestamp matches. Characteristics dump is optional. No robot decision
or camera-off-between-shots implementation. The simpler first experiment is existing
mode A; the new switches let the human distinguish other plausible causes.

## Switches and defaults

| Extra | Values | Default |
|---|---|---|
| capture_template | string still / preview / record | still; B shots only |
| processing | string default / fast / off | default; no request overrides |
| focus_diopters | finite float 0…camera minimum-focus-distance limit | omitted; template AF |
| frame_ms | integer 0 or 200…500 | 0; template AE/FPS |
| camera_id | string, enumerated BACK ID | omitted; same first BACK camera |
| dump_characteristics | boolean | false |

Existing mode B / rate 1 defaults remain. Mode A rate 1 is already supported.
Plain start resets options; option changes use existing restart/generation path.
Manual mode uses AE ON convergence then copies exposure/ISO and requests AE OFF,
retaining exposure time and lengthening duration; every 30 s it re-converges.
AF/AWB are not overridden by manual cadence, but their continued adaptation with
AE OFF is device-dependent and unverified. Invalid syntax stops; unsupported
camera capabilities fail through existing unpublish/retry. Persistent
non-convergence stays AE ON and is labelled accordingly. This is experimental.

`processing` combines advertised lighter NR/edge/hot-pixel/aberration/shading,
tone-map FAST, faces OFF and stabilization OFF. Unsupported keys remain template
defaults. Requested and RESULT values show what was applied, with null for
unreported values. `power` metadata is the latest completed request observation,
possibly preview/previous frame. It is not claimed to be the JPEG's sensor result.
The optional characteristics dump enumerates all independently exposed IDs,
fps, YUV sizes/minimum durations, manual capability, NR/edge/OIS, focal lengths,
focus limit, physical IDs and available session keys. No phone dump was obtained
by the coder; the human smoke run will create it.

## Default preservation verification

Static comparison against origin/main found byte-identical protected method
bodies: JPEG COM/EXIF, YUV copy, frame/preview size selection, onFrame/onPreview,
clock derivation, pacing, onDestroy, camera loss/retry/restart, closeCamera,
generation checks, unpublish/delete, atomic writes and writer loop. Default
request differs only by calling `power.apply`, which sets zero additional keys
for default options. Default still template still resolves to STILL_CAPTURE.
Surfaces, queue sizes, selected fps, publish rate and tick scheduling are unchanged.
`robotcam_reader.py` is byte-identical to HEAD. Whitespace checks passed.

Additional result callbacks observe default captures and add a small unmeasured
overhead; serialization happens only on publication. This verifies request and
contract preservation, not identical wall-clock execution or proven power equality.
The independent reviewers separately confirmed preservation.

## Files and SHA-256

Branch files (the only committed files):

```text
66ee4b27be0787102ff3f50b2678546f6d5fe7b54bfc7ca81c06b13d0c661ce3  .github/workflows/robotcam.yml
7c9f4949ced1279bd2aa74d2e3562b89be894576326e4ae975df85ffc8b62de3  android/robotcam/README.md
a368e0355e1da0f76efb9c161d5d2a9b4a88e8b620083c18553434b8057cdbb4  android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraPower.kt
df8834e55abe761a0525a5fdfdb0265aae1172f85f8c4fe27e8f24d891516cc1  android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraService.kt
3f961d218e08555547336b81e15c0885452948741d2ee8fe3a33b236821499e1  android/robotcam/app/src/main/java/com/pixelrobot/robotcam/StartActivity.kt
```
Local benchmark files (not committed):

```text
5a9106c249ce84adf4ac9c5eb16ae8e1738fefe460ca97a6ba9fd0b71dcbe036  benchmark/camera_power/README.md
c0e43cf8cd94ad6ca1fdf2307af3caa0ad5095b2b05ff4e69825353f8b809e66  benchmark/camera_power/RESEARCH.md
d30dd04c4f5c0e096b257bba7e2eecb15092e50c0ed02345b18c8d17334c004a  benchmark/camera_power/camera_power.py
0540acfb9a8ddedf710755aaa2df64424582ec0577ea6886705046e3e9c1bdfe  benchmark/camera_power/launcher.py
cb557055e51b0aae7811cfe98ce99e8e463bccb6d7db3e273fd6bda3f0499e58  benchmark/camera_power/oneshot.sh
e2fdaaa2d4c54a1b4090bc43b63ff3c0cfd18805c1822d5bc4387a4f2cf9557f  benchmark/camera_power/run_camera_power.sh
c398f90ff173cc2d2a35458e55a9b50705d89d8914a127db04d77c4d7499ce8e  benchmark/camera_power/test_camera_power.py
```
## Checks and build

Debian `python3` first failed with `ModuleNotFoundError: PIL`; no dependency was
added. Existing native Termux Python then ran the offline suite successfully in
all three rounds. Tests cover block list/budget/extras mapping, luma/Laplacian,
trapezoidal power/gaps, CPU identity math, skin slope, old APK/options/proot
refusals, sidecar race, mocked block loops/saved images/core-loss cleanup, atomic
completed-block preservation, inherited fake launcher refusals/timeout/signals,
shutdown before stalled/failed accounting, and fail-closed abort/resume cleanup.
The ONNX Runtime Android platform warning came from helper import; no detector,
weight loading, model inference, Gemma server or selector was constructed/run.

Final offline output, untruncated:

```text

block order/duration/baseline no extras/all six extras: PASS
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
CAMPOWER1 offline checks: PASS (no hardware)
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(

Exit: 0

```

Structural verification output:

```text

Protected JPEG/clock/generation/stop/YUV/size/pacing bodies equal to base: PASS
Default request adds only no-op power.apply; default still template unchanged: PASS
robotcam_reader.py byte-identical to HEAD: PASS
Gradle absent; ANDROID_HOME / ANDROID_SDK_ROOT unset; local APK build not run.
CI compile and HAL behaviour remain unverified.
66ee4b27be0787102ff3f50b2678546f6d5fe7b54bfc7ca81c06b13d0c661ce3 /termux-home/robotcam-exp/.github/workflows/robotcam.yml
7c9f4949ced1279bd2aa74d2e3562b89be894576326e4ae975df85ffc8b62de3 /termux-home/robotcam-exp/android/robotcam/README.md
df8834e55abe761a0525a5fdfdb0265aae1172f85f8c4fe27e8f24d891516cc1 /termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraService.kt
a368e0355e1da0f76efb9c161d5d2a9b4a88e8b620083c18553434b8057cdbb4 /termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam/CameraPower.kt
3f961d218e08555547336b81e15c0885452948741d2ee8fe3a33b236821499e1 /termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam/StartActivity.kt
c0e43cf8cd94ad6ca1fdf2307af3caa0ad5095b2b05ff4e69825353f8b809e66 /termux-home/robot/benchmark/camera_power/RESEARCH.md
d30dd04c4f5c0e096b257bba7e2eecb15092e50c0ed02345b18c8d17334c004a /termux-home/robot/benchmark/camera_power/camera_power.py
0540acfb9a8ddedf710755aaa2df64424582ec0577ea6886705046e3e9c1bdfe /termux-home/robot/benchmark/camera_power/launcher.py
cb557055e51b0aae7811cfe98ce99e8e463bccb6d7db3e273fd6bda3f0499e58 /termux-home/robot/benchmark/camera_power/oneshot.sh
e2fdaaa2d4c54a1b4090bc43b63ff3c0cfd18805c1822d5bc4387a4f2cf9557f /termux-home/robot/benchmark/camera_power/run_camera_power.sh
5a9106c249ce84adf4ac9c5eb16ae8e1738fefe460ca97a6ba9fd0b71dcbe036 /termux-home/robot/benchmark/camera_power/README.md
c398f90ff173cc2d2a35458e55a9b50705d89d8914a127db04d77c4d7499ce8e /termux-home/robot/benchmark/camera_power/test_camera_power.py

```

`gradle` was not found; ANDROID_HOME and ANDROID_SDK_ROOT were unset. Therefore
the conditional local assembleDebug command was not run. SDK presence elsewhere
was not established. GitHub Actions is the actual compile check.

GitHub Actions subsequently passed assembleDebug and APK artifact upload for the
reviewed commit. This clears the compile gap mentioned in the frozen verification
and reviews; runtime/HAL/image quality/power remain unverified.

Actions status captured after push:

```json

{
  "conclusion": "success",
  "databaseId": 37076257885,
  "headSha": "89b5eb366bec61d4c60660948298344be52086fe",
  "status": "completed",
  "url": "https://github.com/abdelhamidzoghlamii-ui/Pixel-robot/actions/runs/37076257885",
  "build_debug_apk": "success",
  "upload_robotcam_debug_apk": "success",
  "completedAt": "2026-10-02T23:11:04Z"
}

```

## Independent reviews, verbatim

The coder did not edit during reviews. Rounds 1 and 2 requested changes for
shutdown ordering and fail-closed continuation respectively; both received
mocked regression checks. Round 3 APPROVE WITH NOTES authorizes no action by
itself; commit/push proceeded under the owner's task-specific authorization.
Reviewer commands sometimes returned exit 182 without diagnostics; targeted
reads succeeded and all reviewers returned substantive verdicts from the packet.
No explicit sandbox mount error, permission/auth/quota failure without verdict,
or incomplete review was treated as approval. Bubblewrap-not-on-PATH warnings
said the bundled binary would be used; full stderr retained separately.

### Round 1

```text

Read-only independent review, Ponytail off. Repository instructions were readable. Some later file-read commands returned exit 182 without diagnostics; targeted `rg` reads succeeded, and the supplied frozen packet provided the remaining context. No edits, hardware commands, benchmarks, launchers, or additional reviewers were run.

One blocking finding:

- **P1 — Camera shutdown waits behind monitoring and accounting after a safety limit.** In [camera_power.py:198](/termux-home/robot/benchmark/camera_power/camera_power.py:198), `run_block()` stops monitoring, joins threads with 60-second timeouts, then takes another CPU snapshot and battery sample **before** calling `dc.stop_camera()`. The inherited CPU query can itself wait 60 seconds. After CRITICAL thermal status, battery limit, sensor failure, or a termination signal, RobotCam can therefore keep streaming while shutdown waits, with limit checks no longer active. Ensure STOP and end-check/force-stop are attempted promptly on every exit, independently of monitor draining or accounting failures. Add an offline check that simulates stalled/failed accounting after a limit and verifies camera shutdown occurs first.

Nonblocking notes:

- [camera_power.py:185](/termux-home/robot/benchmark/camera_power/camera_power.py:185) polls the JPEG roughly ten times per second; the reader fully decodes repeated images before they are classified as repeats. Reported battery watts therefore include this reader workload. Disclose that overhead when comparing these numbers with DUTY1 or attributing savings to RobotCam.
- In the supplied candidate, omitting extras preserves request keys, templates, surfaces, sizes, and publication pacing. JPEG comments, CLOCK_BOOTTIME handling, reader independence from the sidecar, and publication/generation guards remain preserved.
- Diagnostics correctly disclose that the latest capture result may describe preview or a previous frame. Requested duration is not presented as proof of applied sensor cadence.
- Local compilation, HAL behavior, image quality, and savings remain explicitly unverified. Those disclosed gaps are not blockers. Supplied offline results were inspected, not independently rerun.

REQUEST CHANGES


```

### Round 2

```text

One blocking safety finding.

- **P1 — A fail-closed sensor stop does not abort the run.** In [camera_power.py:311](/termux-home/robot/benchmark/camera_power/camera_power.py:311), `main()` saves the block and advances to the next variant even when `b['heat_stop']` is `('fail_closed', ...)`. Only missing startup frames trigger an immediate abort. If battery-temperature monitoring fails after startup, the camera stops after five seconds, then can reopen for the next block with a new five-second grace period. Missing skin/status monitoring similarly resets its sixty-second grace period; the thermal gate can eventually return WARM START with missing readings. Recording INCOMPLETE at the end does not prevent these restarts.

  **Required fix:** retain the block evidence and abort before launching another variant following `fail_closed`. Add a mocked orchestration check proving that a fail-closed block with valid startup frames prevents the next camera start and still performs cleanup.

Nonblocking observations:

- The default profile sets no additional request keys. The supplied diff preserves default templates, surfaces, size selection, pacing, JPEG comments, CLOCK_BOOTTIME handling, and publication/generation guards. The reader remains JPEG-only.
- Sidecar/JPEG pairing identifies the published frame, while result metadata explicitly describes the latest completed request. The documentation appropriately avoids claiming a JPEG-matched sensor result or achieved slow cadence.
- Manual AE leaves AF/AWB at template settings, but their behavior with AE OFF is device-dependent. Treat continued adaptation as unverified during image inspection. [Android Camera2 reference](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest#CONTROL_AF_MODE)
- Final CPU sampling races with teardown, so it can include stop work or miss the app and mark the block incomplete. The disclosed timing limitations warrant caution when comparing small differences.

All twelve supplied file hashes match the readable candidate. I reviewed the supplied offline test output; I did not rerun tests, build, launch benchmarks, or access hardware. Missing build/HAL/power measurements are disclosed and are not blockers.

**REQUEST CHANGES**


```

### Round 3

```text

No blocking findings in the frozen round-3 candidate.

- Both prior safety blockers are resolved. [Shutdown now precedes accounting drain](/termux-home/robot/benchmark/camera_power/camera_power.py:214). [Fail-closed stops retain evidence and abort before another variant](/termux-home/robot/benchmark/camera_power/camera_power.py:315); resume refuses the saved fail-closed block. The supplied mocked checks cover these paths.
- Omitting extras preserves request keys, templates, targets, sizes and publication pacing. JPEG comments, CLOCK_BOOTTIME handling, JPEG-only reader behavior, and unpublish/generation guards remain preserved. Default result callbacks introduce observation overhead.
- Diagnostics distinguish requested values from RESULT values and disclose that the latest result may describe preview or a previous frame. Neither sidecar pairing nor requested duration is presented as proof of JPEG-matched results or achieved sensor cadence.

Nonblocking measurement notes:

- [Polling every 0.1 seconds](/termux-home/robot/benchmark/camera_power/camera_power.py:212) fully decodes repeated JPEGs before rejecting repeats. Battery watts include that reader workload; account for it when comparing against DUTY1 or attributing differences to RobotCam.
- Final CPU accounting runs concurrently with teardown. It can include stop work or miss the app’s final ticks; small CPU differences require caution. Missing post-start app snapshots are marked incomplete.

All twelve supplied hashes match the readable files. Repository instructions were readable; two subsequent `sed` reads returned exit 182 without diagnostics, and targeted `rg` reads succeeded. I inspected the complete packet and supplied verification output without rerunning tests, building, launching benchmarks, accessing hardware, editing files, or invoking another reviewer.

Compilation, HAL behavior, image quality and savings remain explicitly unverified. These disclosed gaps are not blockers. This verdict is a review finding, not commit or push authorization.

**APPROVE WITH NOTES**


```

## Push result

Final reviewed branch commit: `89b5eb366bec61d4c60660948298344be52086fe`
Message: `RobotCam: camera power variants (test only)`.
Commit identity was Git's automatic `root <root@localhost.localdomain>`; it was
not amended. Git push succeeded with the environment's existing credentials:

```text
git push origin robotcam-camera-power
To https://github.com/abdelhamidzoghlamii-ui/Pixel-robot.git
 * [new branch] robotcam-camera-power -> robotcam-camera-power
```

No credentials were added. Worktree clean after push. Main remains at the base
hash with the exact original status above. Benchmark is not in the branch.

## APK retrieval and install — human only

Open [the Actions run](https://github.com/abdelhamidzoghlamii-ui/Pixel-robot/actions/runs/37076257885)
or the repository Actions tab, workflow **robotcam debug APK**, branch
**robotcam-camera-power**, commit above. Wait for a successful run; under
Artifacts download **robotcam-debug-apk**, unzip, and open `app-debug.apk` on the
Pixel (allow that file manager/browser to install unknown apps), or use
`adb install -r app-debug.apk` from your computer.

CI uses a throwaway debug key. If installation fails with a signature mismatch
(`INSTALL_FAILED_UPDATE_INCOMPATIBLE`), uninstall the old RobotCam first and
install again. In native Termux, delete the old
`~/storage/downloads/robotcam` folder once so the new install can own its files;
this deletes old frames/characteristics. Open RobotCam, grant Camera and
Notifications, and stop it before the run. Termux needs its existing shared
storage/All-files access. No install/uninstall was performed by the coder.

## Benchmark commands — human only, native Termux

Quit all agents, exit Debian/proot, stop chat.py/llama-server and other runners,
unplug charger, motors off, Termux foreground/screen on. Keep mount, framing and
lighting fixed. Launcher handles 5 min idle, root zone logger, timeout
read/set/restore, battery/refusal checks. Python refuses proot and old APKs without
the variant field. No YOLO/Gemma/selector workloads run.

```bash
bash ~/robot/benchmark/camera_power/oneshot.sh --smoke
bash ~/robot/benchmark/camera_power/oneshot.sh
bash ~/robot/benchmark/camera_power/oneshot.sh --resume ~/camera_power/run_<UTC>
bash ~/robot/benchmark/camera_power/oneshot.sh --smoke --resume ~/camera_power/run_<UTC>_smoke
```

11 camera blocks: baseline noextras first, preview, record, fast, off, fixed focus
1 m, manual200, manual500, modeA rate1, dump-only, baseline noextras last.
Full nominal block time22min; estimate27–115min including5min idle and up to8min
skin gate per block, plus startup/stop/query overhead. Smoke20s/block, no gate
wait, labelled **SMOKE: not a measurement**, still5min idle and all limits.

After smoke, inspect `~/camera_power/run_<UTC>_smoke/characteristics.json`. To test
one independently enumerated BACK camera, add `--camera-id ID` to smoke/full and
resume commands (same ID on resume): 12blocks/24min nominal block time.
No guessed ultra-wide ID is used; without this argument, camera-ID switching
remains unmeasured. A physical ID not independently enumerated is not selectable.

Resume preserves completed JSONs and redoes the interrupted block; requires same
smoke/camera-ID and helper hashes. A saved fail-closed sensor stop refuses resume
before another variant: repair monitoring and move that failed block JSON aside
before retrying. The original failure evidence should be retained under another
name. Other completed incomplete blocks remain evidence and need the same explicit
move-aside to redo. No sensor failure gets a new grace period by advancing blocks.

Run outputs: `~/camera_power/run_<UTC>[_smoke]/` (metadata/hashes, raw block JSON,
per-frame diagnostics, luma/sharpness/age, three images per block, characteristics,
report.txt). Launcher console/thermal log in `~/camera_power/`. Saved images are
validated upright decoded frames re-encoded quality95, not original JPEG bytes.

## Measurement cautions and unverified items

- Reported power includes 10Hz JPEG polling/decoding, even repeats, plus samplers
  and image metrics. This is additional reader work compared with DUTY1's rate1
  pacing. Do not attribute a difference versus DUTY1 entirely to RobotCam; compare
  variants within this run, with first/last baseline drift and warm starts visible.
- Final CPU accounting is concurrent with STOP/teardown; it may include stop work
  or lose final app ticks. Missing post-start app/provider snapshots and errors
  mark incomplete. Small CPU differences are not precise energy attribution.
  Raw snapshot identities/times are retained; transient processes can be missed.
- Latest capture result is not necessarily the paired JPEG's result. Inspect both
  requested and RESULT values, AE mode and result timestamp; manual duration may
  be clamped/ignored, and convergence/AF/AWB behaviour needs the phone.
- Fuel-gauge averaging and current sign calibration remain unverified; sampler
  is0.5s, uncovered edges/gaps>1.5s yield n/a. STOP time includes3s quiet-frame
  verification and force-stop. Skin/status are heat evidence; CPU zones are only
  fault readings. Limits/short/failed blocks are marked incomplete.
- No new power measurements, phone CameraCharacteristics dump, startup/stop
  timings, HAL-control validation, screen-off validation, other-camera geometry,
  motion/low-light image quality, YOLO accuracy or savings were measured.
- Static default preservation and independent review do not prove identical
  timing/power. No camera variant is chosen for the robot. Camera-off-between-shots,
  extra size/session/AE-AWB-lock controls were deliberately left for later evidence.

The simplest available first comparison is existing mode A; the additional
switches isolate other plausible pipeline costs without changing robot defaults.
