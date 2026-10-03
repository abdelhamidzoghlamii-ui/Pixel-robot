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
