# DECISIONS

Append-only log of settled tradeoffs. One line per decision plus a short why.
**Never rewrite existing entries** — supersede them with a new numbered entry
that references the old one.

Format: `NN. [area] decision — why`

---

## Model & inference

1. **Gemma 4 E2B Q4_K_M as the robot default** — 11-12 tok/s vs E4B's 7.2 tok/s
   on the same config; speed matters more than depth for navigation. (`server_manager.py: setup_q4`)
2. **Gemma 4 E4B kept as an opt-in quality mode, not the default** — measured 9.9s
   warm cycle vs E2B's 5.1s; too slow for the control loop, good for chat. (`setup_e4b`)
3. **Q4_K_M over Q3_K_S for E4B** — Q3_K_S measured *slower* (5.8 vs 7.2 tok/s)
   despite being smaller; llama.cpp's ARM NEON kernels are optimised for Q4_K_M.
4. **4 threads, not 8** — token generation is memory-bandwidth bound on Tensor G2;
   8 threads pulled in the slow LITTLE cores and dropped throughput.
5. **`--parallel 1`** — default 4 parallel slots split the KV cache and cut
   single-request speed.
6. **`--swa-full` instead of `--cache-ram 0`** — Gemma 4's sliding-window attention
   was invalidating the prompt cache every request, forcing a full ~15s reprocess
   of the system prompt. `--swa-full` restores cache reuse; repeat-prefix prompt
   eval dropped from ~15s to ~0.1s. (Supersedes the earlier `--cache-ram 0`
   workaround; the comment in `server_manager.py` is stale and still mentions it.)
7. **`--mlock` rejected** — pinned the model in RAM but measured no speed gain and
   left only ~233 MB free, which is unsafe alongside YOLO and the camera.
8. **mmproj file deleted** — Gemma 4 has native vision; the separate multimodal
   projector was redundant, freed ~940 MB.
9. **Both LoRA adapters rejected (voice and nav)** — fine-tuned adapters scored
   *worse* than the base model (voice 80% vs 83%, nav 31% vs 38%); base Gemma 4 was
   already at the task ceiling and the adapters degraded generalisation.
10. **OpenCL GPU backend abandoned** — `libOpenCL.so` exists on the device but
    returns "platform IDs not available" from Termux userspace; the OpenCL-enabled
    build was also slower than the CPU-only build, so it was rebuilt without it.
11. **KleidiAI / SVE / SME2 path not viable** — Tensor G2 lacks those ARM
    extensions (`/proc/cpuinfo` shows asimd/asimddp only).

## Vision

12. **yolo11m.onnx as the detection model** — chosen over lighter variants for
    accuracy on indoor scenes; runs per-cycle at 640×640. (`detect_person.py`)
13. **CONF 0.35 / IOU 0.45** — detection and NMS thresholds as coded.
14. **Coarse distance from bounding-box area, not a depth model** — area ratio
    buckets (>0.3 very close … >0.03 medium) are free and good enough for the
    Python safety rules; stereo/known-height estimation handles the finer cases.
15. **Position from horizontal thirds of the 640px frame** — cx<213 left,
    cx>427 right, else center. Simple and deterministic.
16. **Single-photo distance via known real-world object heights** — avoids needing
    a second camera; `REAL_HEIGHTS` table in `stereo_depth.py`.
17. **Stereo disparity below 2px is discarded** — below that the triangulation is
    noise; falls back to the single-photo estimate.
18. **FOCAL_PX = 500 is a placeholder, not a measurement** — flagged
    "calibrate later" in the source; all derived distances are approximate until
    a known-distance calibration is done.

## Navigation architecture

19. **Dual-rate split: Python for deterministic rules, Gemma for strategy only** —
    benchmarking Gemma on rules it should never see scored 55%; moving person /
    obstacle / room-arrival decisions into Python and leaving Gemma only the
    exploration and mission-complete calls scored 93%.
20. **Gemma is called on triggers, not every cycle** — interval, person seen, goal
    reached, or new room mapped; keeps both latency and heat down.
21. **Vision (image payload) sent on a narrower trigger set than text** — images
    cost far more prompt tokens, so only the every-5 / person / goal / new-room
    cases attach a photo.
22. **Room identity inferred from single signature objects** — refrigerator→kitchen,
    couch|tv→living room, bed→bedroom, toilet→bathroom. Cheap and sufficient; a
    trained room classifier was deferred.
23. **`nav_sim.py` left as a standalone simulation on Qwen 2.5 3B** — it predates
    the Gemma pipeline and is kept for thermal/loop experiments, not merged into
    the live path.

## Voice

24. **Whisper.cpp `ggml-base.bin` retained over Gemma 4's native audio input** —
    Gemma audio landed in llama.cpp but needs the deleted mmproj re-downloaded
    (~940 MB) and had reported stability crashes; the working Whisper path was not
    disturbed.
25. **Fixed 5-second capture window** — simple and predictable; VAD-based silence
    trimming deferred.
26. **JSON action array as the command contract** — `PARSE_SYS` constrains Gemma to
    emit only a JSON array, with `[` pre-injected into the prompt to force the
    format.

## Platform & safety

27. **Root restored via Magisk `Select and patch a file`, not the old images** —
    an OTA to Android 17 (CP2A.260705.006) overwrote the Magisk boot patch. The
    March boot images in Downloads were from an older build; anti-rollback on
    recent bootloaders makes flashing an older Android 16 image a brick risk, so
    they were deleted and a fresh init_boot from the matching A17 factory image
    was patched and flashed.
28. **Only `init_boot` is ever flashed** — smallest recoverable operation; the
    stock A17 `init_boot.img` is kept on the laptop as the rescue file.
29. **BIG-core thermal zone (zone9) is the thermal signal** — battery temperature
    lags SoC heat by a minute or more and dampens the swing; token-rate collapse
    and zone9 both track the real throttle.
30. **Rest between inference calls makes the system *faster*, not slower** —
    measured 0s rest → 8.7 tok/s sustained with throttling, 2s rest → 11.1 tok/s
    with only 6% droop, because the chip stops self-throttling.

## Hardware

31. **2× MX1508 chosen over the L298N** — two dual H-bridges cover all four motors,
    no screw terminals, and no separate enable pins needed.
32. **MX1508 speed control is PWM on the IN pins** — the part has no ENA/ENB;
    forward = PWM on IN1 with IN2 low, reverse = the inverse.
33. **GPIO map: 16/17/18/19 → front pair, 21/22/23/25 → rear pair.**
34. **2S 18650 pack behind an HW-391 20A BMS** — pack measured 8.2 V with cells
    balanced at 4.1 V / 4.1 V.
35. **Separate buck converters for logic and sensors** — 5 V rail for the ESP32
    (set to 5.03 V no-load before connecting anything), 3.3 V rail for sensors.
36. **Star ground required at final assembly** — a marginal ground return on the
    buck path produced a false 3.76 V reading on the ESP32 3V3 pin and left GPIO2
    floating high; both vanish on USB power, confirming the board itself is fine.
37. **EQV 4WD mecanum kit, 8.2V pack is the intended chassis voltage** — motors
    rated 1.5–12V (nominal 6.0V/100rpm, kit designed for 7–12V external), so the
    8.2V rail is safe by design. Caveat: MX1508 tops out ~10V, so 8.2V runs the
    driver near its ceiling (~82%), and nominal-6V motors at 8.2V sustained run
    hot — may cap effective duty to ~6V-equivalent for longevity once driving.
    (Bench MOTOR_SPEED 130 ≈ 51% duty is gentle.)
38. **Decoupling caps are mandatory, not optional** — 100µF electrolytic + 0.1µF
    ceramic across each MX1508 VCC/GND, plus 470–1000µF bulk at the P+ junction.
    Without them, PWM motor-current spikes collapse the 8.2V rail; the buck feeding
    the ESP32 from that rail brownout-resets the MCU every PWM cycle, producing a
    start-stall-retry stutter. Isolated by elimination: motor direct to pack, to
    BMS, and via MX1508 with a hardwired IN1 jumper all ran continuously; ESP32 DC
    drive ran continuously; ESP32 PWM failed on buck power but ran clean on USB.
    Root cause was shared-rail brownout — not wire gauge, not the drivers, not the
    code.
39. **Interim bench/drive config until caps arrive** — ESP32 powered from phone USB
    (supply independent of the motor rail), motors from the battery rail via the
    MX1508s. Buck→VIN stays disconnected. Verified: all four motors run continuously
    under PWM (1kHz, 8-bit, speed 130) in this configuration.
40. **Physical driver assignment is LEFT/RIGHT, not front/rear** — supersedes the
    wording of #33; the GPIO map itself is unchanged. MX1508 #1 = right side:
    chA P16/P17 = rear right, chB P18/P19 = front right. MX1508 #2 = left side:
    chA P21/P22 = rear left, chB P23/P25 = front left. Motor index for mecanum
    mixing: FL=3, FR=1, RL=2, RR=0.

## Navigation (cont.)

41. **The 45cm obstacle threshold has no benchmark source — do not apply** — it is
    inferred from `nav_logic_test.py:162` test "obstacle center" (40cm, expects
    BACK), which fails under the function's default `obstacle_dist=25`; 45 is
    back-derived to make that one case pass, not measured. Moot regardless — no
    distance sensor is fitted (see STATUS "Pending").

## Hardware (cont.)

42. **HC-SR04 ultrasonic fitted and verified** — TRIG on P27 direct (3.3V drive is
    sufficient for the trigger); ECHO on P26 via a 1kΩ/2kΩ divider (5V × 2/3 =
    3.33V) — the divider is mandatory, ECHO is a 5V output into a 3.3V pin. Sensor
    VCC from the ESP32 5V/VIN pin (phone USB), not the 3.3V rail and not buck #2,
    so the sensor stays off the motor rail. Readings verified against a tape
    measure. Supersedes the "moot regardless" caveat in #41 — the threshold
    question is live again once `HAS_ULTRASONIC` is enabled, and 45 is still
    unsourced.
43. **Driver #2 channel-to-corner assignment corrected** — supersedes #40's
    left-side detail; the GPIO map and the left/right split are unchanged.
    Measured by driving each channel alone: ch0 P16/P17 = RR, ch1 P18/P19 = FR,
    ch2 P21/P22 = FRONT left, ch3 P23/P25 = REAR left. #40 had the left pair
    reversed. Left motors spin backward on positive drive, so indices 2 and 3 are
    inverted in firmware. Motor index map: RR=0, FR=1, FL=2, RL=3.
    (Verified in `mode1_simple.ino`.)

## Serial transport

44. **Board is CP2102 (`10c4:ea60`), not CH340** — endpoints 0x01 OUT / 0x81 IN;
    init is `IFC_ENABLE`(0x00) → `SET_BAUDRATE`(0x1E, 32-bit value) →
    `LINE_CTL` 8N1(0x03). `usb.util.claim_interface()` is mandatory — without it
    control transfers succeed and every bulk read times out silently. DTR/RTS left
    untouched; asserting them resets the ESP32. `motors.py` transport rewritten;
    CH340 version backed up at `motors.py.ch340.bak`. The old transport could never
    have worked with this board, so no serial command from the phone had ever
    round-tripped before this session.
45. **Baud 115200, chosen not inherited** — CP210x sets the rate as an explicit
    32-bit value, so the CH340 divisor-table reasoning that produced 9600 no longer
    applies. Verified phone-side and in the Arduino serial monitor.
46. **All USB motor operations require root plus an explicit `LD_LIBRARY_PATH`** —
    `/dev/bus/usb` is not readable by Termux's app UID, and root does not inherit
    Termux's environment. Applies to `motors.py`, `teleop.py`, `main.py`,
    `run_mission.py` — not to the LLM server, benchmarks, or vision scripts.

## Firmware

47. **`WATCHDOG_MS` = 1000, not 5000** — a dropped link must not leave the chassis
    driving for five seconds. Paired with a host-side keepalive: `motors.py`'s
    duration helpers re-send the command every 200 ms via `_hold()`, so moves longer
    than the watchdog are not cut short. Verified — a 3.0 s forward runs 3.1 s,
    continuous.
48. **Diagonal commands `FWD_L` / `FWD_R` / `BACK_L` / `BACK_R` added** — the
    mecanum mixing already handled them; only the command names were missing. Each
    drives one diagonal pair — two wheels, not four, is correct.
49. **`mode2_auto.ino` corner map corrected to match #43** — RR=0, FR=1, FL=2, RL=3,
    with `INVERT` indices 2 and 3 true. It had been carrying the superseded #40
    mapping with `INVERT` all false, so some commands drove the wrong physical wheel
    and one left motor ran backwards. Plausible cause of the `BACK_R` single-wheel
    fault — NOT yet re-tested after the fix.
50. **`DIST:` emits -1 when no echo returns within ~4 m** — so the host can
    distinguish "nothing in range" from "very close". The previous build emitted
    nothing in that case, leaving the last reading stale.

## Navigation (cont. II)

51. **`Robot.get_distance()` sanitises the sensor** — -1 maps to 400 (clear to
    sensor max), and a reading older than 1.0 s maps to 999 (unusable, treated as
    clear). Without this, `-1 < 15` is true and an open corridor triggers BACK on
    every cycle — a bug that was latent only while distance was pinned at 999.
52. **`navigate_rules` rewritten against live sensor input** — a blocked path
    strafes first (mecanum holds heading, so YOLO keeps the same view), rotates on
    axis if strafing isn't clearing, and sets `nav_stuck` after repeated avoidance
    so `run_cycle` hands the strategy call to Gemma. This is a rewrite, not a port
    of `nav_logic_test.py` — see #55.
53. **Gemma may not override a Python safety move** — `run_cycle` previously
    overwrote `navigate_rules`' output unconditionally, so an obstacle BACK could be
    replaced by the model's FORWARD, a direct violation of #19 that was harmless
    only while distance was pinned at 999. Safety moves now bypass the Gemma call
    entirely unless `nav_stuck` is set.
54. **`OBSTACLE_DIST` stays 25 cm** — coast distance measured as effectively zero
    (gearmotors, duty 130), so the binding constraint is sensing latency (200 ms
    ping interval plus cycle time), not braking. 25 cm carries adequate margin.
    #41 stands: 45 remains unsourced and rejected.

## Benchmarks

55. **`nav_logic_test.py`'s 93% does not describe `main.py`** — it matched the
    literal string `'obstacle'` in synthetic scene text, a label YOLO cannot emit,
    and ran with no distance sensor. The live rules take obstacles from
    `get_distance()`. `main.py`'s nav logic is currently UNBENCHMARKED.

## Platform (cont.)

56. **Four bare `except:` clauses in `main.py` were swallowing KeyboardInterrupt** —
    Ctrl+C during a Gemma call returned a default move and the loop continued. Fixed
    to re-raise. This was the emergency stop.
57. **`llama-server` and `teleop.py` both bind port 8080 and cannot run
    concurrently** — autonomous is unaffected; `main.py` is a client of the server,
    it binds nothing.

## Scope

58. **Mode 2 (video/audio teleop) parked indefinitely** — MJPEG encoding via IP
    Webcam alongside the AP and USB-OTG heats the Pixel enough to compete with
    llama.cpp and YOLO for thermal headroom. Not a primary objective. Firmware and
    control path are mode-agnostic, so nothing is lost by parking it. Note: this
    parks the video/audio path only. `mode2_auto.ino` is the serial-command firmware
    for both teleop and autonomous and is unaffected by the name.

## Navigation (cont. III)

59. **Avoidance escalation uses an explicit counter (`blocked_n`), not state
    inferred from `last_moves`** — supersedes #52's `nav_stuck`-on-repeated-
    avoidance mechanism. The first implementation read `last_moves[-4:]` and
    flipped `avoid_side` whenever four avoidance moves were present — true on every
    subsequent cycle, so it alternated LEFT/RIGHT forever (rotations cancelling each
    other) and set `nav_stuck` every cycle, calling Gemma at ~5s per cycle. Found by
    `nav_test.py`, never ran on hardware.
60. **The robot never REVERSES to avoid an obstacle** — with one forward-facing
    sensor, BACK drives fully blind and away from the only direction the robot can
    see. Rotation on axis translates the chassis not at all and is always safe while
    blocked. Strafing is blind sideways and is NOT covered by this rule — see #61.
    Applies to both the `<15` branch and the ladder terminal. Revisit if the six IR
    sensors (side + rear) are fitted.
61. **Escalation ladder: 2 strafes, 4 rotations one way, flip side, 6 rotations back
    through and past centre, then restart** — blocked continuously for 60s
    (`ESCAPE_TIMEOUT`) → STOP. Time-based, not cycle-based, because cycle duration
    varies widely once Gemma calls land. The two opening strafes are a DELIBERATE
    bounded risk, not an oversight: mecanum holds heading so YOLO keeps the same
    view (#52), which rotation destroys. They are blind sideways — capped at 2
    cycles for that reason, and the first thing the side IR sensors should make
    sensed. Do not remove them citing #60; #60 is about BACK. Supersedes #52's
    strafe-then-rotate sequencing, which had no step counts.
62. **Gemma is consulted once per blockage episode, at the side-flip, guarded by
    `asked_gemma`** — supersedes #52's per-avoidance escalation: Gemma is consulted
    once per blockage episode, not on every avoidance. The flag clears only when the
    path actually clears. Chosen over per-cycle consultation (thermal cost, defeats
    #20) and over never re-asking. Interim — revisit once the loop runs on hardware.

## Testing

63. **`nav_test.py` calls `navigate_rules` directly with synthetic distances** — the
    obstacle branches cannot be reached in `run_mission.py --dry` because
    `self.motors` is None pins distance at 999, so every safety path was unexercised
    until this harness existed. It found #59.

## Calibration

64. **Forward motion is modelled as IT1 with dead time** — velocity responds to a
    PWM step as PT1 (drivetrain inertia), position is its integral.
    `G(s) = K_I/(s(T1·s+1))·e^(−Tt·s)`. Feedforward only — there is no odometry, so
    the loop cannot be closed during normal operation. On synthetic data (K_I
    42cm/s, T1 0.18s, Tt 0.09s), commanding a 10cm move from the naive d/v duration
    travels 1.98cm — an 80% distance shortfall, equivalently a 52% duration error.
    At 100cm the duration error falls to 10%. So the transient dominates exactly at
    the short corrective moves the ladder issues most. Source: `identify_it1.py`'s
    duration table. `identify_it1.py` (numpy-only grid fit, K_I solved in closed
    form) validated on synthetic data: recovered 42.06/0.1788/0.0953 against true
    42/0.18/0.09.
65. **`DIST_EVERY_MS` 200 is adequate for K_I but not for T1/Tt** — the whole
    transient is 2-3 samples and Tt's quantisation error exceeds Tt. Use 50ms (and
    `ECHO_TIMEOUT` 12000µs, ~2m) for calibration runs. `log_run.py` dedupes on
    `dist_at` so it records true sensor sample times, and warns when the interval is
    too coarse to fit the transient. CALIBRATION BUILDS ONLY. Restore `ECHO_TIMEOUT`
    to 25000 before driving: at 12000µs the sensor's max range is ~2m, so `DIST:-1`
    means ">=2m" while #50 defines it as ">=4m" and #51 maps it to 400cm — the
    sanitiser would assert 4m of clearance on 2m of evidence. Not hazardous at the
    25cm threshold, since both readings are "clear", but it feeds a false number to
    `GEMMA_SYS` and would become hazardous if any rule ever compares against a long
    distance.

## Licensing & docs

66. **`LICENSES.md` is canonical for the licence allowlist; `APP_CLAUDE.md` §0.5
    carries a derived copy on purpose** — Claude Code auto-reads only `CLAUDE.md`,
    so a pointer to a second file would sometimes go unread, and an unread
    allowlist is how an AGPL dependency gets added. Duplication accepted; drift
    handled by marking the copy derived: on mismatch `LICENSES.md` wins and the
    agent reports rather than reconciles.

## App blockers

67. **Blocker A1 passes: root-free USB Host serial confirmed** — Android 17, unrooted
    app sandbox. Verified with *Serial USB Terminal* (kai-morich) against an ESP32-S3
    via CH343 bridge (`1a86:55d3`), 115200 8N1: `PING` → `ALIVE` in 5 ms, with the
    library asserting DTR and resetting the board at connect (boot banner, then
    `READY` 275 ms later) — so control transfers work, not just stream reads. Tested
    on the S3 rather than the robot's CP2102 because the blocker is the Android
    sandbox, not the bridge chip, and the CP2102 path is already proven on this phone
    with root (#44). Residual CP2102 work is a manifest VID:PID filter plus driver
    selection — a follow-up, not a gate.
68. **Blocker A2 passes: Gemma sustained throughput inside a real app process** —
    Gemma 4 E2B Q4_K_M sustains 7.4-7.6 tok/s for 10 minutes in an unrooted app
    process (PocketPal AI, screen on), stable with no measurable degradation. Below
    the Termux baseline of 11-12 but stable, and the shortfall is attributed to the
    absence of a `--swa-full` equivalent: PocketPal exposes no such flag, so Gemma's
    sliding-window attention invalidates the prompt cache every turn (#6). The
    shipped app drives llama.cpp directly and regains the flag, so 7.6 is treated as
    a floor for the app, not a ceiling — predicted, not measured. Runner defaults
    were actively hostile — 6 CPU threads (violates #4) and a 131072 context — and
    cost roughly 40% of throughput before correction; the app must set threads,
    context and cache flags in code, never inherit defaults. zone9 peaked at 82 °C,
    median 66 °C; that reading is routed to the Local AI thread as corroborating
    evidence only and does not fill STATUS.md's outstanding loaded measurement.
69. **The app requires a foreground service, not merely a wake lock — a liveness
    requirement, not an optimisation** — observed during A2: PocketPal's generation
    stopped the moment the app lost focus, while Termux's concurrent logger survived,
    the difference being the foreground service Termux holds. Without one, the
    reasoning layer dies on any notification, call or app switch mid-mission. Not a
    safety defect: rule 0.2 keeps drive, obstacle stop and emergency stop below the
    reasoning layer, and the 1000 ms firmware watchdog (#47) sits below the app
    entirely. Likely manifest type is `connectedDevice`, since the app is genuinely
    driving attached hardware. Screen-off throughput under a partial wake lock is NOT
    measured — A2's 7.4-7.6 was screen-on, and equal performance must not be assumed.

## Inference (cont.)

70. **#10's OpenCL rejection is scoped to Termux userspace, not Android GPU access
    generally** — the app runtime decision reopens GPU/NPU via vendor-supported
    paths. #10 stands for the prototype; it does not bind the app. Scoping only: no
    path is chosen and nothing is measured, so this authorises the investigation
    rather than the use.

## Model & inference (cont.)

71. **Gemma 4 QAT confirmed real and official, but the "~1 GB E2B" footprint claim
    is unsupported for a llama.cpp deployment** — verified against Google's model
    card and ai.google.dev/gemma/docs/core: official Gemma 4 QAT collection exists
    (Apache-2.0, E2B included), QAT holds near-bf16 quality at 4-bit. BUT: the QAT
    Q4_0 GGUF is the same 4-bit width as the deployed q4_k_m, so it lands near the
    current 3.3 GB, not 1 GB (E4B QAT Q4_0 GGUF measures 5.15 GB on HF for
    reference). The ~1 GB figure can only refer to the "mobile-optimized" wNa8o8
    format (2-bit decode layers), and NO evidence was found that llama.cpp can load
    wNa8o8 — it is described as a custom schema for vLLM-class runtimes, not GGUF.
    Conclusion: the QAT-for-footprint premise is blocked on inference-engine format
    support and must be confirmed before any download. QAT-at-equal-size
    (quality/robustness at ~3.3 GB) remains a legitimate thing to benchmark. Do not
    swap the default on footprint grounds.
72. **MTP speculative decoding requires a QAT drafter at the same precision as the
    target** — per Google's QAT model card, the drafter must itself be a QAT
    checkpoint matching the target's precision. MTP cannot pair with the current
    q4_k_m model — the whole chain must be QAT. This supersedes the reasoning (not
    the conclusion) behind #4/#10's bandwidth argument only insofar as MTP shares
    the target KV cache rather than running a separate draft model; whether it
    helps a bandwidth-bound phone is still unmeasured. Strictly downstream of a QAT
    model being on the device.
73. **The documented llama.cpp rollback path in COMMANDS.md §9 is broken** —
    `llama-server.swafix` is a 5.9 KB fragment, not a runnable binary. Root cause:
    the build is shared-library, so the on-disk `llama-server` is a ~5.9 KB launcher
    that links against `libllama-server-impl.so` (7.9 MB), `libllama.so.0.0.1609`
    (3.4 MB), `libllama-common.so.0.0.1609` (6.2 MB); copying only the launcher
    captures nothing. `llama-server.working` (12 MB) is a valid earlier
    statically-linked binary and is a genuine standalone fallback, but it PREDATES
    the `--swa-full` fix (#6) — rolling back to it silently loses SWA. Correct
    rollback for a shared-lib build is the whole `build/` tree. Established this
    session: `build-b1609.tar.gz` (49 MB, verified: all four critical files
    present, gzip integrity OK) archives commit e1a1abb7 / version 1609. Recovery:
    `cd ~/llama.cpp && rm -rf build && tar xzf ~/build-b1609.tar.gz`. Second path:
    rebuild from git commit e1a1abb7.

## Thermal (cont.)

74. **Loaded BIG-core temperature measured for the first time: 97-101 °C sustained
    under llama.cpp inference, against 31-38 °C idle** — zone9 confirmed as `BIG`
    by reading `/sys/class/thermal/thermal_zone9/type`, so #29 stands. The
    kernel's own trip points for that zone, as read: active 20/55/80 °C, PASSIVE
    100 °C, active 104/106/110 °C, hot 120 °C. The device therefore runs in
    thermal equilibrium exactly at its designed passive throttle point — 11.5
    tok/s is what the envelope permits, not what the silicon could do cold. 20 °C
    of margin remains above that. Die temperature is not skin temperature: during
    the same run `disp_therm` read 29.4 °C and `usb_pwr_therm2` 29.6 °C, which is
    why the phone feels cool at 100 °C junction. Silicon has almost no thermal
    mass, so it swings ~60 °C within a single inference burst and falls back
    during rest — idle and loaded readings are both correct and not in conflict.
    Supersedes STATUS's unsourced 79-82 °C figure.
75. **The "two thermal thresholds in circulation" problem is worse than
    recorded: `thermal_guard.py` (warn 82 / critical 86) and `get_temp.py` are
    BOTH orphans** — nothing imports either, and `main.py` uses its own inline
    `get_temp`. So 82/86 are not competing live values, they are dead code. Only
    `run_mission`'s inline >80 °C pause runs. Against a measured 97-101 °C
    operating band that pause fires on essentially every check, so it is not
    "likely too low" (STATUS's wording) but far below the normal range. The
    threshold lives in exactly one place, which makes the fix simple; the correct
    value is NOT yet determined and must come from a real mission run, not this
    bench measurement.

## Inference (cont. II)

76. **#30's rest-improves-throughput finding reproduces in direction but NOT in
    magnitude on build 1609** — measured, same harness, 15 cycles: 2 s rest →
    11.5 tok/s median; 0 s rest → 11.1 tok/s median, a 3.5% gap. #30 recorded 8.7
    vs 11.1, a 28% gap. The mechanism is real — within the no-rest run, cycles
    1-5 averaged 11.4 and cycles 11-15 averaged 10.8, while the rested run stayed
    flat — but the effect size is an order of magnitude smaller than recorded.
    Cause not established; candidates are a different llama.cpp build,
    ambient/charging state (the device was on mains), or a longer original run
    reaching a steady state our 15 cycles did not. Treat "2 s rest" as a
    reasonable default, not a tuned constant.
77. **llama.cpp current upstream builds clean on-device and #6 still holds; NOT
    deployed** — built commit 4d917609 (version 0.4.0-dev, build 2233), 624
    commits ahead of e1a1abb7, from clean configure in 6m50s with
    `-DGGML_OPENCL=OFF` per #10. `--swa-full` is still accepted and still reuses
    the prompt cache: an identical repeat request processed 1 prompt token
    instead of 45, on both builds. Cold prompt eval improved (2435 ms → 1035 ms
    for 45 tokens). Generation tok/s was NOT compared honestly — the two
    single-sample readings (14.3 vs 12.2) were taken at different thermal states
    and are not a benchmark. Version reporting changed format:
    `0.4.0-dev (build 2233, commit …)` where b1609 reported only
    `version: 1609`. The robot default stays b1609: the upgrade is proven safe to
    build and proven to preserve #6, not proven better under the real workload
    (long prompts, vision payloads, the nav loop), and nothing currently requires
    it.

## Vision (cont.)

78. **`FOCAL_PX` cannot be calibrated from the `bench_photos` set, and at
    `FOCAL_PX`=500 the `PERSON_STOP_DIST` branch is unreachable** — measured
    across 12 person photos: `estimate_distance_single` overshoots by +55% to
    +1000%, every error in the same direction, minimum estimate 164 cm against
    an 80 cm threshold — so the person-stop branch cannot fire. A person at an
    estimated 45 cm reads as 497 cm. In practice people are handled by the 25 cm
    ultrasonic obstacle branch as generic obstacles, so the robot does not hit
    them, but person-aware behaviour does not exist. Calibration was not
    possible from this data: distances were visual estimates (±30-50 cm, stated
    in `labels.csv`, NOT tape-measured as intended), the subject was reclining
    so bbox height does not track the 170 cm standing assumption in
    `REAL_HEIGHTS`, and focal correlates with distance (r = +0.74), indicating
    the pinhole model is violated rather than the data being noisy. Derived
    median 192 fits no better (−40% to +324%). `FOCAL_PX` stays 500 and stays
    UNCALIBRATED. Do not substitute 192.
79. **Person-stop moves from computed centimetres to the coarse area bucket** —
    `PERSON_STOP_DIST` in cm depends on `FOCAL_PX`, which #78 shows cannot be
    calibrated, and on a standing-and-fully-visible pose the robot will rarely
    see indoors. The area-ratio bucket (field `r[3]`, DECISIONS #14) is already
    computed every cycle, needs no calibration, and degrades sensibly on partial
    or seated subjects. Chosen over recalibrating (fixes only the standing case)
    and over widening the ultrasonic rule (loses person-awareness entirely). NOT
    YET IMPLEMENTED — `navigate_rules` still calls `estimate_distance_single`.
    The bucket threshold ('very close' vs 'close') is not yet chosen and must be
    picked against real photos.
80. **`detect_scene()` resizes to 640×640 WITHOUT preserving aspect ratio** — a
    4080×3072 source scales 0.157 horizontally and 0.208 vertically, so bbox
    height is inflated relative to a true-aspect resize, and portrait sources
    distort the other way — the same subject at the same distance yields a ~33%
    different pixel height depending on phone orientation. Since
    `navigate_rules` feeds that height (`r[7]`) to `estimate_distance_single`,
    this is a second and independent reason the cm-based person distance is
    unreliable, compounding #18 and #78. Also recorded: the `detect_scene`
    docstring (`detect_person.py:50`) documents a 4-field tuple; the function
    returns 8 fields.

## Navigation (cont. IV)

81. **Gemma can still override a safety move at the one moment it matters
    most — OPEN DEFECT, not fixed** — `run_cycle`'s guard reads
    `use_gemma = (… or stuck) and not (safety_move and not stuck)`. At ladder
    step n==7 `nav_stuck` is set, so `safety_move` and `stuck` are both true and
    the guard admits the Gemma call; the parse loop then overwrites `move` with
    the model's output, including FORWARD, with no re-check afterwards. The
    exposed window is exactly: distance under `OBSTACLE_DIST`, robot boxed in,
    both sweep directions exhausted. This is the case #53 was written to
    prevent; the step-73 patch closed the wide hole and left a narrow one at
    maximum danger. Found by static review, NOT by `nav_test.py`, which
    exercises `navigate_rules` in isolation and never reaches `run_cycle`. Fix
    must re-check the move after the Gemma block, and must be verified against
    `nav_test.py` plus a `run_cycle`-level test that does not yet exist.

## Tooling

82. **Claude Code runs on-device via proot-distro Debian, not AVF** — AVF was
    rejected on two grounds independent of availability: it emulates no USB host
    controller, and it shares only `/mnt/shared`, so it can reach neither the
    ESP32 nor `~/robot` — the two things wanted.
    (`getprop ro.virtualization.supported` also returned empty on this device.)
    proot-distro Debian + the official installer gives Claude Code 2.1.261
    inside Termux's own namespace; `proot-distro login debian --bind
    /data/data/com.termux/files/home:/termux-home` exposes the real working
    tree. Confirmed limits: no `su`, no `/dev/bus/usb`, so `motors.py`,
    `teleop.py`, `run_mission.py`, `log_run.py`, `dist_raw.py` and
    `cp2102_test.py` remain manual. Confirmed value: it reads whole files at
    once, which is the reviewing thread's blind spot — it produced the
    dependency map in `FILES.md` and found #81 in a single pass. Its limit is
    that it reasons from the files it is given: it reported
    `Robot.run_mission()` as dead because nothing in `main.py` calls it, while
    `run_mission.py` does. Scope cross-file questions explicitly.

## Navigation (cont. V)

83. **Mecanum mixing uses a flipped vx sign relative to the textbook mix —
    verification predates the current wiring** — both `mode1_simple.ino` and
    `mode2_auto.ino` compute `fl = vy-vx+w`, `fr = vy+vx-w`, `rl = vy+vx+w`,
    `rr = vy-vx-w`, with vx's sign reversed from the standard formula, because
    this chassis's rollers sit in a mirrored X pattern rather than the textbook
    arrangement. Verified live on `mode1_remote.ino` (recovered artifact,
    7956 bytes, last modified after a camera-panel edit that postdates the
    verification but did not touch `drive()`): forward and rotate were already
    correct, strafe came out reversed, and only the vx signs were changed.
    `mode1_remote.ino` was forked into `mode1_simple.ino`; its own corner map
    (`FL=3, FR=1, RL=2, RR=0`, `INVERT` all-false) is the pre-#43 mapping and
    is NOT safe to flash as a reference build — it is a provenance record only.
    `mode2_auto.ino` carries the same equation by copy, not by independent
    verification, and has never been run with motors. Re-verification of
    strafe and diagonals under the #43 mapping is OUTSTANDING. Not a suspected
    error — a rewrite of `drive()` that "corrects" the sign to the textbook
    formula will strafe backward — but the flip's continued correctness under
    the current wiring is unconfirmed.
84. **Orphaned prototype files resolved — some removed, some kept deliberately** —
    `detect_scene.py`, `llm.py`, `voice.py`, `ch340_test.py`, `thermal_benchmark.py`,
    `thermal_benchmark2.py`, `quality_benchmark.py` removed from the working tree:
    each is superseded by a working replacement (`detect_person.py`,
    `main.parse_command`, `main.listen()`, `thermal_benchmark3.py`,
    `quality_benchmark2.py`), and `ch340_test.py` is actively misleading — it
    targets the wrong chip and could send someone debugging serial to a false
    "board is broken" conclusion. `nav_sim.py` is kept per #23. `get_temp.py` and
    `thermal_guard.py` are kept because the thermal threshold is unresolved (#75)
    and these are the only per-zone readers available if it gets wired in.
    `mission_test.py` is kept because it is the only harness driving `Robot` via a
    `SimulatedMotors` stub — the shape #81's outstanding `run_cycle`-level test
    needs. `git rm` only; history is not rewritten, so removal is reversible.
