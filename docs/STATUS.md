# STATUS

Snapshot of the running system. Values below are read from the code as it exists
today — not from benchmark recommendations. Pending deltas are listed explicitly.

## Architecture

**Brain:** Pixel 7 (rooted, Termux, Android 17). **Body:** ESP32 NodeMCU + 2× MX1508
driving 4 mecanum motors. Phone ↔ ESP32 over USB serial.

**Currently flashed:** UNCONFIRMED on hardware. A `mode2_auto.ino` matching
the corrected spec (DECISIONS #89 — BAUD 115200, INVERT indices 2,3,
WATCHDOG_MS 1000, HAS_ULTRASONIC 1) is committed at
firmware/mode2_auto/mode2_auto.ino (commit 9a90d7e), verified field-by-field
against the file itself. Whether this exact file is what's running on the
board is not yet confirmed.

**Entry point:** `run_mission.py` runs autonomous. `main.py.__main__` only runs a
single test cycle against a photo with no motors attached; it does not run missions.

**Development tools** (not part of the robot path): `nav_test.py` (nav rule harness,
no hardware), `log_run.py` (ultrasonic motion logger, needs root), `identify_it1.py`
(IT1 fit, numpy only), `build_bench.py` (A/B latency bench vs whatever server is on
8080), `capture_bench.py` (guided photo capture via the same `termux-camera-photo`
path `main.py` uses), `focal_calibrate.py` (FOCAL_PX derivation), `bench_photos/` +
`labels.csv` (38 labelled captures: 15 room-signature, 14 person, 5 empty, 4
blocked), `FILES.md` (dependency map), `BENCHMARK_PLAN.md` (replacement-benchmark
design, not implemented), ~/json_mechanism_test.py (bounded read-only diagnostic,
13 synthetic text-only scenes, five output mechanisms, per-model profile — gemma
port 8080 / qwen port 8081, correct chat wrapper and stop tokens per family;
writes ~/json_mechanism_results_<model>.json; no camera, no motors, no repo
writes).

`~/llama.cpp-upstream` built and SWA-verified b2233 (commit 4d917609) in
DECISIONS #77, then was rebuilt in place to b2351-790cf51a. The b2233 binary
is gone, but source commit 4d917609 remains reachable for a rebuild
(DECISIONS #107). `~/llama.cpp` (b1609) is the documented deployed server.
The 1609-vs-2233 comparison remains open, not cancelled (DECISIONS #77, #107).
`~/robot` is a Git repository. The earlier `fe14be2` snapshot is historical;
it does not identify the current checkout.

**Public repo:** github.com/abdelhamidzoghlamii-ui/Pixel-robot.
Canonical documentation and role instructions are in `/docs` relative to the
repository root. Root `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` are session
discovery pointers. `STATUS.md` remains canonical for prototype config values.
`RECOVERY.md` was removed; [HANDOFF.md](HANDOFF.md) is the entry point to recovery
information, with procedures in [COMMANDS.md](COMMANDS.md#9-housekeeping-and-recovery)
and historical provenance in [OPERATIONS_HISTORY.md](OPERATIONS_HISTORY.md).

Read `git rev-parse HEAD` and `git status --short --branch` from the repository
to determine the current revision and local state; do not maintain a "current
commit" literal here. The `af66fab` sync was historical (2026-09-05).
Remote-tracking refs reflect the last locally known remote state.

`.gitignore` excludes model weights (`.onnx`/`.gguf`/`.bin`), `bench_photos/`,
and run artifacts; those stay on the phone only. `test_photos/` remains tracked
from earlier commits.

**Per-cycle flow** (`main.py: Robot.run_cycle`):
```
camera photo → YOLO detect_scene → navigate_rules (pure Python, instant)
             → Gemma consulted only when a trigger fires → execute move
```

| Layer | Handles | Cost |
|---|---|---|
| YOLO (yolo11m.onnx) | object/person detection, position, coarse distance | every cycle |
| Python `navigate_rules` | obstacle stop, person approach, room signature logging | every cycle, 0ms |
| Gemma 4 E2B | strategy when triggered (see triggers below) | ~5s warm |
| Whisper.cpp (base) | 5s voice capture → text | on demand |
| Gemma `PARSE_SYS` | voice text → JSON action array | on demand |
| `termux-tts-speak` | speech output | on demand |

**Gemma trigger conditions** (`run_cycle`, any one fires a call):
`cycle % GEMMA_INTERVAL(10) == 0` · `cycle % 5 == 0` · person detected ·
move == STOP · a new room was just mapped · `nav_stuck` is set. At `nav_stuck`,
Gemma is consulted once, but `run_cycle()` restores the exact Python-selected
safety move after processing (DECISIONS #88). An image payload is sent only on
the `every_5` / person / goal / new_room subset; the deployed server was
observed to ignore it (DECISIONS #96).

As coded, `main.py:Robot.run_cycle()` still consults Gemma on its existing
triggers after YOLO and `navigate_rules()`; Python-selected safety moves are
restored. DECISIONS #104 records the intended removal of the conversational
LLM from low-level navigation decisions, and that code rework remains pending.
DECISIONS #109 opens a separate, unimplemented high-level mission-script
selector using Laya or Von. The current repo describes a mecanum prototype;
the phone-mounted RC car with 2D LiDAR is a proposed redesign.

## Known-good config (as coded)

**Navigation** (`main.py`)
```
OBSTACLE_DIST     25 cm     hard stop threshold
PERSON_STOP_DIST  80 cm     stop when person this close
GEMMA_INTERVAL    10        cycles between forced Gemma checks
STEREO_BASELINE   5.0 cm    strafe distance for depth   (UNVERIFIED)
MOTOR_SPEED       130       FORWARD/BACK
rotation speed    120       hardcoded literal in Robot.move(), not a constant
strafe speed      100       hardcoded literal in Robot.move(), not a constant
CYCLE_MOVE_TIME   1.5 s
thermal pause     >80 °C    reads thermal_zone9 (BIG cores), sleeps 5s
```

**Vision** (`detect_person.py`)
```
yolo11m.onnx · CONF 0.35 · IOU 0.45 · 640×640 input
position: cx<213 left, cx>427 right, else center
coarse distance by box area: >0.3 very close, >0.1 close, >0.03 medium, else far
```

`yolo11m.onnx` is fine-tuned, not stock — trained on the owner's own photos
(human-stated 2026-09-11; training set held on the owner's laptop, not in
this repository or independently verified from this session). See DECISIONS
#97.

**Stereo depth** (`stereo_depth.py`)
```
FOCAL_PX    500 px    UNCALIBRATED — placeholder, comment says "calibrate later"
FRAME_SIZE  640 px
disparity < 2px → rejected as unreliable, falls back to single-photo estimate
object match requires vertical delta < 80px
REAL_HEIGHTS: person 170, refrigerator 180, chair 90, couch 85, dining table 75,
              tv 60, bed 50, toilet 40, potted plant 40, bottle 25, vase 25,
              cup 10, laptop 3   (cm)
```

**LLM server** (`server_manager.py`)
```
--threads 4 --threads-batch 4 --parallel 1 --swa-full --ctx-size 2048
setup_q4     Gemma 4 E2B Q4_K_M  3.3GB  11-12 tok/s   ← robot default
setup_e4b    Gemma 4 E4B Q4_K_M  5.0GB  7.2 tok/s     ← quality/chat mode
setup_qwen3b / setup_qwen1b                            ← fallbacks
(Qwen3.5 0.8/2/4B GGUFs at ~/models/qwen35/ are eval-only, not wired into
server_manager; see DECISIONS #106. Removing the LLM from navigation is
decided in #104 but not implemented.)
```

Current llama.cpp build: commit e1a1abb7, version 1609 (Clang 21.1.8, Android
aarch64). Shared-library build — `bin/llama-server` is a ~5.9 KB launcher, real
code in the `.so` files. Rollback is the whole `build/` tree (`build-b1609.tar.gz`,
verified) or a rebuild of commit e1a1abb7, NOT a single-binary copy (DECISIONS #73).

A third build, ~/llama.cpp-upstream rebuilt 2026-09-13 to b2351-790cf51a,
loads Qwen3.5-0.8B-Q4_K_M with architecture accepted and "modalities: text"
— MEASURED. At that text-load test, no mmproj was downloaded or passed;
later vision tests in #105 found no usable on-device LLM vision. Note: `strings`
on the rebuilt binary returns no qwen35 match despite the successful load —
the `strings` check used in #95 is not a reliable negative; an actual load
attempt is ground truth.

Benchmarked on build 1609, 15 cycles, realistic robot prompt shape (fixed system
prefix + varying scene): 11.5 tok/s median (11.1-11.9), prompt eval 1091 ms median,
free RAM ~3.1 GB, zone9 97-101 °C. Prefix cache reuse confirmed on the live prompt
shape — 109 tokens on the first call, 18-22 thereafter.

## Thermal governance

VIRTUAL-SKIN-CPU-GPU is the reported HAL CPU-throttling signal (first trip
37.0 °C); BIG/zone9 is not its HAL governor (DECISIONS #90). The live
`main.py` pause still reads BIG above 80 °C; a correct sensor and threshold
have not been selected or verified in a real mission. The dated device
thresholds, formula, zone map, throttle checks, and evidence limitations are
preserved in [PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).

## Memory

Reported 2026-09-11: Gemma-4-E2B Q4_K_M left approximately 0.5 GB practical
headroom after loading, without simultaneous full perception or dialogue.
This is a dated measurement, not a current co-residency test; see
[PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md) and
DECISIONS #94.

## Model vision

The as-coded image request was confirmed to run as text only (DECISIONS #96).
Later b2351 projector tests did not produce usable on-device LLM vision
(DECISIONS #105). YOLO remains the perception path. The original live-call
observations and follow-up are retained in
[PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).

## Working now

- LLM server stable; prompt-cache fix landed (`--swa-full`) — repeat-prefix calls
  reuse the cached system prompt instead of reprocessing it.
- YOLO scene detection, position + coarse distance.
- Voice pipeline end to end: record → Whisper → Gemma JSON parse → TTS.
- Root restored on Android 17; real SoC temps readable (zone9 BIG / 10 MID /
  11 LITTLE / 12 GPU / 14 TPU). The battery zone number is unconfirmed.
- **Mode 1 teleop — ESP32 standalone** (`mode1_simple.ino`, tracked and retained; commit 9a90d7e, DECISIONS #89 —
  no removal planned): softAP "MecanumBot"
  → web UI at http://192.168.4.1. Mecanum mixing, live HC-SR04 readout, forward
  blocked under 25cm, 600ms no-command stop failsafe. Phone and AI not required —
  ESP32 standalone.
- **Mode 1 teleop — from the Pixel** (`teleop.py`): Pixel hotspot + web UI, driven
  from a second phone's browser. Diagonals, speed slider, live HC-SR04 readout,
  forward blocked under 25 cm. Requires root; collides with the LLM server on 8080.
- **Serial link live end to end**: Pixel → CP2102 → ESP32 → MX1508 → wheels, and
  HC-SR04 → `DIST:` → `motors.py`. First commanded motion from the phone achieved
  this session.
- **Distance sensor live.** `HAS_ULTRASONIC 1`; `DIST:` flowing and verified against
  hand movement. Root cause of the initial `DIST:0` readings was sensor VCC wired to
  GND instead of the ESP32 5V pin — the trigger pulse was coupling through an
  unpowered sensor. Both obstacle branches (`main.py` `<15` and `<OBSTACLE_DIST`)
  are now live.
- Benchmarks: voice parsing 93% (schema caveat below). The current integrated
  `main.py` navigation path remains unbenchmarked; see Pending and
  `BENCHMARK_PLAN.md`. Separately, the offline synthetic strategic-selector v2
  archive records Laya's development-selected `filtered_text` at 9/11
  acceptable held-out cases, median 3028.8 ms and 4/11 option-order flips.
  Von's development-selected `two_stage_text` scored 8/11, median 4797.9 ms
  and 9/11 flips. A post hoc Von `filtered_text` follow-up also scored 8/11,
  median 2053.8 ms and 9/11 flips; only its stdout was saved. The models saw
  disjoint generated variants, so these are not controlled head-to-head
  accuracy results. Raw original-run JSON, stdout, manifests, the executed v2
  source, v1 history and explicit evidence gaps are archived under
  `benchmark/strategic_selector/`.
- Strategic-selector evidence: a 2026-09-23 phone audit confirmed four archived
  runs and benchmark sources published on origin/main; publication does not
  establish model accuracy or robot hardware validity (see
  [RUN_INDEX.md](../benchmark/strategic_selector/RUN_INDEX.md)).

## Pending / untested

- **Dual-rate nav merged as a rewrite. Rule-level safety branches and the #81
  `run_cycle()` path are regression-tested; everything else is untested.**
  `navigate_rules` escalation ladder verified against synthetic
  distances via `nav_test.py`: thresholds exact at the `<15` / `<25` / `>=25`
  boundaries, no oscillation, `nav_stuck` fires on exactly one cycle per episode,
  60s timeout reached. NOT tested: the person branches
  (`estimate_distance_single`), and the whole loop with motors live —
  `run_mission.py` has only ever been run with `--dry`. The 93% figure still does
  not transfer (#55). The #81 Gemma-override defect is fixed in code and covered
  by `run_cycle_safety_test.py` in both avoidance directions with fake external
  operations; no hardware validation was performed (DECISIONS #88).
- **Rotation is uncalibrated.** Nobody knows how many degrees one `LEFT`/`RIGHT` at
  speed 120 for `CYCLE_MOVE_TIME` produces, so the ladder's "4 steps one way, then
  sweep past centre" is a guess about coverage, not a measured 90°/180°. The
  ultrasonic cannot measure angle — needs a protractor or overhead video.
- **Forward motion uncalibrated.** `MOTOR_SPEED` 130 → cm/s unknown; tooling written
  (`log_run.py`, `identify_it1.py`) and validated on synthetic data, but no run
  against a wall has been taken. K_I will also drift as the pack sags from 8.2V
  toward cutoff — unmeasured.
- **Corrected `mode2_auto.ino` now exists and is version-controlled** —
  firmware/mode2_auto/mode2_auto.ino, commit 9a90d7e, matches #42/#43/#45/#47
  exactly (verified). Not yet confirmed against the physical board — that
  requires a direct hardware check, not a file read.
- **`BACK_R` diagonal drove one wheel instead of two.** Observed before the
  corner-map correction (#49); not re-tested since.
- **Mecanum vx-sign flip unverified under current wiring.** Last confirmed live
  on a recovered pre-#43 artifact (DECISIONS #83); strafe and diagonals have not
  been re-tested since the corner-map correction and a left-motor lead swap.
  This directly underpins the escalation ladder's opening strafes (#61), which
  assume strafe holds heading — if the sign convention no longer matches the
  physical wiring, "strafe" could produce rotation or drift instead of lateral
  movement. Re-verify forward/strafe/rotate/diagonals under the corrected map
  before trusting the ladder on hardware.
- **Gemma cannot replace a Python safety move during `nav_stuck`.** `run_cycle()`
  saves the exact result from `navigate_rules()` and restores it after Gemma
  processing whenever the existing `safety_move` condition was true. Gemma still
  receives the one `nav_stuck` consultation. `run_cycle_safety_test.py` covers
  both avoidance directions and verifies that `FORWARD` is rejected, the original
  turn executes, and the following blocked cycle does not consult Gemma again.
  This is regression coverage with fake external operations, not hardware
  validation. See DECISIONS #81 and #88.
- **Dead and orphaned code mapped** (`FILES.md`). Superseded: `detect_scene.py`,
  `llm.py`, `voice.py`, `ch340_test.py`, `nav_sim.py`, `thermal_benchmark.py`,
  `thermal_benchmark2.py`, `quality_benchmark.py`. Orphans never wired in:
  `get_temp.py`, `thermal_guard.py`, `server_manager.py` (nothing starts the
  server programmatically). `mission_test.py` predates the current `Robot` API
  and is probably broken.
- **Orphaned-file deletion deferred (DECISIONS #85).** `detect_scene.py`,
  `llm.py`, `voice.py`, `ch340_test.py`, `thermal_benchmark.py`,
  `thermal_benchmark2.py`, and `quality_benchmark.py` remain tracked and present.
  DECISIONS #84's stated `git rm` did not occur in the checked repository; retain
  all seven until an explicit future removal task. `nav_sim.py` (#23),
  `get_temp.py`/`thermal_guard.py` (#75), and `mission_test.py` remain deliberately
  kept. 420 MB of unused ONNX models remains untouched.
- **`get_temp()` pause threshold >80 °C is far below the operating band.**
  Measured 97-101 °C sustained under inference, 31-38 °C idle; the kernel's own
  passive trip for zone9 is 100 °C, so the chip runs in equilibrium at its designed
  throttle point. The >80 °C pause would fire on nearly every check.
  `thermal_guard.py`'s 82/86 are dead code — nothing imports it and `main.py` uses
  an inline `get_temp`. A replacement needs the correct thermal signal and a
  threshold validated against a real mission trace; neither is selected.
  See DECISIONS #74, #75, #90.
- **Battery thermal zone number unconfirmed.** STATUS previously implied zone
  22; a later enumeration reported zone 25. Neither is canonical until a
  direct thermal_zoneN/type read confirms it.
- **Two JSON schemas in circulation.** `main.py: PARSE_SYS` emits
  `{"type": ..., "name": ..., "room": ...}`; the benchmark scripts used
  `{"action": ..., "target": ...}`. Not reconciled.
- **Prompt vs code disagree on person distance.** `GEMMA_SYS` says stop under
  100cm; `PERSON_STOP_DIST` is 80cm.
- **FOCAL_PX uncalibrated; PERSON_STOP_DIST unreachable as coded.** Overshoot
  +55% to +1000% across 12 measured photos, minimum estimate 164 cm against an
  80 cm threshold, so the person-stop branch never fires. People are handled by
  the 25 cm ultrasonic branch as generic obstacles. Person-stop is to move to the
  area bucket (#79) — decided, not implemented. See DECISIONS #78, #79, #80.
- **QAT/MTP evaluation gated, not started.** The earlier QAT Q4_0 E2B proposal
  required a valid nav benchmark for an LLM model swap. After #104, that is a
  historical model-in-loop gate, not a gate for the intended Python navigation
  path. Any future voice or selector model swap needs its own evaluation criteria.
  wNa8o8 mobile format (the only path to the ~1 GB claim)
  needs llama.cpp load-support confirmed before download. MTP needs a full QAT
  chain incl. a matching QAT drafter. See DECISIONS #71, #72.
- **`nav_sim.py` is a standalone older simulation** (Qwen 2.5 3B, own thermal
  limits WARN 75 / KILL 88 / COOL 48 °C). Not part of the live robot path.
- Stale comment: `server_manager.py` still says `--cache-ram 0` disables the broken
  SWA cache; the code actually uses `--swa-full`.
- Not yet built/wired: Piper TTS, Whisper VAD, room classifier, memory system,
  face recognition.
- The proposed strategic selector is not wired to the robot. Apartment
  localization from 2D LiDAR, semantic room map, camera–LiDAR object/range
  association, named-person verification, eligible-script gates, safe
  interruption, and model/sensor co-residency need separate validation.
  YOLO `person` detection does not establish Chiara's identity; Laya/Von
  confidence is not a braking guarantee. Script duration and whether an
  additional car-mounted accelerometer helps remain undetermined. The
  current ultrasonic sensor and phone sensors have not been integrated
  into this proposed selector.

## Work priorities

Preserved from the 2026-09-05 handoff; these are pending tasks, not new test results
or permission to execute hardware work.

1. Keep `nav_test.py` and the #81 `run_cycle()` regression passing before the
   first motors-live autonomous run. The code defect is fixed, but no hardware
   validation has been performed (DECISIONS #88).
2. Implement the person-stop area-bucket decision (#79). Choose 'very close' versus
   'close' against `bench_photos/`; the choice remains pending.
3. Build the replacement navigation benchmark described in
   [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md): real photos → real `detect_scene()` →
   real `navigate_rules()`, with distance injected. The proposed scores are token
   validity and sub-25 cm forward violations; they do not establish navigation
   accuracy or cover the `run_cycle` override. See #104 and the historical
   QAT/MTP model-swap gate above.
4. Take forward-motion calibration with the robot; use
   [COMMANDS.md §6](COMMANDS.md#6-nav-rules--calibration).
5. Take rotation calibration with the robot; method remains pending, as recorded
   above and in COMMANDS.md §6.
6. After the safety fix, perform the first motors-live autonomous run under human
   control, with wheels on a stand. Capture the mission thermal trace to validate
   the sensor and a proposed pause threshold (#75, #90).

Mode 2 video/audio teleop remains parked indefinitely for thermal cost (#58).
This does not park the serial-command firmware named `mode2_auto.ino`.

## Last hardware test

Bench power-up, first four-motor PWM spin, and HC-SR04 readings were reported.
The ESP32 driving motors from buck power remains unverified after an earlier
brownout; capacitors, solid ground return, and current-wiring direction tests
remain pending. Full measured values and as-built limitations are preserved in
[PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).
