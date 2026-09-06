# STATUS

Snapshot of the running system. Values below are read from the code as it exists
today — not from benchmark recommendations. Pending deltas are listed explicitly.

## Architecture

**Brain:** Pixel 7 (rooted, Termux, Android 17). **Body:** ESP32 NodeMCU + 2× MX1508
driving 4 mecanum motors. Phone ↔ ESP32 over USB serial.

**Currently flashed:** `mode2_auto.ino`, the corrected build (DECISIONS #43 corner
map, `HAS_ULTRASONIC 1`, diagonals, 115200, `WATCHDOG_MS` 1000). `mode1_simple.ino`
is the standalone softAP teleop sketch and is not on the board — the two firmwares
are mutually exclusive on one ESP32.

**Entry point:** `run_mission.py` runs autonomous. `main.py.__main__` only runs a
single test cycle against a photo with no motors attached; it does not run missions.

**Development tools** (not part of the robot path): `nav_test.py` (nav rule harness,
no hardware), `log_run.py` (ultrasonic motion logger, needs root), `identify_it1.py`
(IT1 fit, numpy only), `build_bench.py` (A/B latency bench vs whatever server is on
8080), `capture_bench.py` (guided photo capture via the same `termux-camera-photo`
path `main.py` uses), `focal_calibrate.py` (FOCAL_PX derivation), `bench_photos/` +
`labels.csv` (38 labelled captures: 15 room-signature, 14 person, 5 empty, 4
blocked), `FILES.md` (dependency map), `BENCHMARK_PLAN.md` (replacement-benchmark
design, not implemented).

`~/llama.cpp-upstream` is a disposable work area holding build 2233 (commit
4d917609), built and SWA-verified this session but NOT deployed. `~/llama.cpp`
(b1609) remains the running server. Safe to delete `~/llama.cpp-upstream` and
`~/upstream-server.log` at any time. `~/robot` is a git repo; working state
committed as fe14be2.

**Public repo:** github.com/abdelhamidzoghlamii-ui/Pixel-robot — synced at
af66fab (2026-09-05), carries the live code. `.gitignore` excludes model
weights (`.onnx`/`.gguf`/`.bin`), `bench_photos/`, and run artifacts; those
stay on the phone only. `test_photos/` remains tracked from earlier commits.
STATUS.md is still canonical for config values. `RECOVERY.md` and the repo's
`CLAUDE.md` are unchanged since the April session and remain stale — the code
sync did not touch them.

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
move == STOP · a new room was just mapped. Vision (image attached) is sent only on
the `every_5` / person / goal / new_room subset.

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
```

Current llama.cpp build: commit e1a1abb7, version 1609 (Clang 21.1.8, Android
aarch64). Shared-library build — `bin/llama-server` is a ~5.9 KB launcher, real
code in the `.so` files. Rollback is the whole `build/` tree (`build-b1609.tar.gz`,
verified) or a rebuild of commit e1a1abb7, NOT a single-binary copy (DECISIONS #73).

Benchmarked on build 1609, 15 cycles, realistic robot prompt shape (fixed system
prefix + varying scene): 11.5 tok/s median (11.1-11.9), prompt eval 1091 ms median,
free RAM ~3.1 GB, zone9 97-101 °C. Prefix cache reuse confirmed on the live prompt
shape — 109 tokens on the first call, 18-22 thereafter.

## Working now

- LLM server stable; prompt-cache fix landed (`--swa-full`) — repeat-prefix calls
  reuse the cached system prompt instead of reprocessing it.
- YOLO scene detection, position + coarse distance.
- Voice pipeline end to end: record → Whisper → Gemma JSON parse → TTS.
- Root restored on Android 17; real SoC temps readable (zone9 BIG / 10 MID /
  11 LITTLE / 12 GPU / 14 TPU / 22 battery).
- **Mode 1 teleop — ESP32 standalone** (`mode1_simple.ino`, project files): softAP "MecanumBot"
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
- Benchmarks: voice parsing 93% (schema caveat below). Nav logic unbenchmarked —
  see Pending.

## Pending / untested

- **Dual-rate nav merged as a rewrite. Safety branches bench-tested, everything
  else untested.** `navigate_rules` escalation ladder verified against synthetic
  distances via `nav_test.py`: thresholds exact at the `<15` / `<25` / `>=25`
  boundaries, no oscillation, `nav_stuck` fires on exactly one cycle per episode,
  60s timeout reached. NOT tested: the person branches
  (`estimate_distance_single`), and the whole loop with motors live —
  `run_mission.py` has only ever been run with `--dry`. The 93% figure still does
  not transfer (#55). An open safety defect exists in the untested path — see the
  Gemma-override bullet below (DECISIONS #81) — and should be resolved before the
  motors-live run.
- **Rotation is uncalibrated.** Nobody knows how many degrees one `LEFT`/`RIGHT` at
  speed 120 for `CYCLE_MOVE_TIME` produces, so the ladder's "4 steps one way, then
  sweep past centre" is a guess about coverage, not a measured 90°/180°. The
  ultrasonic cannot measure angle — needs a protractor or overhead video.
- **Forward motion uncalibrated.** `MOTOR_SPEED` 130 → cm/s unknown; tooling written
  (`log_run.py`, `identify_it1.py`) and validated on synthetic data, but no run
  against a wall has been taken. K_I will also drift as the pack sags from 8.2V
  toward cutoff — unmeasured.
- **The `mode2_auto.ino` copy in project files is the pre-correction build** —
  `BAUD` 9600, `INVERT` all false, `FL`/`RL` on the superseded #40 mapping,
  `WATCHDOG_MS` 5000, `HAS_ULTRASONIC` 0, no diagonals. It is NOT what is flashed;
  at 9600 the 115200 link would not round-trip and `DIST:` would not flow. Do not
  reflash from it — the corrected build lives on the laptop and has not been
  uploaded.
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
- **OPEN SAFETY DEFECT: Gemma can override a safety move when `nav_stuck` is
  set.** At ladder step n==7 the `run_cycle` guard admits the Gemma call and the
  result overwrites the safety move with no re-check — including FORWARD while
  blocked under 25 cm. Not fixed. See DECISIONS #81.
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
  an inline `get_temp`. Correct value still undetermined; take it from a real
  mission run. See DECISIONS #74, #75.
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
- **QAT/MTP evaluation gated, not started.** QAT Q4_0 E2B benchmark
  (footprint/speed/thermal, plus accuracy) requires a valid nav benchmark first —
  the 93% figure does not describe live code (#55), so the accuracy gate is
  currently undefined. wNa8o8 mobile format (the only path to the ~1 GB claim)
  needs llama.cpp load-support confirmed before download. MTP needs a full QAT
  chain incl. a matching QAT drafter. See DECISIONS #71, #72.
- **`nav_sim.py` is a standalone older simulation** (Qwen 2.5 3B, own thermal
  limits WARN 75 / KILL 88 / COOL 48 °C). Not part of the live robot path.
- Stale comment: `server_manager.py` still says `--cache-ram 0` disables the broken
  SWA cache; the code actually uses `--swa-full`.
- Not yet built/wired: Piper TTS, Whisper VAD, room classifier, memory system,
  face recognition.

## Last hardware test

**Power-up bring-up (bench, not yet driving).**
Battery pack 2× 18650 in series, cells balanced at 4.1 V / 4.1 V, pack 8.2 V.
BMS (HW-391 2S 20A) output steady 8.2 V. Buck converter set to 5.03 V no-load,
holding 5 V under ESP32 load. ESP32 powers up, 3V3 rail reads exactly 3.3 V on USB.
On buck power the 3V3 pin misreads ~3.76 V and the GPIO2 blue LED lights — both
traced to a marginal ground return on the buck path; fix is a solid star ground at
final assembly.

BMS output leg confirmed — P+/P− reads stable 8.17 V (⊕ = P+). Rail-to-star short
check clean (~1.35 kΩ, caps charging). Power distribution verified end to end.

**First motor spin achieved.** All four motors run continuously under PWM (1kHz,
8-bit, speed 130) with the ESP32 on USB/phone power and motors on the battery rail.
Pin map confirmed working (P16/17/18/19 → MX1508 #1 right side, P21/22/23/25 → MX1508 #2
left side — see DECISIONS #43 for corner mapping). Motors NOT yet verified with the ESP32 on buck power — no decoupling caps
fitted, and PWM rail collapse brownout-resets a buck-powered ESP32 (DECISIONS #38).
Remaining: fit caps, re-verify on buck power, then confirm direction sense per corner.

HC-SR04 verified on P27/P26 with 1kΩ/2kΩ divider on ECHO (DECISIONS #42).

Buck #2 (3.3 V sensor rail) remains unwired, per the hardware thread's as-built
diagram. Nothing currently depends on it — HC-SR04 draws from the ESP32 5V pin
(DECISIONS #42), not buck #2.

Buck #2 (3.3 V sensor rail) remains unwired, per the hardware thread's as-built
diagram. Nothing currently depends on it — HC-SR04 draws from the ESP32 5V pin
(DECISIONS #42), not buck #2.
