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
RobotCam frame (640×480, 2/s, fail-closed reader) → YOLO yolo11s (320, 640 every 5 s or in person search) → navigate_rules (pure Python, instant) → execute move
```

| Layer | Handles | Cost |
|---|---|---|
| YOLO yolo11s (320/640 policy, both sessions resident; yolo11m kept for rollback) | object/person detection, position, coarse distance | every cycle |
| RobotCam app (Camera2 foreground service) | 640×480 frames, 2/s; missing/old/repeated frame → STOP | continuous while a mission runs |
| Python `navigate_rules` | obstacle stop, person approach, room signature logging | every cycle, 0ms |
| Gemma 4 E2B (Q4_0, default since 2026-09-26) | voice→intent only, outside navigation | on demand |
| Whisper.cpp (base) | 5s voice capture → text | on demand |
| Gemma `PARSE_SYS` | voice text → JSON action array | on demand |
| `termux-tts-speak` | speech output | on demand |

DECISIONS #109 opens a separate, unimplemented high-level mission-script
selector. It is still not wired to the robot. After #112–#116 the lead
candidate is Gemma scoring the offered options by letter probability; Laya and
Von continue only on the fine-tuning track (#113). A single resident Gemma for
both conversation and selection is under evaluation, not decided (#116).
The current repo describes a mecanum prototype; the phone-mounted RC car with
2D LiDAR is a proposed redesign.

## Known-good config (as coded)

**Navigation** (`main.py`)
```
OBSTACLE_DIST     25 cm     hard stop threshold
PERSON_STOP_DIST  80 cm     stop when person this close
STEREO_BASELINE   5.0 cm    strafe distance for depth   (UNVERIFIED)
MOTOR_SPEED       130       FORWARD/BACK
rotation speed    120       hardcoded literal in Robot.move(), not a constant
strafe speed      100       hardcoded literal in Robot.move(), not a constant
CYCLE_MOVE_TIME   1.5 s
thermal pause     >80 °C    reads thermal_zone9 (BIG cores), sleeps 5s
```

**Vision** (`detect_person.py`)
```
yolo11s_320.onnx + yolo11s_640.onnx (stock COCO) · CONF 0.35 · IOU 0.45
SizePolicy: drive = 320 every frame + one 640 every 5 s; person_search = 640 until person box ≥ 1/3 frame height
legacy detect_scene default 640; legacy tuples in 640 space; detector dicts in source-frame pixels
position: left/center/right by thirds of frame width
coarse distance by box area: >0.3 very close, >0.1 close, >0.03 medium, else far
```

The detector is stock COCO. The earlier claim that yolo11m.onnx was fine-tuned
was wrong: no fine-tuned YOLO has ever existed (DECISIONS #121, correcting #97).

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
--threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 --ctx-size 2048
setup_q4     Gemma 4 E2B Q4_0    2.8GB  ~12 gen tok/s  ← robot default (since 2026-09-26)
Gemma E2B Q4_0 on this build: selector letter-scoring 1408 ms median (DECISIONS #119). MTP drafter measured +46% gen speed, +609 MiB, identical output; not deployed (DECISIONS #120). Build names: b1609 = upstream b10194, b2351 = upstream b10936 (#119).
setup_e4b    Gemma 4 E4B Q4_K_M  5.0GB  ~5.4 gen tok/s ← kept, not recommended (#115)
(setup_qwen3b / setup_qwen1b and the Qwen2.5 files were removed 2026-09-26.
Qwen3.5 2B and 4B GGUFs at ~/models/qwen35/ are eval-only, not wired into
server_manager; Qwen3.5-0.8B was deleted. See DECISIONS #106, #115.
#104's navigation removal was implemented in #111.)
```

Current llama.cpp build: commit e1a1abb7, version 1609, **rebuilt 2026-09-26
with `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`** at `~/llama.cpp-b1609-dotprod`
(commit 341bde6 points `server_manager.py` there; build commands and evidence in
`benchmark/llama_dotprod_rebuild/`). The original build had no dotprod
(0 `sdot` instructions vs 1045); the rebuild reads prompts ~3× faster and
generates ~+32% faster, with identical voice parses (DECISIONS #114). The
original tree is kept untouched as rollback. Original build details:
(Clang 21.1.8, Android aarch64). Shared-library build — `bin/llama-server` is a ~5.9 KB launcher, real
code in the `.so` files. Rollback of the original is the whole `build/` tree
(`build-b1609.tar.gz`, verified) or a rebuild of commit e1a1abb7, NOT a
single-binary copy (DECISIONS #73).

A third build, ~/llama.cpp-upstream rebuilt 2026-09-13 to b2351-790cf51a,
loads Qwen3.5-0.8B-Q4_K_M with architecture accepted and "modalities: text"
— MEASURED. At that text-load test, no mmproj was downloaded or passed;
later vision tests in #105 found no usable on-device LLM vision. Note: `strings`
on the rebuilt binary returns no qwen35 match despite the successful load —
the `strings` check used in #95 is not a reliable negative; an actual load
attempt is ground truth. b2351 has dotprod; with it, Gemma E2B Q4_0 scored
options in 1.55 s median on the ladder (DECISIONS #116). Its flash-attention
path segfaulted on this phone; the robot does not use b2351.

Benchmarked on build 1609, 15 cycles, realistic robot prompt shape (fixed system
prefix + varying scene): 11.5 tok/s median (11.1-11.9), prompt eval 1091 ms median,
free RAM ~3.1 GB, zone9 97-101 °C (CPU-core reading, not a heat state; #126). Prefix cache reuse confirmed on the live prompt
shape — 109 tokens on the first call, 18-22 thereafter. EVIDENCE CONFLICT: the
2026-09-26 rebuild report measured the original non-dotprod build at 6.5 gen
tok/s (200-token reply), not 11.5; the conditions of the earlier figure are not
recorded, so neither old figure is canonical. Current dotprod-build figures are
in DECISIONS #114 and #115.

Archived controlled conversation rates (cold vs cached block aggregate token-weighted gen tok/s for replies >=10 tokens): E2B Q4_0 12.10–12.36, E2B Q4_K_M 10.59–11.72, E4B 5.41–5.48, Qwen3.5 4B Q4_K_M 5.39–5.52, Qwen3.5 2B Q4_K_M 10.36–11.83. Sources: `benchmark/llm_objective_setting/conversation/conversation_20260926T185957Z.stdout.txt` (rates), `benchmark/llm_objective_setting/conversation/conversation_20260926T185957Z/run_20260926T185957Z.json`, and `benchmark/llm_objective_setting/conversation/conv_speed.py` (run model GGUF paths, `server_manager` binary/flags, ctx 2048, cores 4–7). No controlled dotprod chat speed result exists for the other menu entries in this archive. Note: Standalone chat uses ctx 4096 for some models, so these rates do not exactly describe its chat performance.

## Thermal governance

VIRTUAL-SKIN-CPU-GPU is the reported HAL CPU-throttling signal (first trip
37.0 °C); BIG/zone9 is not its HAL governor (DECISIONS #90). The live
`main.py` pause still reads BIG above 80 °C; a correct sensor and threshold
have not been selected or verified in a real mission. The dated device
thresholds, formula, zone map, throttle checks, and evidence limitations are
preserved in [PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).

Camera paths (3-min blocks, on battery): RobotCam 2.3 W vs
`termux-camera-photo` 3.3 W over 0.7 W idle (DECISIONS #122; its zone9 rise
rates are not trusted, #126). Measured (#126, room 22 °C): under RobotCam 2/s +
640 every frame + continuous Gemma, the CPU is first capped within 2 s while the
skin is about 32 °C; from 6 min the phone holds 3.68–4.24 W; status LIGHT at
78 s, MODERATE at 10.7 min, SEVERE (skin 45 °C) at 18.3 min. Co-residency
(#125): mix 4.1–4.3 W, 640-only 5.8–6.3 W. Heat rules decided, not implemented
(#127); `main.py` still pauses on zone9 > 80 °C. Every heat reading in the repo
is labelled in `benchmark/thermal_char/HEAT_EVIDENCE.md`. Power map, power split,
duty-cycle and camera-power measurements (#128) and the owner's heat-pause design
(#129, not implemented) are archived in `benchmark/power_map/`,
`benchmark/duty_cycle/` and `benchmark/camera_power/`.

## Memory

Reported 2026-09-11: Gemma-4-E2B Q4_K_M left approximately 0.5 GB practical
headroom after loading, without simultaneous full perception or dialogue.
This is a dated measurement, not a current co-residency test; see
[PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md) and
DECISIONS #94.

2026-09-26 (dotprod build, Gemma E2B Q4_0): peak server RSS ~4.3 GB in the
conversation benchmark. Without `--cache-ram 0`, one unique-question
conversation run grew ~1.9 MiB/turn (then dropped 82 MiB, cause unrecorded;
VmSwap not recorded); with it ~0.1 MiB/turn. Robot-shaped voice prompts did not
grow either way. Qwen3.5-4B without the flag grew until Android killed it.
`--cache-ram 0` is now passed (DECISIONS #115). Resident Gemma with RobotCam
and yolo11s (motors off) measured: minimum MemAvailable 2523 MiB, no LMK kills
(#125); with the drive loop it remains untested.

Standalone chat uses ctx 4096 for Gemma E2B Q4_0/Q4_K_M and Qwen3.5 2B Q4_K_M; ctx 2048 for all other menu models. No controlled peak memory result at ctx 4096; archived conversation peaks used ctx 2048.

Measured components (each alone): Gemma E2B Q4_0 selector peak 3920 MiB;
Gemma + MTP drafter conversation peak 4584 MiB; Qwen3.5-4B about 5.0 GB;
yolo11s RSS about 230 MiB (320) / 325 MiB (640); RobotCam app 36–39 MB PSS
plus camera provider about 270 MB. Measured together (#125): llama-server peak PSS 3851 MiB, runner 310 MiB, RobotCam 51 MiB, camera provider 253 MiB, minimum MemAvailable 2523 MiB; with the MTP drafter llama-server PSS 4154 MiB, MemAvailable 2619 MiB.

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
- RobotCam app (`android/robotcam/`, `af57939`) and integration (`9b49ce0`):
  640×480 at 2/s, fail-closed reader, `termux_photo` rollback. Motors-off checks
  only (native and `su -c --dry`); no motors-live run yet.
- Dual-size yolo11s detector with SizePolicy (`9b49ce0`), unit-tested (18 tests).
- Voice pipeline end to end: record → Whisper → Gemma JSON parse → TTS.
  `main.warm_up()` (called from `main.py` and `run_mission.py`) waits for the
  server and sends one discarded parse; measured cold start: server ready 19 s,
  warm-up 45.5 s, then five parses in 7.5–12.7 s under the 40 s timeout
  (DECISIONS #115).
- Manual selector playground `robot-jevlike`
  (`benchmark/strategic_selector/manual/jevlike/`): runs one input through any
  installed selector model side by side. Research only, not a benchmark.
- **Standalone chat** (`robot-chat`): Menu filters against installed models (Gemma E2B Q4_0/Q4_K_M/Q8_0, Gemma E4B Q4_K_M, Qwen3.5 2B Q4_K_M, Qwen3.5 4B Q4_K_M/Q5_K_M/Q6_K, Mistral 7B Q4_K_M). None of the menu entries other than the robot's deployed model (Gemma E2B Q4_0) are deployed to the robot loop. Image chat is unavailable in this interface.
- Benchmark archive index: `benchmark/INDEX.md` lists every archive folder,
  its key result and status.
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
  A separate offline objective-setting benchmark of on-device GGUF models
  (Gemma-4-E2B/E4B, Qwen3.5-2B) ran on build 1609; results, caveats and
  incomplete portions are archived with their run artifacts at
  `benchmark/llm_objective_setting/` (see DECISIONS #110). It scores a proposed
  dynamic-map prompt, not the deployed enum, its conversation/Q&A buckets are
  ungraded, and two rubric defects were found after the fact. It is a baseline
  for the LLM's retained conversation/objective-setting role under #109, not
  evidence about navigation.
- Strategic-selector evidence: a 2026-09-23 phone audit confirmed four archived
  runs and benchmark sources published on origin/main; publication does not
  establish model accuracy or robot hardware validity (see
  [RUN_INDEX.md](../benchmark/strategic_selector/RUN_INDEX.md)).

## Pending / untested

- **Dual-rate nav merged as a rewrite. Rule-level safety branches and the current
  `run_cycle()` path are regression-tested; everything else is untested.**
  `navigate_rules` escalation ladder verified against synthetic
  distances via `nav_test.py`: thresholds exact at the `<15` / `<25` / `>=25`
  boundaries, no oscillation, `nav_stuck` fires on exactly one cycle per episode,
  60s timeout reached. NOT tested: the person branches
  (`estimate_distance_single`), and the whole loop with motors live —
  `run_mission.py` has only ever been run with `--dry`. The 93% figure still does
  not transfer (#55). The #104 removal is covered by `run_cycle_safety_test.py`
  in both avoidance directions with fake external operations; no hardware
  validation was performed (DECISIONS #111).
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
- **Gemma is no longer consulted during `nav_stuck`.** `run_cycle()` executes
  the move from `navigate_rules()` directly; `nav_stuck` and `asked_gemma` are
  still set by the unchanged rules but have no consultation consumer.
  `run_cycle_safety_test.py` checks both avoidance directions and no outbound
  LLM request with fake external operations. This is regression coverage, not
  hardware validation. See DECISIONS #81, #88, and #111.
- **Dead and orphaned code mapped** (`FILES.md`). Superseded: `detect_scene.py`,
  `llm.py`, `voice.py`, `ch340_test.py`, `thermal_benchmark.py`,
  `thermal_benchmark2.py`, `quality_benchmark.py`. Orphans never wired in:
  `get_temp.py`, `thermal_guard.py`, `server_manager.py` (nothing starts the
  server programmatically). `mission_test.py` predates the current `Robot` API
  and is probably broken.
- **Orphaned-file deletion deferred (DECISIONS #85).** `detect_scene.py`,
  `llm.py`, `voice.py`, `ch340_test.py`, `thermal_benchmark.py`,
  `thermal_benchmark2.py`, and `quality_benchmark.py` remain tracked and present.
  DECISIONS #84's stated `git rm` did not occur in the checked repository; retain
  all seven until an explicit future removal task. `get_temp.py`/`thermal_guard.py` (#75) and `mission_test.py` remain
  deliberately kept; `nav_sim.py` was deleted (#118). 420 MB of unused ONNX models remains untouched.
- **`get_temp()` pause (zone9 > 80 °C) reads the wrong signal.** zone9 is a
  CPU-core reading: it reaches about 100 °C within seconds under load, then is
  held at 61–68 °C while the skin keeps rising (#126). The replacement rules are
  decided but not implemented (#127).
  `thermal_guard.py`'s 82/86 are dead code — nothing imports it and `main.py` uses
  an inline `get_temp`. A replacement needs the correct thermal signal and a
  threshold validated against a real mission trace; neither is selected.
  See DECISIONS #74, #75, #90, #126, #127.
- **Battery thermal zone number unconfirmed.** STATUS previously implied zone
  22; a later enumeration reported zone 25. Neither is canonical until a
  direct thermal_zoneN/type read confirms it.
- **Two JSON schemas in circulation.** `main.py: PARSE_SYS` emits
  `{"type": ..., "name": ..., "room": ...}`; the benchmark scripts used
  `{"action": ..., "target": ...}`. Not reconciled.
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
- Not yet built/wired: Piper TTS, Whisper VAD, room classifier, memory system,
  face recognition.
- Candidate speed and quality work (DECISIONS #117):
  grammar-constrained JSON output for voice parsing; per-step timing of
  Whisper (never measured; YOLO measured, #121); Moonshine (English-only, compute scales with audio
  length) against Whisper on the same recordings; a Vulkan GPU feasibility test
  in native Termux; YOLO26 (same speed as YOLO11 at equal size; NMS-free mode
  not worth it, #121); LFM2.5 and SmolLM3-3B as conversation
  candidates, both needing a newer llama.cpp than b1609.
- The proposed strategic selector is not wired to the robot. Apartment
  localization from 2D LiDAR, semantic room map, camera–LiDAR object/range
  association, named-person verification, eligible-script gates, safe
  interruption, and model/sensor co-residency need separate validation.
  YOLO `person` detection does not establish Chiara's identity; Laya/Von
  confidence is not a braking guarantee. Script duration and whether an
  additional car-mounted accelerometer helps remain undetermined. The
  current ultrasonic sensor and phone sensors have not been integrated
  into this proposed selector.
- Objective-setting benchmark: the rubric fixes were applied and five models
  rerun on the dotprod build 2026-09-26 (think-off for Qwen3.5); buckets A and
  B were blind-graded by Local AI (DECISIONS #115, archive
  `benchmark/llm_objective_setting/`). Qwen3.5 think-on results remain invalid
  (42% truncated) and must not be quoted. Still open: all models invent sensor
  readings they do not have (battery, why it stopped, vision range), lack
  self-knowledge (wheels, why it stops), sometimes reply in the wrong language,
  and occasionally leak `</start_of_turn>` into replies. A robot fact sheet,
  live sensor values and an English-only reply rule in the system prompt
  (English-only is acceptable, DECISIONS #117), plus stripping template
  tokens, are the planned fixes; not implemented.
- b2351 flash-attention segfaults on this phone; not investigated. The robot
  stays on the dotprod b1609 build.
- Repo root cleaned 2026-09-27 (#118, commit `78b1cc5`): untracked scratch
  files deleted; #105 vision files archived in `benchmark/qwen_vision/` (edited
  harness not re-reviewed, result files unlocated) and the #108 verification
  output in `benchmark/failclosed_108/`.
- benchmark/llm_objective_setting/ was published with the docs/WORKFLOW.md
  independent review WAIVED by human decision; the waiver is recorded in its
  README.
- YOLO power map done (#128): chosen setting 1 frame/s, 640 every 5 s, MID
  cores (4–5); continuous use reaches skin about 37 °C in about 17 min, so
  heat pauses are needed (#129).
- Implement the #127 heat rules and the #129 pause design (camera off,
  sensor-only moves, Gemma resident, torch in the dark) in `main.py`
  (navigation work); needs independent review and a mission thermal trace.
- RobotCam manual-exposure 2 fps (mode A) is a candidate on unmerged branch
  `robotcam-camera-power` (90ef439); not adopted; torch untested (#128, #129).
- RobotCam screen-off capture is untested; frame loss ends a mission
  (fail-closed, no retry by design).
- Evaluate RelateAnything speed, RAM and weight licence after co-residency (#124).
- Gemma fine-tuning restarts from scratch; old LoRA notebooks discarded (#124).

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
   the #127 heat rules (#75, #90, #127).

Mode 2 video/audio teleop remains parked indefinitely for thermal cost (#58).
This does not park the serial-command firmware named `mode2_auto.ino`.

## Last hardware test

Bench power-up, first four-motor PWM spin, and HC-SR04 readings were reported.
The ESP32 driving motors from buck power remains unverified after an earlier
brownout; capacitors, solid ground return, and current-wiring direction tests
remain pending. Full measured values and as-built limitations are preserved in
[PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).
