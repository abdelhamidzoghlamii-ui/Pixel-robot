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
RobotCam frame (640×480, 1/s, fail-closed reader) → YOLO yolo11s (320, 640 every 5 s or in person search; 2 threads on MID cores 4–5) → navigate_rules (pure Python, instant; shaped by the picker's mode) → person-stop (area bucket) → execute move
Picker (background worker, BIG cpus 6–7; ACTIVE only, never at Android status SEVERE): RelateAnything M2 on the latest 640 frame → Gemma letter scoring of EXPLORE / APPROACH / GREET / LOOK → mode for the next cycles (events, min gap 5 s; heartbeat 20 s). In the motors-off run it always chose EXPLORE (#139, #140).
```

| Layer | Handles | Cost |
|---|---|---|
| YOLO yolo11s (320/640 policy, both sessions resident; yolo11m kept for rollback) | object/person detection, position, coarse distance | every cycle |
| RobotCam app (Camera2 foreground service) | 640×480 frames, 1/s (#136); missing/old/repeated frame → STOP | continuous while a mission runs |
| Python `navigate_rules` | obstacle stop, person approach (APPROACH mode or `--no-picker`), room signature logging | every cycle, 0ms |
| Person-stop (#79, #138) | any person in bucket `close`/`very close` turns a forward move into HOLD (stop for the cycle; the mission continues) | every cycle |
| Gemma 4 E2B (Q4_0, default since 2026-09-26) | mission picker (letter scoring of the offered modes, #138) and voice→intent; never a motor command | picker: median 2.8 s per pick in the motors-off run (#139) |
| RelateAnything M2 (relsgg-vits16, fp32, `relate.py`) | object-relation text for the picker (layout L2, #131, #138) | before each pick; median 0.73 s in the motors-off run (#139) |
| Whisper.cpp (base) | 5s voice capture → text | on demand |
| Gemma `PARSE_SYS` | voice text → JSON action array | on demand |
| `termux-tts-speak` | speech output | on demand |

The high-level selector (#109) is wired into the mission loop as the picker
(#138, `2ce2b3b`), on by default; `--no-picker` keeps the old rules. In the
motors-off run it chose EXPLORE every time and cut the ACTIVE share to 0.48
(#139); desk tests found no prompt or model that qualifies (#140). Owner
direction (#140): Gemma becomes the robot's "mind", called at events (talk,
describe, remember, plan); Python keeps every move. Updated owner direction
(#146, 2026-10-29; DESIGN ONLY): begin with a Python-mediated, event-driven
Gemma E2B high-level picker, initially offering a fixed bounded set of actions.
Python handles sensor interpretation, safety, motor execution, heat and scheduling;
Gemma proposes only high-level strategy. Evaluate cached navigation context,
relevant mission memory, picker quality, latency and MTP before deciding whether
to allow more flexible Python operations or independently warm navigation and
companion contexts. These interfaces and caches are NOT implemented or qualified.
Whether `--no-picker` becomes the default is open. Fine-tuning (Laya, Von or
Gemma) is deferred to a later robot version. One resident Gemma for talk and
selection was decided in #125.
The current repo describes a mecanum prototype; the phone-mounted RC car with
2D LiDAR is a proposed redesign.

## Known-good config (as coded)

**Navigation** (`main.py`)
```
OBSTACLE_DIST     25 cm     hard stop threshold
PERSON_STOP_BUCKETS ('close', 'very close')  any person in these area buckets → forward move becomes HOLD (#79, #138)
LOOK_CYCLES       4         LOOK rotates in place, then EXPLORE (#138)
picker            on by default; `--no-picker` = old rules; server from setup_q4 (#138)
STEREO_BASELINE   5.0 cm    strafe distance for depth   (UNVERIFIED)
MOTOR_SPEED       130       FORWARD/BACK
rotation speed    120       hardcoded literal in Robot.move(), not a constant
strafe speed      100       hardcoded literal in Robot.move(), not a constant
CYCLE_MOVE_TIME   1.5 s
heat control      adaptive VIRTUAL-SKIN 37/35 °C pauses + #127 stops (heat_control.py, #135)
```

**Vision** (`detect_person.py`)
```
yolo11s_320.onnx + yolo11s_640.onnx (stock COCO) · CONF 0.35 · IOU 0.45
SizePolicy: drive = 320 every frame + one 640 every 5 s; person_search = 640 until person box ≥ 1/3 frame height
robot process: one PinnedDetector, 2 intra-op / 1 inter-op threads, ORT workers + calling thread pinned to cpus 4–5 (#136)
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
setup_q4_mtp setup_q4 + MTP drafter  opt-in, not the robot default (#137)
Gemma E2B Q4_0 on this build: selector letter-scoring 1408 ms median (DECISIONS #119). MTP drafter measured +46% gen speed, +609 MiB, identical output (DECISIONS #120); on in robot-chat, opt-in for the robot (`setup_q4_mtp`), not in the robot default (#137). At chat settings: +43% gen speed, memory cost not established (#137). Build names: b1609 = upstream b10194, b2351 = upstream b10936 (#119).
setup_e4b    Gemma 4 E4B Q4_K_M  5.0GB  ~5.4 gen tok/s ← kept, not recommended (#115)
setup_e4b_q4     Gemma 4 E4B Q4_0  4.8GB  7.31 gen tok/s at chat settings ← opt-in, not the robot default (#141)
setup_e4b_q4_mtp setup_e4b_q4 + E4B MTP drafter  10.57 gen tok/s INDICATIVE; opt-in (#141)
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

Research-only alternative, NOT the deployed server: `~/llama.cpp-vis`, llama.cpp
b11716, source commit 797de63b2fa7396dcc8caca76834c699ac84e1c3 (the b11716
label is not verified locally: shallow clone). VIS-3D owner-run text rates with
the E2B Q4_0 model were 13.22 tok/s without projector/MTP, 13.46–13.53 tok/s
(two B runs) with Q8 projector and MTP off, and 18.11 tok/s with projector and
MTP on. Individually VALID, but cross-run environmental comparisons UNMATCHED.
The original b1609 with five-second monitoring reproduced 13.35 OFF / 18.55 ON
tok/s in VIS-3C under matched paired conditions (owner-reported label). Do not
interpret these as robot deployment/co-residency results (#142, #143).

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
37.0 °C); BIG/zone9 is not its HAL governor (DECISIONS #90). `run_mission` uses `heat_control.py`
(#135): Android status and VIRTUAL-SKIN every 5 s, adaptive 37/35 °C camera-off
pauses and the #127 stops; validated in one motors-off dry run, not with motors live. The dated device
thresholds, formula, zone map, throttle checks, and evidence limitations are
preserved in [PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).

Camera paths (3-min blocks, on battery): RobotCam 2.3 W vs
`termux-camera-photo` 3.3 W over 0.7 W idle (DECISIONS #122; its zone9 rise
rates are not trusted, #126). Measured (#126, room 22 °C): under RobotCam 2/s +
640 every frame + continuous Gemma, the CPU is first capped within 2 s while the
skin is about 32 °C; from 6 min the phone holds 3.68–4.24 W; status LIGHT at
78 s, MODERATE at 10.7 min, SEVERE (skin 45 °C) at 18.3 min. Co-residency
(#125): mix 4.1–4.3 W, 640-only 5.8–6.3 W. Heat rules (#127) and the #129 pauses
are implemented with the adaptive 37/35 °C schedule (#135); the #129 torch is not. Every heat reading in the repo
is labelled in `benchmark/thermal_char/HEAT_EVIDENCE.md`. Power map, power split,
duty-cycle and camera-power measurements (#128) and the owner's heat-pause design
(#129; implemented in #135 except the torch) are archived in `benchmark/power_map/`,
`benchmark/duty_cycle/` and `benchmark/camera_power/`.

## Memory

MEM-1 PRIMARY: owner reports a completed phone-local benchmark and an owner-
verified artifact manifest (`MEM1_ALL_HASHES_OK`, reported 2026-10-29).
GitHub still holds the prepared harness and no archived run artifacts. No
original run outputs were inspected in this documentation transition; exact
results, run labels, and any broader MEM-1 completion claim await the original
artifacts and archival/review checks. No production memory feature is implied.

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

The as-coded/deployed image request was previously confirmed to run as text
only (DECISIONS #96); b2351 projector experiments did not produce usable
on-device LLM vision (DECISIONS #105). These historical findings still apply to
those paths. Separately, the research-only llama.cpp b11716 build successfully
processed eight saved images with a Q8_0 Gemma 4 E2B vision projector in VIS-3D
on 2026-10-23: 8/8 completed, median wall latency 4.42 s for images 2–8, with
correct Gemma 4 chat templating and a 70-token image budget. Qualitative
accuracy is inconsistent: the close couch and partly visible person were
omitted entirely in one response. Vision is NOT integrated into deployed
robot-chat or the mission loop. YOLO and the existing sensor/rule path remain
responsible for deployed perception and navigation; b11716 vision has not been
validated for safety-critical decisions. The original live-call observations
and follow-up are retained in [PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).
See DECISIONS #142–#145. VIS-0 (2026-10-18, BF16 projector, same research
build) completed 40/40 image calls, but two of its server shutdowns exited with
SIGABRT, not investigated (#145). The VIS-3C, VIS-3D, VIS-0 and M2-Q1 archives
and catalogs are in commit `4fd2b98` (#144, #145).

## Working now

- LLM server stable; prompt-cache fix landed (`--swa-full`) — repeat-prefix calls
  reuse the cached system prompt instead of reprocessing it.
- YOLO scene detection, position + coarse distance.
- RobotCam app (`android/robotcam/`, `af57939`) and integration (`9b49ce0`):
  640×480 at 1/s since `e208aaf` (#136; 2/s before), fail-closed reader, `termux_photo` rollback. Motors-off checks
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
- **Standalone chat** (`robot-chat`): Menu filters against installed models (Gemma E2B Q4_0, Gemma E4B Q4_K_M/Q4_0, Qwen3.5 2B Q4_K_M, Qwen3.5 4B Q4_K_M/Q5_K_M/Q6_K, Mistral 7B Q4_K_M). None of the menu entries other than the robot's deployed model (Gemma E2B Q4_0) are deployed to the robot loop. Image chat is unavailable in this interface. The Gemma E2B Q4_0 and Gemma E4B Q4_0 entries run their MTP drafters (#137, #141); without the drafter file chat warns and starts without it. The menu shows only models present on disk.
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
  `get_temp.py`, `thermal_guard.py`. `server_manager.py` is no longer an orphan:
  `chat.py` imports its MTP drafter constants (#137); the robot still does not
  start the server programmatically (the owner runs `setup_q4`). `mission_test.py` predates the current `Robot` API
  and is probably broken.
- **Orphaned-file deletion deferred (DECISIONS #85).** `detect_scene.py`,
  `llm.py`, `voice.py`, `ch340_test.py`, `thermal_benchmark.py`,
  `thermal_benchmark2.py`, and `quality_benchmark.py` remain tracked and present.
  DECISIONS #84's stated `git rm` did not occur in the checked repository; retain
  all seven until an explicit future removal task. `get_temp.py`/`thermal_guard.py` (#75) and `mission_test.py` remain
  deliberately kept; `nav_sim.py` was deleted (#118). 420 MB of unused ONNX models remains untouched.
- **Heat control (#135) is validated motors-off only.** One unattended `--dry`
  run (5 h, 67 pauses, 0 rule violations) ended by low battery; emergency rules
  (CRITICAL hold, battery 45 °C, CPU fault) have not fired on hardware. Pause
  FORWARD speed 100 is uncalibrated. `thermal_guard.py` (82/86) and `get_temp.py`
  remain orphans; `main.py` no longer has `get_temp()`. See DECISIONS #126, #127,
  #129, #135.
- **Battery thermal zone number unconfirmed.** STATUS previously implied zone
  22; a later enumeration reported zone 25. Neither is canonical until a
  direct thermal_zoneN/type read confirms it.
- **Two JSON schemas in circulation.** `main.py: PARSE_SYS` emits
  `{"type": ..., "name": ..., "room": ...}`; the benchmark scripts used
  `{"action": ..., "target": ...}`. Not reconciled.
- **FOCAL_PX uncalibrated; person-stop now uses the area bucket.** Overshoot
  +55% to +1000% across 12 measured photos. Person-stop no longer uses the
  distance estimate (#138): in the bench-photo table (visual distance labels)
  `close` stops in 26 of 34 detector rows, including people labelled 170–260 cm;
  `very close` in 7 of 34. At 45 cm one 320 frame gave `medium` (no stop). The
  area bucket tracks distance poorly; the 25 cm ultrasonic stop still applies.
  In the motors-off run it held 168 forward moves (#139). See DECISIONS #78, #79, #80, #138.
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
  harness not re-reviewed; the only recovered result is an aborted
  battery-precondition attempt, no successful run, #133) and the #108 verification
  output in `benchmark/failclosed_108/`.
- benchmark/llm_objective_setting/ was published with the docs/WORKFLOW.md
  independent review WAIVED by human decision; the waiver is recorded in its
  README.
- YOLO power map done (#128): chosen setting 1 frame/s, 640 every 5 s, MID
  cores (4–5); continuous use reaches skin about 37 °C in about 17 min, so
  heat pauses are needed (#129). Deployed in `e208aaf` (#136); motors-off dry run
  5.26 h archived (#139: 0 camera stops, cycle 1.80 s, ACTIVE duty 0.838 from the
  first pause). Motors-live run pending.
- Heat rules and pauses implemented (#135, commit `6f81f2a`); motors-off dry run
  archived in `benchmark/heat_control/dryrun1/` (`92492ec`). Torch for dark
  scenes not designed (#129).
- RobotCam manual-exposure 2 fps (mode A) is a candidate on unmerged branch
  `robotcam-camera-power` (90ef439); not adopted; torch untested (#128, #129).
- RobotCam screen-off capture is untested; frame loss ends a mission
  (fail-closed, no retry by design).
- RelateAnything M2 (relsgg-vits16, our fp32 export): speed, RAM and co-residency
  measured (#130–#132); used by the picker (`relate.py`, L2, #138); model files
  stay outside the repo and are checked by SHA-256 at load; weight licence for
  shipped use not evaluated. Relation quality (M2-Q1, #145): 63 % of shown lines
  strictly right vs the 70 % bar — M2 as built FAILS.
- Picker (#138): motors-off run done (#139) — always EXPLORE, ACTIVE share 0.48;
  no qualifying prompt or model in the desk tests (#140). Open owner decision:
  make `--no-picker` the default. Next research gate (#146): design a Python-
  mediated mind interface and compare the old picker with event-driven, cached,
  memory-aware high-level selection using independently labeled motors-off cases.
  Measure quality before accepting speed gains; include option-order changes,
  prohibited choices, verified cache_n/reused prefix, per-request latency,
  memory relevance and distractors, MTP on/off, RAM, heat and mode-switch cost.
  Current picker uses cache_prompt false; two independently warm modes and
  flexible Python operations are only possibilities, not working features.
- Gemma fine-tuning restarts from scratch; old LoRA notebooks discarded (#124).
- VIS-3C / VIS-3D / VIS-0 / M2-Q1 archives and vision qualification
  (#142–#145): archives, catalogs, REGENERATED comparison reports and review
  records are checksum-verified and archived in commit `4fd2b98`; the archival
  review records themselves remain on the phone. The archival reviews are accepted only as task-specific
  owner exceptions (VIS-3C/VIS-3D R4; VIS-0/M2-Q1 R1, which stays INVALID).
  VIS-3D image quality requires an agreed evaluation rubric; robot-critical
  missed objects remain a concern. VIS-0's SIGABRT shutdown exits are
  uninvestigated. Controlled projector/MTP overhead, longer-run thermal
  behavior, memory attribution, prompt and image-token-budget tradeoffs, and
  simultaneous operation with camera/YOLO/sensors remain unverified. No vision
  integration into the mission loop is approved. Existing camera and ultrasonic
  safety responsibilities remain separate; lidar is not integrated into
  navigation safety logic (2D LiDAR is a proposed redesign; integration
  unverified).

## Work priorities

Preserved from the 2026-09-05 handoff; these are pending tasks, not new test results
or permission to execute hardware work.

1. Keep `nav_test.py` and the #81 `run_cycle()` regression passing before the
   first motors-live autonomous run. The code defect is fixed, but no hardware
   validation has been performed (DECISIONS #88).
2. Person-stop area bucket implemented (#138, `2ce2b3b`) with `close` (owner
   decision 2026-10-15, after the `bench_photos/` table); motors-off only (#139).
   A better distance signal needs tape-measured photos.
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
   control, with wheels on a stand. `run_mission` writes the heat trace to
   `~/robot_runs/`; keep it with the run (#127, #135).

Mode 2 video/audio teleop remains parked indefinitely for thermal cost (#58).
This does not park the serial-command firmware named `mode2_auto.ino`.

## Last hardware test

Bench power-up, first four-motor PWM spin, and HC-SR04 readings were reported.
The ESP32 driving motors from buck power remains unverified after an earlier
brownout; capacitors, solid ground return, and current-wiring direction tests
remain pending. Full measured values and as-built limitations are preserved in
[PROTOTYPE_EVIDENCE.md](PROTOTYPE_EVIDENCE.md).
