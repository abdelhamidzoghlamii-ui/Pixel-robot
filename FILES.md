# File Inventory

Only four files are imported anywhere: **`main.py`**, **`motors.py`**, **`detect_person.py`**,
**`stereo_depth.py`**. Everything else is a standalone script (entry point, benchmark, test, or
diagnostic) that nothing else in the directory imports. "Imported/called by" below means Python
`import`; scripts you'd run by hand are marked *(entry point)*.

## Core runtime (the live stack)

| File | What it does | Imported/called by |
|---|---|---|
| `main.py` | `Robot` class: perceive→navigate→LLM→move cycle loop; the robot brain. | `mission_test.py`, `nav_test.py`, `run_mission.py` |
| `motors.py` | `Motors` class — CP2102 USB-serial link to the motor MCU (drive/strafe/rotate, ultrasonic distance, keepalive thread). | `log_run.py`, `run_mission.py`, `teleop.py` |
| `detect_person.py` | YOLO11m ONNX inference: `detect_scene`, `scene_to_text`, `person_direction`, cached session, legacy `detect_person` shim. | `main.py`, `mission_test.py`, `stereo_depth.py`, `test_suite.py`, `test_suite_m.py` |
| `stereo_depth.py` | Two-frame stereo + object-height→distance estimation (`stereo_scan`, `scene_with_depth`, `estimate_distance_single`). | `main.py`, `mission_test.py` |
| `run_mission.py` | Autonomous launcher: wires `Motors`+`Robot`, runs a mission from argv (`--dry` = no motors). | nobody *(entry point)* |
| `teleop.py` | HTTP server for manual driving — web UI, obstacle-gated forward, camera passthrough. | nobody *(entry point)* |
| `chat.py` | Interactive terminal chat client for the local llama-server: model menu, server start/kill, camera/vision. | nobody *(entry point)* |

## Calibration tooling (current, Aug 16)

| File | What it does | Imported/called by |
|---|---|---|
| `log_run.py` | Drive forward at a duty toward a wall, log travel-vs-time to CSV via ultrasonic. | nobody *(entry point)* |
| `identify_it1.py` | Fit an IT1 (integrator+lag+deadtime) motion model to `log_run.py`'s CSV; print duration-for-distance table. | nobody *(entry point)* |

## Hardware bring-up / diagnostics

| File | What it does | Imported/called by |
|---|---|---|
| `cp2102_test.py` | CP2102 USB enumerate + interface-claim smoke test (current adapter). | nobody *(entry point)* |
| `dist_raw.py` | Dump raw ultrasonic serial lines from the CP2102 for 5 s. | nobody *(entry point)* |
| `ch340_test.py` | Same idea for the **old CH340** adapter, replaying a Wireshark init sequence. | nobody — **superseded** |
| `diagnose_yolo.py` | One-shot YOLO ONNX diagnostic (prints I/O shapes, runs one image). | nobody — **stale** (defaults to `yolov8n.onnx`; stack uses `yolo11m`) |

## Thermal (six overlapping files)

| File | What it does | Imported/called by |
|---|---|---|
| `get_temp.py` | Root per-zone SoC temp reader (BIG/MID/LITTLE/GPU/TPU/battery), `get_temps`/`get_cpu_max`. | nobody — **orphan**: recent, but `main.py` uses its own inline `get_temp` |
| `thermal_guard.py` | `big_temp()` + throttle/cooldown helper (WARN 82 / CRIT 86). | nobody — **orphan**: `main.py` does thermal inline |
| `thermal_real.py` | Benchmark: llama-server tok/s vs real CPU temp. | nobody *(benchmark)* |
| `thermal_benchmark3.py` | Benchmark: combined YOLO+LLM thermal load with real temps (newest of the line). | nobody *(benchmark)* |
| `thermal_benchmark2.py` | Earlier thermal benchmark (zone9 + battery zone25). | nobody — **superseded** by `thermal_benchmark3.py` |
| `thermal_benchmark.py` | Earliest — uses battery temp as an SoC proxy. | nobody — **superseded** |

## LLM / nav benchmarks

| File | What it does | Imported/called by |
|---|---|---|
| `build_bench.py` | A/B latency bench vs whatever llama-server is on :8080 (cached prefix + fresh scene). Newest file. | nobody *(benchmark)* |
| `benchmark_nav.py` | Bench Gemma nav decisions (FORWARD/LEFT/…) against scripted scenes. | nobody *(benchmark)* |
| `swa_benchmark.py` | Bench sliding-window-attention cache effect on nav-cycle latency (the 16s→5s fix). | nobody *(benchmark)* |
| `system_benchmark.py` | Bench end-to-end llama-server tok/s with temp logging, 10 runs. | nobody *(benchmark)* |
| `full_benchmark.py` | Broad LLM bench (multi-zone temp scan + timed calls). | nobody *(benchmark)* |
| `benchmark_compare.py` | Compare command-parser prompt variants (JSON action array) across models. | nobody *(benchmark)* |
| `benchmark_lora.py` | Bench a LoRA-tuned parser against the short training-format prompt + test set. | nobody *(benchmark)* |
| `json_benchmark.py` | Bench JSON command-parsing accuracy/latency. | nobody *(benchmark)* — overlaps `benchmark_compare.py` |
| `quality_benchmark2.py` | Bench Gemma answer *quality* on nav/vision prompts (150 tok). | nobody *(benchmark)* |
| `quality_benchmark.py` | Earlier version (80 tok). | nobody — **superseded** by `quality_benchmark2.py` |
| `nav_logic_test.py` | Standalone dual-rate nav bench: Python rules + Gemma only for explore/give-up, scored vs cases. Prototype of `Robot.navigate_rules`. | nobody *(benchmark)* |
| `nav_sim.py` | Older nav sim driving a **Qwen2.5-3B** gguf directly with its own thermal loop. | nobody — **superseded** (stack moved to Gemma E2B on :8080) |

## Tests

| File | What it does | Imported/called by |
|---|---|---|
| `nav_test.py` | Exercises `Robot.navigate_rules` safety branches directly (no motors/server). Current (Aug 16). | nobody *(entry point)* |
| `mission_test.py` | Runs `Robot` with a `SimulatedMotors` stub through a scripted mission. | nobody — **likely stale** (Apr 11; predates the `navigate_rules`/`run_cycle` rewrite in `main.py`) |
| `test_suite_m.py` | Camera-in-the-loop detection test, trimmed (2 people, close/far). | nobody *(entry point)* |
| `test_suite.py` | Fuller version of the same (people×distance×direction×reps + empty shots). | nobody — largely **superseded** by `test_suite_m.py` |
| `yolo_benchmark.py` | Bench YOLO ONNX inference speed/temp over `test_photos`. | nobody *(benchmark)* |
| `yolo_scene_test.py` | One-shot YOLO scene detection on a fixed test photo, prints detections. | nobody *(entry point)* — overlaps `diagnose_yolo.py` |

## Other utilities

| File | What it does | Imported/called by |
|---|---|---|
| `server_manager.py` | Programmatic llama-server lifecycle (kill/start/wait, named setup configs), CLI `server_manager.py <setup>`. | nobody — **orphan** (only named in a `benchmark_compare.py` help string; `run_mission.py`/`main.py` assume the server is already up) |
| `llm.py` | Old command parser: GBNF-grammar-constrained JSON actions via ChatML `<|im_start|>` prompt; needs `~/actions.gbnf`. | nobody — **superseded** by `main.parse_command` (Gemma `<start_of_turn>` format, no grammar) |
| `voice.py` | Standalone `listen(duration)`: mic → ffmpeg → whisper-cli transcription. | nobody — **superseded**: `main.listen()` is a copy with the same paths |
| `detect_scene.py` | Earlier YOLO module: `detect()→(results, elapsed)`, 4-tuple detections, no session cache. | nobody — **superseded** by `detect_person.py` (only the *symbol* `detect_scene`, defined in `detect_person.py`, is used) |

## Dead / superseded — flagged

**Clearly obsolete (safe-to-delete candidates):**
- `detect_scene.py` — replaced by `detect_person.py`
- `llm.py` — replaced by `main.parse_command`
- `voice.py` — replaced by `main.listen`
- `ch340_test.py` — old CH340 hardware; adapter is now CP2102 (see `motors.py`)
- `nav_sim.py` — Qwen-era nav sim, pre-Gemma
- `thermal_benchmark.py`, `thermal_benchmark2.py` — replaced by `thermal_benchmark3.py` / `thermal_real.py`
- `quality_benchmark.py` — replaced by `quality_benchmark2.py`

**Written but never wired in (orphans — decide keep-as-tool or drop):**
- `get_temp.py`, `thermal_guard.py` — recent, but `main.py` still does temperature inline and imports neither
- `server_manager.py` — nothing starts the LLM server programmatically

**Weaker overlap (one of each pair is redundant):**
- `test_suite.py` vs `test_suite_m.py`
- `json_benchmark.py` vs `benchmark_compare.py`
- `diagnose_yolo.py` / `yolo_scene_test.py` vs `yolo_benchmark.py`
- `mission_test.py` — predates the current `main.Robot` API; probably broken
