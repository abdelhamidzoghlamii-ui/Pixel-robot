# Heat evidence register (2026-10-01)

Which heat readings in this repository (and in the phone-only run folders named below) can be trusted. Written by
the ARCHIVE1 task; it labels evidence and draws no new conclusions.

## The rule (owner decision, 2026-10-01)

- **VALID heat evidence** judges the phone's heat state from **VIRTUAL-SKIN**, the **Android thermal status** and
  **cpufreq caps** (`scaling_max_freq` below `cpuinfo_max_freq`). Two runs do this: thermal_char
  `run_20260930T233208Z` and co-residency `run_20261001T041743Z`. Their smoke runs (thermal_char
  `run_20260930T231117Z_smoke`, co-residency `run_20261001T040522Z_smoke`) are **VALID AS FUNCTIONAL CHECK ONLY**:
  they show the tools work, but they are not measurements.
- **NOT TRUSTED:** any heat conclusion (how fast the phone heats, heat limits, cooldown times, "too hot", gate
  temperatures read as a heat state) drawn from zone9, zone10, zone11 or other CPU-core sensors. These sensors jump
  to about 100 °C within 2 s under load and drop within 10 s, and the kernel caps the CPU around 100 °C. zone9 read
  100 °C while the skin was 34.6 °C, and 62 °C while the skin was 45 °C.
- **STILL VALID WITH CAVEAT:** speed, RAM and power numbers from older runs remain valid as measured. Their cooldown
  gates used zone9, so whether the CPU was capped during them is unknown.
- **NOT A HEAT TEST:** files that mention temperature only in passing.
- **NOT ARCHIVED:** phone-only runs left out of the repository; their label is given too.

### The two VALID measurement runs (numbers copied from their reports)

**thermal_char `run_20260930T233208Z`** (`runs/run_20260930T233208Z/report.txt`; room 22 °C, phone in the robot
mount, on battery). Load: RobotCam 2 frames/s + yolo11s 640 on every frame + Gemma E2B Q4_0 streaming back-to-back,
1200 s, then 300 s cooldown logging.
- policy0 `scaling_max_freq` 1803 → 1401 MHz at load +1.8 s (first cap); policy6 2850 → 2802 MHz at +3.8 s.
- Android status 0 → 1 (LIGHT) at +77.9 s, 1 → 2 (MODERATE) at +642.9 s, 2 → 3 (SEVERE) at +1098.0 s; in cooldown
  3 → 2 at +1252.9 s, 2 → 1 at +1327.8 s.
- VIRTUAL-SKIN 31.6 °C at load start, 37.0 at +30 s, 40.4 at +120 s, 42.9 at +600 s, 45.4 at the stop (+1200 s).
- Max MHz from +600 s to the stop: policy0 738–930, policy4 1024–1197, policy6 984.
- First vs last minute: 640 ms median 1811 → 2967; tok/s streamed 5.6 → 3.2.
- Cooldown: skin start 31.4 °C, not within 2 °C after 305.7 s (last 38.7 °C).

**co-residency `run_20261001T041743Z`** (`../coresidency/runs/run_20261001T041743Z/report.txt`; 4 × 180 s blocks,
gated on skin ≤ idle skin + 1.5 °C and zone9 ≤ idle + 4 °C; idle skin 31.4 °C).

| | B1_mix_nollm | B2_only640_nollm | B3_mix_gemma | B4_only640_gemma |
|---|---:|---:|---:|---:|
| VIRTUAL-SKIN start/end/max °C | 31.4/37.8/37.8 | 32.8/40.0/40.2 | 32.8/39.4/39.4 | 32.8/40.0/40.3 |
| Android status max | 0 | 1 | 1 | 1 |
| policy capped s (% of block) | 178 (99%) | 179 (99%) | 179 (99%) | 179 (99%) |
| lowest scaling_max MHz 0/4/6 | 1401/2348/2630 | 930/1491/500 | 1328/2253/1745 | 1098/1491/500 |
| gate wait s | 0 | 253 | 71 | 405 |
| stop limit | – | – | – | – |

## Register

One row per archive folder or run with temperature readings or heat conclusions. Paths are relative to
`benchmark/` unless they start with `/`. Phone paths: `/data/data/com.termux/files/home/` (= `/termux-home/` in
Debian).

| Path | Date | What it measured | Sensor | Status | Reason |
|---|---|---|---|---|---|
| `thermal_char/runs/run_20260930T233208Z/` | 2026-09-30 | heating, throttling and slowdown under 20 min robot-like load, 5 min cooldown | VIRTUAL-SKIN, Android status, cpufreq caps (zones also logged) | **VALID** | heat state judged from skin, status and caps |
| `thermal_char/runs/run_20260930T231117Z_smoke/` | 2026-09-30 | the same, 60 s load | same | **VALID AS FUNCTIONAL CHECK ONLY** | smoke run, not a measurement |
| `thermal_char/runs/thermal.log`, `thermal_char/runs/oneshot_console_*.log` (launcher lines) | 2026-09-30 | launcher's 5 s zone log and idle readings | zone9/10/11 | **NOT TRUSTED** | CPU-core readings; not used as a heat state by the VALID run |
| `coresidency/runs/run_20261001T041743Z/` | 2026-10-01 | camera + detector + Gemma co-residency: speed, RAM, power, skin, status, caps | VIRTUAL-SKIN, Android status, cpufreq caps (zone9 row and gate also recorded) | **VALID** | heat state judged from skin, status and caps; its `zone9` row is a CPU-core reading, not a heat state |
| `coresidency/runs/run_20261001T040522Z_smoke/` | 2026-10-01 | the same, 20 s blocks, no gate wait | same | **VALID AS FUNCTIONAL CHECK ONLY** | smoke run, not a measurement |
| `coresidency/runs/thermal.log` | 2026-09-30/10-01 | launcher's 5 s zone log over all four co-residency sessions | zone9/10/11 | **NOT TRUSTED** | CPU-core readings |
| `/…/home/coresidency/run_20260930T182618Z/` (phone) | 2026-09-30 | co-residency full run, older runner `65c572e0…` | zone9 > 80 °C heat stop | **NOT ARCHIVED** (NOT TRUSTED) | every block ended by the zone9 heat stop after 2.2–155.9 s |
| `/…/home/coresidency/run_20260930T150019Z_smoke/` (phone) | 2026-09-30 | co-residency smoke, runner `ef622037…` | zone9 | **NOT ARCHIVED** (NOT TRUSTED) | superseded; selector count bug, zone9 readings only |
| `camera_heat/runs/camera_heat_20260929T022511Z_16984/` | 2026-09-29 | camera-path power, RAM and zone9 rise per block | zone9 (temperatures); battery sysfs (W); dumpsys meminfo (PSS) | **NOT TRUSTED** (temperatures) / **STILL VALID WITH CAVEAT** (power, RAM) | rise °C/min is from zone9; zone9 gate, no cpufreq record |
| `/…/home/ladder/camera_heat_20260929T021119Z_12612/` (phone) | 2026-09-29 | camera-heat 20 s toy | zone9 | **NOT ARCHIVED** (NOT TRUSTED) | functional toy, zone9 only |
| `/…/home/ladder/camera_heat_20260929T010749Z_10076/` (phone) | 2026-09-29 | aborted camera-heat start | – (empty `sensors.jsonl`) | **NOT ARCHIVED** (NOT TRUSTED) | no data |
| `strategic_selector/ladder/` (`runs/real_run`, `runs/s1o_speed_*`, `runs/s1o_dp_q40_*`, `logs/thermal.log`, oneshot consoles) | 2026-09-25/27 | selector accuracy and decision speed; zone9 start/end per block; zone9 gate | zone9/10/11 | **STILL VALID WITH CAVEAT** (speed, RAM); heat readings **NOT TRUSTED** | zone9 gate and readings; DECISIONS #119 quotes "Zone9 rose 31 → 72 °C" |
| `strategic_selector/von12_rerun/` | 2026-09-24 | Von 1.2 selector accuracy, latency, RSS; z9 peaks 106/104 °C; cooldown gate | zone9/10/11 (`raw/thermal.log`, `raw/cooldown_gate.txt`) | **STILL VALID WITH CAVEAT** (speed, RAM); heat readings **NOT TRUSTED** | z9 peak values in RUN_INDEX; zone9 gate |
| `llm_objective_setting/runs/` (#110 bench results) and `llm_objective_setting/conversation/conversation_quality*/` | 2026-09-19/26 | objective-setting and conversation quality and speed; `zone9_pre`/`zone9_post` per call; zone9 cooldown | zone9 | **STILL VALID WITH CAVEAT** (speed, RAM); heat conclusion **NOT TRUSTED** | README and DECISIONS #110 entry call the phone "hot" / speeds "thermally contaminated" |
| `llm_objective_setting/conversation/conversation_20260926T185957Z/` and `conversation/prep/` | 2026-09-26 | conversation speed per model; z9/z10/z11 start → end per block; ladder zone9 gate | zone9/10/11 | **STILL VALID WITH CAVEAT** (speed, RAM); heat readings **NOT TRUSTED** | zone9 gate and readings |
| `llm_objective_setting/memory_check/` | 2026-09-26 | llama-server memory growth; z9/z10/z11 at turn 0 and around the timeout checks | zone9/10/11 | **STILL VALID WITH CAVEAT** (RAM); heat readings **NOT TRUSTED** | zone readings recorded beside RAM results |
| `llm_objective_setting/conversation/conv_mtp/` | 2026-09-28 | MTP generation speed; z9 start/end per block; ladder zone9 gate | zone9/10/11 | **STILL VALID WITH CAVEAT** (speed, RAM); heat conclusion **NOT TRUSTED** | DECISIONS #120 "about 100 °C … thermally confounded" and #124 "heat reference … cooldown 1.5–4 min" come from zone9 |
| `yolo_speed/` | 2026-09-28 | YOLO detector speed, RSS; z9 start → end per config; zone9 gate | zone9 | **STILL VALID WITH CAVEAT** (speed, RAM); heat conclusion **NOT TRUSTED** | DECISIONS #121 "raised zone9 to 75–102 °C within 20–60 s" is zone9 |
| `llama_dotprod_rebuild/` | 2026-09-26 | build speed checks; `"thermal": null`, no gate | – | **NOT A HEAT TEST** | proot could not read zones; "thermal" only in passing |
| `qwen_vision/` | 2026-09-14/16 | harness only, no results archived; fixed 60 s rest | – | **NOT A HEAT TEST** | "cooldown" is a fixed rest, explicitly not thermal equilibrium |
| `strategic_selector/` (v1/v2 root, `results/`) | 2026-09-23 | selector accuracy | – (helper code can sample zones; no readings in the archived results) | **NOT A HEAT TEST** | no temperature readings archived |
| `strategic_selector/v3/` | 2026-09-24/26 | selector harness v3 | – | **NOT A HEAT TEST** | "thermal state varies" in passing |
| `failclosed_108/` | 2026-09-14 | fail-closed verification output | – | **NOT A HEAT TEST** | cycle header prints `0°C` (no reading) |

No hits: `strategic_selector/candidates_phase0/`, `strategic_selector/manual/jevlike/`, `robotcam/`.

Other hits, not archives and not rows: `benchmark/INDEX.md` (camera_heat row quotes power only; it still says
"results not yet archived"; not edited here); tool code in `coresidency/`, `thermal_char/`, `camera_heat/`; root
scripts `main.py` (live pause on zone9 > 80 °C), `get_temp.py`, `thermal_guard.py`, `thermal_real.py`,
`thermal_benchmark*.py`, `system_benchmark.py`, `full_benchmark.py`, `build_bench.py`, `yolo_benchmark.py`, `chat.py`,
`mission_test.py` (code, no archived readings); binary false positives in `*.onnx`, `test_photos/*.jpg`,
`bench_photos/*.jpg`, `yolo_speed/frames/*.jpg`, `__pycache__/`.

## Banners added

Each README below got one blockquote line at the top (nothing else changed, except the item 5a lines in the ladder
README). Their folders' ARTIFACTS files still list the pre-banner README hash; they were not edited.

| README | SHA-256 before | after |
|---|---|---|
| `strategic_selector/von12_rerun/README.md` | `edf7d044a86a…` | `ace8731cd27d…` |
| `strategic_selector/ladder/README.md` | `d6a76e47fb21…` | `99f9e3e1e7b5…` |
| `llm_objective_setting/README.md` | `e2861e168a57…` | `f4e7f0c2d002…` |
| `llm_objective_setting/conversation/conv_mtp/README.md` | `ad14e65db0cc…` | `431a2698c31b…` |
| `yolo_speed/README.md` | `215c32f4019c…` | `028b8b2fe2d9…` |

## Zone9-based heat numbers and conclusions quoted in docs/*.md

List only; the docs were not edited. Quotes collapse repeated spaces and line breaks.

| File:line | Quote |
|---|---|
| `docs/APP_STATUS.md:74` | "zone9 peak under load 82 C, median 66 C" |
| `docs/APP_STATUS.md:75` | "throttle onset NOT REACHED — no throttle collapse observed" |
| `docs/COMMANDS.md:438` | "Historical idle readings of 52–67 °C do not establish the loaded ceiling." |
| `docs/COMMANDS.md:445` | "the 24-cycle dry run peaked at 67 °C with YOLO" |
| `docs/COMMANDS.md:446` | "#74 records 97–101 °C under sustained llama.cpp inference and 31–38 °C idle" |
| `docs/COMMANDS.md:447` | "superseding the older unsourced 79–82 °C sustained-inference claim" |
| `docs/DECISIONS.md:99` | "BIG-core thermal zone (zone9) is the thermal signal" |
| `docs/DECISIONS.md:101` | "and zone9 both track the real throttle" |
| `docs/DECISIONS.md:346` | "zone9 peaked at 82 °C, median 66 °C" |
| `docs/DECISIONS.md:407` | "Loaded BIG-core temperature measured for the first time: 97-101 °C sustained" |
| `docs/DECISIONS.md:411-412` | "runs in thermal equilibrium exactly at its designed passive throttle point" |
| `docs/DECISIONS.md:413-414` | "20 °C of margin remains above that" |
| `docs/DECISIONS.md:416` | "why the phone feels cool at 100 °C junction" |
| `docs/DECISIONS.md:417` | "so it swings ~60 °C within a single inference burst" |
| `docs/DECISIONS.md:419` | "Supersedes STATUS's unsourced 79-82 °C figure." |
| `docs/DECISIONS.md:424-425` | "Against a measured 97-101 °C operating band that pause fires" |
| `docs/DECISIONS.md:616-617` | "#74's measurement (97-101 °C sustained, at zone9's kernel" … "stands unchanged" |
| `docs/DECISIONS.md:870-871` | "Speed figures are thermally contaminated (back-to-back runs on a hot phone)" |
| `docs/DECISIONS.md:1120` | "Zone9 rose 31 → 72 °C during the block." |
| `docs/DECISIONS.md:1149` | "Gemma and Qwen Q4_K_M blocks reached about 100 °C" |
| `docs/DECISIONS.md:1150` | "Q4_K_M/Q4_0 comparison is thermally confounded" |
| `docs/DECISIONS.md:1172` | "continuous detection raised zone9 to 75–102 °C within 20–60 s" |
| `docs/DECISIONS.md:1217` | "2.30 W and zone9 +1.4 °C/min; RobotCam 2/s 2.29 W and +0.3 °C/min" |
| `docs/DECISIONS.md:1218` | "old path 3.30 W and +3.1 °C/min" |
| `docs/DECISIONS.md:1254` | "13 Gemma conversation turns raised zone9 27 → about 100 °C in 56 s" |
| `docs/DECISIONS.md:1255` | "cooldown to idle+4 °C took 1.5–4 min" |
| `docs/PROTOTYPE_EVIDENCE.md:50` | "BIG 31.0-40.0 °C (swings ~9 °C, meaningless)" |
| `docs/PROTOTYPE_EVIDENCE.md:54` | "`main.py: get_temp()` field-checked: one live call returned 41000 -> 41 °C" |
| `docs/STATUS.md:155` | "free RAM ~3.1 GB, zone9 97-101 °C" |
| `docs/STATUS.md:174` | "zone9 rise +1.4 °C/min vs" |
| `docs/STATUS.md:175` | "+3.1 °C/min (DECISIONS #122). Continuous YOLO reached 75–102 °C within a" |
| `docs/STATUS.md:176` | "Gemma conversation reached about 100 °C within a minute" |
| `docs/STATUS.md:325` | "`get_temp()` pause threshold >80 °C is far below the operating band." |
| `docs/STATUS.md:326` | "Measured 97-101 °C sustained under inference, 31-38 °C idle" |
| `docs/STATUS.md:327` | "so the chip runs in equilibrium at its designed" |
| `docs/STATUS.md:328` | "The >80 °C pause would fire on nearly every check." |

zone9 thresholds quoted (code settings, not measurements): `docs/STATUS.md:93` "thermal pause >80 °C reads
thermal_zone9 (BIG cores)"; `docs/STATUS.md:168` "`main.py` pause still reads BIG above 80 °C";
`docs/DECISION_INDEX.md:16` "the live `main.py` pause still reads BIG at >80 °C"; `docs/COMMANDS.md:426`
"warn 82 °C / critical 86 °C"; `docs/COMMANDS.md:441` "pause above **80 °C** is currently live"; `docs/COMMANDS.md:442`
"its **82/86 °C** values are not competing live thresholds"; `docs/DECISIONS.md:622` "main.py's >80 °C pause is unchanged"; `docs/FILES.md:44`
"(WARN 82 / CRIT 86)".

Throttling or thermal-state claims whose sensor the docs do not state: `docs/DECISIONS.md:103` "0s rest → 8.7 tok/s
sustained with throttling"; `docs/DECISIONS.md:450` "taken at different thermal states".

Not listed: kernel trip points (configuration, e.g. `docs/PROTOTYPE_EVIDENCE.md:24-26`), the zone map
(`docs/COMMANDS.md:427`) and process descriptions without a number.

## How the register was searched

From `/termux-home/robot`, before this archive's files were copied in, over the whole working tree (tracked, untracked
and ignored files; `.git` excluded):

```
command grep -rn --binary-files=text -i --exclude-dir=.git -- "$p" . | wc -l
```

| Pattern `$p` | Hit lines |
|---|---:|
| zone9 | 2410 |
| zone10 | 37 |
| zone11 | 36 |
| thermal_zone | 124 |
| thermal | 3930 |
| degC | 2396 |
| °C | 252 |
| cooldown | 74 |
| cool-down | 0 |
| heat | 94 |

297 files matched at least one pattern; each was assigned to a row above or to "Other hits". The docs were searched
with the same patterns plus `z9`, `hot`, `throttl` and bare `NN C`. The phone-only folders (`/…/home/coresidency/`,
`/…/home/thermal_char/`, `/…/home/ladder/camera_heat_*`) were read directly.
