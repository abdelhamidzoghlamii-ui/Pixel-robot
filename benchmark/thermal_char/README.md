# Thermal characterization (2026-09-30)

Status: run `run_20260930T233208Z` is **VALID** heat evidence: heat state is judged from VIRTUAL-SKIN, Android
thermal status and cpufreq caps. The smoke run `run_20260930T231117Z_smoke` is **VALID AS FUNCTIONAL CHECK ONLY**,
not a measurement. Both runs end with an INCOMPLETE line for a RobotCam process left after the load stop (see
limitations). The zone9/10/11 (BIG/MID/LITTLE) values they record are CPU-core readings, not a heat state.
[HEAT_EVIDENCE.md](HEAT_EVIDENCE.md) is the register of every heat reading in the repository and its label.
Research only: motors off.

## What and how

The runner loads the phone the way the robot does and records how the phone heats and throttles:

- **Load:** RobotCam mode B at 2 frames/s, yolo11s at 640 on every new frame (both detector sessions resident), and
  llama-server (Gemma E2B Q4_0, `setup_q4` flags) streaming back-to-back `/completion` requests (one fixed prompt,
  n_predict 256, temperature 0). The full run planned 1200 s of load and 300 s of cooldown logging; the smoke run
  60 s and 30 s.
- **Readings:** every 1 s from one root shell: every thermal zone, `scaling_cur_freq`/`scaling_max_freq` per cpufreq
  policy, every cooling device `cur_state`, battery temperature and W. Every 5 s: `dumpsys thermalservice` (Android
  thermal status, VIRTUAL-SKIN and the other HAL temperatures; DECISIONS #123 form).
- **Stops (first of):** VIRTUAL-SKIN ≥ 48.0 °C, battery ≥ 45.0 °C, Android status ≥ 5, BIG/MID/LITTLE max ≥ 110 °C in
  3 consecutive samples, or the planned load time. Both runs stopped at the planned load time.

Both runs were started from native Termux with `oneshot.sh` (5 min idle) and the note "room 22 C, phone in robot
mount".

### Executed runner versions

Both runs' `run.json` record `thermal_char.py` `cbf73d12…` and `coresidency.py` `65c572e0…` (imported for RobotCam,
server and core helpers). Neither is the current file in this folder (`thermal_char.py` is now `b5b0c812…` and
`../coresidency/coresidency.py` is `85a6e4cb…`, both changed by CLEANUP1). Exact copies were found by SHA-256 in an
earlier Coder session's frozen review copy and are archived as:

| File | SHA-256 | Source |
|---|---|---|
| `thermal_char_executed_cbf73d12b5bc.py` | `cbf73d12b5bcebb80db1d3c09f5b1e72f0231df915c490669c925f1bd9365d1d` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/22fa5409-6f42-426e-899a-74805d4e9c0c/scratchpad/frozen/thermal_char/thermal_char.py` |
| `coresidency_executed_65c572e01826.py` | `65c572e01826525e5f26881e9d5167145beb430a07e41f6ff15b37c58d1cf60d` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/22fa5409-6f42-426e-899a-74805d4e9c0c/scratchpad/frozen/coresidency/coresidency.py` |

`cbf73d12…` is the THERMAL_CHAR FIX 1 version (`reports/THERMAL_CHAR_FIX1_REPORT.md`, review **APPROVE WITH NOTES**).
`65c572e0…` is the CORESIDENCY FIX 1 version (round 3 **REQUEST CHANGES**, open finding about smoke runs of the
co-residency runner, not about the helpers thermal_char uses). The executed `oneshot.sh` was not recorded by the
runs; the current `oneshot.sh` here is `16787b06…` (unchanged since THERMAL_CHAR FIX 1).

## Results: `run_20260930T233208Z` (copied from `runs/run_20260930T233208Z/report.txt`)

- Stop: planned load duration 1200 s reached at load +1200.1 s. 305 thermalservice dumps (all with skin and status),
  1524 one-second samples, largest sample gap 1.1 s, 0 failed RobotCam reads.
- Layout: policy0 cpus 0–3, cpuinfo_max 1803 MHz; policy4 cpus 4–5, 2348 MHz; policy6 cpus 6–7, 2850 MHz.
- First cap: policy0 `scaling_max_freq` 1803 → 1401 MHz at load +1.8 s; policy6 2850 → 2802 MHz at +3.8 s.
- Android thermal status: 0 at baseline; 0 → 1 (LIGHT) at +77.9 s; 1 → 2 (MODERATE) at +642.9 s; 2 → 3 (SEVERE) at
  +1098.0 s; in cooldown 3 → 2 at +1252.9 s and 2 → 1 at +1327.8 s.
- VIRTUAL-SKIN (30 s timeline): 31.6 °C at load start, 37.0 at +30 s, 40.4 at +120 s, 41.3 at +300 s, 42.9 at
  +600 s, 44.0 at +900 s, 45.4 at the stop (+1200 s); battery 31.9 → 40.9 °C.
- Max frequencies in the second half of the load (timeline): policy0 738–930 MHz, policy4 1024–1197 MHz, policy6
  984 MHz.
- Slowdown: first minute 33 frames, 640 ms median 1811, 5.6 tok/s streamed (5.8 from timings); last minute
  21 frames, 640 ms median 2967, 3.2 tok/s streamed (3.3 from timings).
- Battery W (30 s windows): 10.36 at +30 s, 3.68–4.24 from +300 s to the stop.
- Cooldown: skin started at 31.4 °C; not within 2 °C after 305.7 s (last 38.7 °C).

## Smoke run `run_20260930T231117Z_smoke`

60 s of load, 30 s of cooldown, stopped at the planned 60 s. It shows that the sampler, the dump parser, the load and
the report work on the phone. Its numbers are not measurements.

## Phone thermal configuration evidence (`phone_config/`)

`thermal_info_config.json`, `thermal_info_config_charge.json`, `thermal_info_config_proto.json` and `thermal_zones.txt`
(every `/sys/class/thermal/thermal_zone*` with its type, a temperature and its trip points), copied from
`/sdcard/Download/` (files dated 2026-09-30 18:52). How they were read from the phone was not recorded;
`docs/PROTOTYPE_EVIDENCE.md` names `/vendor/etc/thermal_info_config.json` as the source of the governance it reports.

## Known limitations

- **INCOMPLETE in both runs:** after the load stop, `pidof com.pixelrobot.robotcam` still found a process (17813 in
  the full run, 14507 in the smoke run). The executed runner waited 2 s and then ran `pidof`; CLEANUP1 replaced this
  with `camera_end_check` (capture stopped for ≥ 3 s, then force-stop). The measurement during load is not affected.
- The cooldown section measures time back to within 2 °C of the idle start; 300 s was not enough for skin in the full
  run.
- One full run, one room temperature, phone in the robot mount, on battery.
- GGUF SHA-256 is not recorded (size only); see the reviewer note in `reports/THERMAL_CHAR_CODER_REPORT.md`.
- The review of the original runner (THERMAL_CHAR_CODER_REPORT.md) stopped at round 3 with CHANGES REQUESTED (sample
  time stamped before the read). THERMAL_CHAR FIX 1 fixed that finding, and its single review round returned
  APPROVE WITH NOTES; that is the executed version.
- The current `thermal_char.py` (`b5b0c812…`) has not been run on the phone. CLEANUP1's three open round-3 findings
  (B5 not gated, pid after force-stop recorded only, capped-time convention) concern the co-residency runner; the
  Local AI decisions on them are in `../coresidency/README.md`.

## Contents

| Path | What |
|---|---|
| `thermal_char.py`, `oneshot.sh`, `run_thermal_char.sh`, `test_thermal_char.py`, `test_oneshot.sh`, `RUN.md` | current runner, launcher, wrapper, offline tests, run instructions |
| `thermal_char_executed_cbf73d12b5bc.py`, `coresidency_executed_65c572e01826.py` | the runner and imported module both runs executed |
| `runs/run_20260930T233208Z/`, `runs/run_20260930T231117Z_smoke/` | `report.txt`, `run.json`, `samples_1s.jsonl`, `thermalservice_5s.jsonl`, `thermalservice_raw.jsonl`, `frames.jsonl`, `gen.jsonl`, `llama-server.log` |
| `runs/oneshot_console_*.log` | launcher consoles (they repeat the report) |
| `runs/thermal.log` | the launcher's root zone9/10/11 logger, 5 s, 2026-09-30 23:06Z to 23:57Z; zone readings only, not a heat state |
| `phone_config/` | phone thermal configuration evidence |
| `reports/` | THERMAL_CHAR_CODER_REPORT.md, THERMAL_CHAR_FIX1_REPORT.md, verbatim (CLEANUP1 is in `../coresidency/reports/`) |
| `HEAT_EVIDENCE.md` | the heat register |

[RUN_INDEX.md](RUN_INDEX.md) lists the runs; [ARTIFACTS.md](ARTIFACTS.md) hashes every file.
