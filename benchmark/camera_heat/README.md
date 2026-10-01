# Camera path heat, power and RAM (2026-09-29)

> **Temperature data in this archive is NOT TRUSTED (zone9).** The `temp_*` and `rise_c_per_min` values come from
> zone9, a CPU-core sensor, so they say nothing about how hot the phone got. Power and RAM results are **STILL VALID
> WITH CAVEAT**: valid as measured, but the cooldown gate used zone9 and cpufreq was not recorded, so whether the CPU
> was capped is unknown. See [../thermal_char/HEAT_EVIDENCE.md](../thermal_char/HEAT_EVIDENCE.md).

Research only: motors off. Relates to DECISIONS #122.

## What and how

`camera_heat.py` (SHA-256 `63275fa0…`) with `run_camera_heat.sh` (`4058b710…`) compares the camera paths in four
180 s blocks, in order:

| Block | What |
|---|---|
| idle | nothing running |
| robotcam_1 | RobotCam at 1 frame/s, read by `robotcam_test.py` every 1.0 s |
| robotcam_2 | RobotCam at 2 frames/s, read every 0.5 s |
| old_path | the old `termux-camera-photo` capture loop |

A native Termux helper (root) samples every 5 s zone9/10/11, battery `current_now`/`voltage_now` and charging status,
and takes one `dumpsys meminfo` of the RobotCam app and the camera provider in the middle of blocks 2–4. Before each
block the runner waits for zone9 ≤ idle + 4 °C. W = −(current_now × voltage_now).

The run was started from native Termux with the helper and `~/ladder/oneshot.sh` (SHA-256 `db1f3156…`, byte-identical
to `../strategic_selector/ladder/oneshot_executed_conversation_run.sh`), 5 min idle first; the command is in
`reports/CODER_REPORT_camera_heat.md`. The run did not record script hashes. The archived scripts have the hashes the
Coder report gives for the reviewed candidate, and the phone's `~/ladder/run_camera_heat.sh` has the same hash, so
they are most likely the executed versions (inferred).

## Results: `camera_heat_20260929T022511Z_16984` (copied from `runs/camera_heat_20260929T022511Z_16984/results.json`)

| Block | Mean battery W | Frames ok / failed | RobotCam app PSS kB | Camera provider PSS kB | Status |
|---|---:|---|---:|---:|---|
| idle | 0.712 | – | – | – | Discharging |
| robotcam_1 | 2.301 | 179 / 2 (181 attempts) | 35703 | 273696 | Discharging |
| robotcam_2 | 2.286 | 358 / 2 (360 attempts) | 39411 | 275247 | Discharging |
| old_path | 3.302 | 102 / 11 | 24391 | 270539 | Discharging |

Zone9 values, **NOT TRUSTED** (kept only as recorded): idle 32 → 31 °C (−0.34 °C/min); robotcam_1 34 → 38 (+1.39);
robotcam_2 39 → 40 (+0.34); old_path 39 → 48 (+3.14). Idle zone9 at the start was 33 °C.

## Known limitations

- **Temperatures are zone9 (NOT TRUSTED).** No VIRTUAL-SKIN, Android thermal status or cpufreq was recorded, so the
  run has no valid heat evidence and CPU capping during the blocks is unknown.
- Review: four rounds with a fresh Claude Code reviewer; round 4 was **INCOMPLETE** (no code blockers, but every
  hardware path unverified, because the toy could not run in proot). The 20 s toy then ran on the phone
  (`camera_heat_20260929T021119Z_12612`, not archived) before this run. The run results were not reviewed.
- The sign of `current_now` was not verified (negative = discharge assumed); the raw µA and µV are in
  `sensors.jsonl`.
- RobotCam warm-up reads count as failed reads.
- The mid-block `dumpsys meminfo` adds load to blocks 2–4 only.
- In old_path RobotCam is stopped, so its app PSS was expected to be null; the run recorded 24391 kB (the app process
  was still present).
- The raw `dumpsys meminfo` files stay in the phone's `/data/local/tmp/` (root only), not archived:
  `camera_heat_20260929T022511Z_16984_app.txt` and `camera_heat_20260929T022511Z_16984_provider_1048.txt`.
- The launcher's screen-timeout restore failed at the end ("Failure calling service settings", see the console);
  this is the bug fixed by DECISIONS #123 in `../coresidency/oneshot.sh`.
- One run, on battery.

## Contents

| Path | What |
|---|---|
| `camera_heat.py`, `run_camera_heat.sh` | runner/helper and its wrapper |
| `runs/camera_heat_20260929T022511Z_16984/` | `results.json`, `sensors.jsonl` (5 s), RobotCam reader outputs, helper request/response files |
| `runs/oneshot_console_20260929T022511Z.log` | launcher console |
| `reports/CODER_REPORT_camera_heat.md` | the Coder report with the final review, verbatim |

[RUN_INDEX.md](RUN_INDEX.md) lists the runs, including the two not archived; [ARTIFACTS.md](ARTIFACTS.md) hashes
every file.
