# camera_power — run index

One row per run/console attempt. UTC console launch time and run.json start time are separate; failed starts retain metadata without measurement data.

| Folder / console | Date UTC | Set / blocks | Label |
|---|---|---|---|
| `runs/run_20261003T004418Z_smoke/`; `runs/oneshot_console_20261003T003910Z.log` | 2026-10-03 00:39:10Z (console); 2026-10-03T00:44:18+00:00 (run.json) | baseline_first, preview, record, fast, off, focus_1m, manual_200, manual_500, mode_A, dump, baseline_last | smoke console 20261003T003910Z: SMOKE, dark room (luma ~4), CAMPOWER1 build; not representative. |
| `runs/run_20261003T062831Z_smoke/`; `runs/oneshot_console_20261003T062323Z.log` | 2026-10-03 06:23:23Z (console); 2026-10-03T06:28:32+00:00 (run.json) | baseline_first, preview, record, fast, off, focus_1m, manual_200, manual_500, mode_A, dump, baseline_last | smoke console 20261003T062323Z: SMOKE, lit room, CAMPOWER1 build. |
| `runs/run_20261003T073546Z_smoke/`; `runs/oneshot_console_20261003T073038Z.log` | 2026-10-03 07:30:38Z (console); 2026-10-03T07:35:47+00:00 (run.json) | baseline_first, manual_200, manual_500, manual_1000, mode_A_manual_1000, mode_A_manual_500, mode_A, baseline_last | smoke console 20261003T073038Z: SMOKE, lit, CAMPOWER2 build; manual_1000 / mode_A_manual_1000 INCOMPLETE (1 fps at its edge). |
| `runs/run_20261003T080207Z/`; `runs/oneshot_console_20261003T075659Z.log` | 2026-10-03 07:56:59Z (console); 2026-10-03T08:02:07+00:00 (run.json) | baseline_first, manual_200, manual_500, mode_A_manual_500, mode_A, baseline_last | full console 20261003T075659Z: VALID except manual_500 INCOMPLETE (one gap 0.01 s over tolerance); CAMPOWER2 build 90ef439. Runner 10 Hz polling adds CPU load equally to all blocks; watts not comparable to DUTY1. |
| —; `runs/oneshot_console_20261003T073018Z.log` | 2026-10-03 07:30:18Z (console) | none; no data | LAUNCH REFUSED: battery Charging, not Discharging; no run folder or measurement data (console text). |
| —; `runs/oneshot_console_20261003T075652Z.log` | 2026-10-03 07:56:52Z (console) | none; no data | LAUNCH REFUSED: battery Charging, not Discharging; no run folder or measurement data (console text). |

## Fixed labels (copied exactly)

- smoke console 20261003T003910Z: SMOKE, dark room (luma ~4), CAMPOWER1 build; not representative.
- smoke console 20261003T062323Z: SMOKE, lit room, CAMPOWER1 build.
- smoke console 20261003T073038Z: SMOKE, lit, CAMPOWER2 build; manual_1000 / mode_A_manual_1000 INCOMPLETE (1 fps at its edge).
- full console 20261003T075659Z: VALID except manual_500 INCOMPLETE (one gap 0.01 s over tolerance); CAMPOWER2 build 90ef439. Runner 10 Hz polling adds CPU load equally to all blocks; watts not comparable to DUTY1.

## Console matches

- oneshot_console_20261003T003910Z.log: post-idle timestamp 2026-10-03T00:44:17Z matches run.json idle; run folder is the next second; block count/order and CAMPOWER version agree.
- oneshot_console_20261003T062323Z.log: post-idle timestamp 2026-10-03T06:28:30Z matches run.json idle; run folder is the next second; block count/order and CAMPOWER version agree.
- oneshot_console_20261003T073038Z.log: post-idle timestamp 2026-10-03T07:35:45Z matches run.json idle; run folder is the next second; block count/order and CAMPOWER version agree.
- oneshot_console_20261003T075659Z.log: post-idle timestamp 2026-10-03T08:02:06Z matches run.json idle; run folder is the next second; block count/order and CAMPOWER version agree.

No ambiguous match was filled by guessing. `thermal.log` spans sessions and is not a per-run console.
