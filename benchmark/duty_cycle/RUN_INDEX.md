# duty_cycle — run index

One row per run/console attempt. UTC console launch time and run.json start time are separate; failed starts retain metadata without measurement data.

| Folder / console | Date UTC | Set / blocks | Label |
|---|---|---|---|
| `runs/run_20261002T164148Z_PARTS_smoke/`; `runs/oneshot_console_20261002T163641Z.log` | 2026-10-02 16:36:41Z (console); 2026-10-02T16:41:49+00:00 (run.json) | P0U, LOAD, P0, CAM, YOLO, YOLO_NOSPIN, SEL | PARTS smoke (console 20261002T163641Z): SMOKE, not a measurement. |
| `runs/run_20261002T172729Z_CYCLE_smoke/`; `runs/oneshot_console_20261002T172221Z.log` | 2026-10-02 17:22:21Z (console); 2026-10-02T17:27:30+00:00 (run.json) | CONT, CYC50, CYC25, HEATCOOL | CYCLE smoke attempt (console 20261002T172221Z): FAILED START, cores lost before first block (phone in pocket); no data. |
| `runs/run_20261002T190111Z_CYCLE_smoke/`; `runs/oneshot_console_20261002T185603Z.log` | 2026-10-02 18:56:03Z (console); 2026-10-02T19:01:12+00:00 (run.json) | CONT, CYC50, CYC25, HEATCOOL | CYCLE smoke run_20261002T190111Z_CYCLE_smoke (console 20261002T185603Z): SMOKE; its INCOMPLETE flag comes from the truncated 5 s final active phase in 20 s smoke blocks. |
| `runs/run_20261002T201443Z_PARTS/`; `runs/oneshot_console_20261002T200935Z.log` | 2026-10-02 20:09:35Z (console); 2026-10-02T20:14:44+00:00 (run.json) | P0U, LOAD, P0, CAM, YOLO, YOLO_NOSPIN, SEL | PARTS run_20261002T201443Z_PARTS (console 20261002T200935Z): VALID. LOAD paused ~62 min by core loss, discarded and redone; YOLO_NOSPIN and SEL warm starts (heat slopes not comparable; power fine). |
| `runs/run_20261003T090742Z_CYCLE/`; `runs/oneshot_console_20261003T090234Z.log` | 2026-10-03 09:02:34Z (console); 2026-10-03T09:07:43+00:00 (run.json) | CONT, CYC50, CYC25, HEATCOOL | CYCLE run_20261003T090742Z_CYCLE (console 20261003T090234Z): VALID, no INCOMPLETE. CYC50, CYC25, HEATCOOL warm starts: between-block heat slopes not comparable; cooling curve and power valid. Skin stayed ~33 C during the 7 min gate wait before CYC25 (cause unknown). Active power 3.5 W vs CONFIRM 2.84 W with the same settings (unexplained). |
| —; `runs/oneshot_console_20261003T154437Z.log` | 2026-10-03 15:44:37Z (console) | none; no data | console 20261003T154437Z: OPERATOR TYPO (extra argument), no data. |

## Fixed labels (copied exactly)

- PARTS smoke (console 20261002T163641Z): SMOKE, not a measurement.
- CYCLE smoke attempt (console 20261002T172221Z): FAILED START, cores lost before first block (phone in pocket); no data.
- CYCLE smoke run_20261002T190111Z_CYCLE_smoke (console 20261002T185603Z): SMOKE; its INCOMPLETE flag comes from the truncated 5 s final active phase in 20 s smoke blocks.
- PARTS run_20261002T201443Z_PARTS (console 20261002T200935Z): VALID. LOAD paused ~62 min by core loss, discarded and redone; YOLO_NOSPIN and SEL warm starts (heat slopes not comparable; power fine).
- CYCLE run_20261003T090742Z_CYCLE (console 20261003T090234Z): VALID, no INCOMPLETE. CYC50, CYC25, HEATCOOL warm starts: between-block heat slopes not comparable; cooling curve and power valid. Skin stayed ~33 C during the 7 min gate wait before CYC25 (cause unknown). Active power 3.5 W vs CONFIRM 2.84 W with the same settings (unexplained).
- console 20261003T154437Z: OPERATOR TYPO (extra argument), no data.

## Console matches

- oneshot_console_20261002T163641Z.log: explicitly names run_20261002T164148Z_PARTS_smoke; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261002T172221Z.log: only failed-start run at 17:27:29Z after the 17:22:21Z launch/5 min idle; traceback is after post-warm-up wait, and folder contains run.json/warmup log only, no blocks or measurement report.
- oneshot_console_20261002T185603Z.log: explicitly names run_20261002T190111Z_CYCLE_smoke; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261002T200935Z.log: explicitly names run_20261002T201443Z_PARTS; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261003T090234Z.log: explicitly names run_20261003T090742Z_CYCLE; run timestamp follows console launch and 5 min idle.

No ambiguous match was filled by guessing. `thermal.log` spans sessions and is not a per-run console.
