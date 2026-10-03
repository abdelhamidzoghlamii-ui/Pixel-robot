# DUTY1 — component and duty-cycle power archive

Research benchmark with no robot motion. PARTS separates unloaded idle, Gemma load, resident idle, camera, YOLO, YOLO with ORT spinning disabled, and selector-only work. CYCLE compares continuous activity, 50 s active/10 s pause, 25 s active/5 s pause, and one heat/cool curve. Active settings are RobotCam 1/s, 320 with 640 instead every 5 s, MID threads, and selector calls every 20 s of planned active time. It records phase/whole-cycle battery power and energy, load time/cache residency, camera transitions, inference, skin, status and caps.

Human runs only, native Termux, motors off, unplugged, screen on and Termux in front; exit all agents first. [RUN.md](RUN.md) gives commands and prerequisites. Run separately: `bash ~/robot/benchmark/duty_cycle/oneshot.sh --set PARTS --smoke`, `--set CYCLE --smoke`, `--set PARTS`, `--set CYCLE`. No hardware execution is part of ARCHIVE3.

Recorded results and limits: the full PARTS run is valid after discarding and redoing LOAD following about 62 min of core loss. Full CYCLE has no INCOMPLETE; warm starts prevent between-block heat-slope comparisons, while cooling curve and power remain valid. Skin stayed near 33 C in the 7 min gate wait before CYC25 for an unknown reason. Active power about 3.5 W differs from CONFIRM 2.84 W with the same settings, unexplained. The short CYCLE smoke INCOMPLETE reflects its truncated final active phase.

Archive prepared by ARCHIVE3 without staging, commit or push. Source code is unchanged; report hashes were checked before writing archive documentation. Run files and reports were copied byte for byte. `runs/thermal.log` covers multiple sessions and is not assigned to a single run. Zone9/10/11 are CPU-core readings, not a heat state. Skin, Android status and cpufreq caps carry the heat evidence. Saved JPEGs are owner-approved public benchmark photos.

See [RUN_INDEX.md](RUN_INDEX.md) for fixed validity labels, run/log matches and block lists; [ARTIFACTS.md](ARTIFACTS.md) for sizes, hashes, sources and exclusions. Independent ARCHIVE3 review is retained verbatim in the Downloads coder report; it checks archival completeness and labels, not new code behaviour. Historical coder/reviewer reports in `reports/` retain their original verdicts and limitations.

## Fixed run labels (verbatim task text)

- PARTS smoke (console 20261002T163641Z): SMOKE, not a measurement.
- CYCLE smoke attempt (console 20261002T172221Z): FAILED START, cores lost before first block (phone in pocket); no data.
- CYCLE smoke run_20261002T190111Z_CYCLE_smoke (console 20261002T185603Z): SMOKE; its INCOMPLETE flag comes from the truncated 5 s final active phase in 20 s smoke blocks.
- PARTS run_20261002T201443Z_PARTS (console 20261002T200935Z): VALID. LOAD paused ~62 min by core loss, discarded and redone; YOLO_NOSPIN and SEL warm starts (heat slopes not comparable; power fine).
- CYCLE run_20261003T090742Z_CYCLE (console 20261003T090234Z): VALID, no INCOMPLETE. CYC50, CYC25, HEATCOOL warm starts: between-block heat slopes not comparable; cooling curve and power valid. Skin stayed ~33 C during the 7 min gate wait before CYC25 (cause unknown). Active power 3.5 W vs CONFIRM 2.84 W with the same settings (unexplained).
- console 20261003T154437Z: OPERATOR TYPO (extra argument), no data.
