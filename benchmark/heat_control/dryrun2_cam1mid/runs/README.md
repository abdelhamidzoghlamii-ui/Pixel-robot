# ARCHIVE-2 — runs

Motors-off `run_mission.py --dry`, code e208aaf (CAM1-MID), owner-run, native Termux, 2026-10-15 10:56–16:11Z, 5.26 h, stopped by Ctrl-C. 9,293 ACTIVE cycles; 29 heat pauses; 0 rule violations; Android status 0 throughout; skin 34.76–37.07 °C after the first pause; first pause after 5,099 s; later ACTIVE slots median 376 s; ACTIVE duty 0.838 from the first pause. 0 camera stops; max frame age 1.31 s; slot-median detect 139–148 ms (320) and 518–554 ms (640); cycle period 1.80 s. Detector affinity drifted to cpus 0–7 in slot 17 and was re-pinned OK at the next camera start. VALID for CAM1-MID as one motors-off run; no ultrasonic (dry: 999 cm); not a motors-live result.

Fixed labels: Local AI 2026-10-17; L2 probability phrase corrected by Local AI, with owner authorization, on 2026-10-17. Source files are unchanged byte copies. Script files here are historical evidence, not entry points.

[Run index](RUN_INDEX.md) · [Artifacts](ARTIFACTS.md).

“0 camera stops” counts unexpected frame-read stops; planned heat camera-off pauses are counted separately. Minimum-bound pauses start on the first mission cycle after the 60 s floor (console rounds these to 60–62 s). Rule replay uses the original commit’s pure decide(), the latest pre-camera dumpsys row, and elapsed seconds in the original console; L2’s unlogged final restart uses its qualifying dump. The 5 s trace cannot independently reconstruct all 1 s CPU streak/staleness checks. Owner observations and Ctrl-C cause are owner testimony.
