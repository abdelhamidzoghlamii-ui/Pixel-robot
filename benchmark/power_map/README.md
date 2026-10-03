# POWERMAP1 — YOLO power map archive

Research benchmark for RobotCam, YOLO11s and resident Gemma selector calls, with no robot motion. It varies frame rate (R1: 2/s, R2: 1/s), 640 cadence (5 s, 10 s, off), and ORT cores/threads (R3: default, MID, LITTLE); CONFIRM holds the chosen configuration for 20 min. It records battery watts, detector latency, selector results, worker affinity, per-policy capped time, skin and Android status.

Human runs only, native Termux, motors off, unplugged, screen on and Termux in front; exit all agents first. Commands and prerequisites are in [RUN.md](RUN.md). Examples: `bash ~/robot/benchmark/power_map/oneshot.sh --set R1 --smoke`, then separate full R1/R2/R3 runs. Confirmation used `bash ~/robot/benchmark/power_map/oneshot.sh --set CONFIRM --rate 1 --cadence 5 --threads mid`. Do not run benchmarks during archival/review.

Key results from the recorded runs: 1 frame/s about halves power relative to 2/s; 640 cadence was not a useful power lever. MID was chosen and LITTLE missed frames (149/180). Confirmation reached policy4/6 caps near minute 17 at skin about 37 C, with skin still rising. Warm starts limit heat comparisons. Run-to-run watts differ by about 0.7 W; absolute watts still contradict #122 without explanation. These are fixed task labels and recorded results, not new conclusions.

Archive prepared by ARCHIVE3 without staging, commit or push. Source code is unchanged; report hashes were checked before writing archive documentation. Run files and reports were copied byte for byte. `runs/thermal.log` covers multiple sessions and is not assigned to a single run. Zone9/10/11 are CPU-core readings, not a heat state. Skin, Android status and cpufreq caps carry the heat evidence. Saved JPEGs are owner-approved public benchmark photos.

See [RUN_INDEX.md](RUN_INDEX.md) for fixed validity labels, run/log matches and block lists; [ARTIFACTS.md](ARTIFACTS.md) for sizes, hashes, sources and exclusions. Independent ARCHIVE3 review is retained verbatim in the Downloads coder report; it checks archival completeness and labels, not new code behaviour. Historical coder/reviewer reports in `reports/` retain their original verdicts and limitations.

## Fixed run labels (verbatim task text)

- run_20261001T213425Z_R1_smoke, run_20261001T215543Z_R3_smoke: SMOKE, not a measurement; pinning verified.
- run_20261001T222645Z_R1: VALID. Blocks 2-3 started warm (heat comparisons approximate); 640 cadence not a lever.
- run_20261001T231621Z_R2: VALID. 1 frame/s about halves power vs 2 frames/s.
- run_20261002T083606Z_R3: VALID. MID chosen; LITTLE out (149/180 frames). Default block 2.83 W vs 2.10 W in R2: run-to-run noise ~0.7 W, first-block bias suspected.
- run_20261002T090821Z_CONFIRM: VALID. 20 min; policy4/6 capping starts ~min 17 at skin ~37 C; skin still rising.
- Open across power_map: absolute watts contradict #122 (unexplained).
