# Co-residency benchmark, DECISIONS #124 (2026-10-01)

Status: run `run_20261001T041743Z` is **VALID** heat evidence: heat state is judged from VIRTUAL-SKIN, Android
thermal status and cpufreq caps. Its speed, RAM and power results are valid as measured. The smoke run
`run_20261001T040522Z_smoke` is **VALID AS FUNCTIONAL CHECK ONLY**, not a measurement. The zone9/10/11 values in
both runs are CPU-core readings: they are recorded, but they are not a heat state (see
[../thermal_char/HEAT_EVIDENCE.md](../thermal_char/HEAT_EVIDENCE.md)). Research only: motors off, no `motors.py`.

## What and how

`coresidency.py` (SHA-256 `85a6e4cb…`, recorded in both runs' `run.json`) runs the robot's camera, detector and LLM
together on the phone. All blocks use RobotCam mode B at 2 frames/s, `read_frame` and yolo11s:

| Block | Workload |
|---|---|
| B1_mix_nollm | detector at SizePolicy drive sizes: 320 on each frame, 640 every 5 s |
| B2_only640_nollm | detector always at 640 |
| (loads) | Gemma E2B Q4_0 cold load after a page-cache drop (handshake), then warm load by restart |
| B3_mix_gemma | B1 + resident llama-server (`setup_q4` flags) + a selector letter-scoring call every 20 s |
| B4_only640_gemma | B2 + the same |
| B5_mtp_ram_snapshot | untimed: Gemma with the MTP drafter, one short request, RAM snapshot |

Blocks last 180 s (smoke: 20 s). Before each of B1–B4 and before the loads, the gate waits until VIRTUAL-SKIN ≤ idle
skin + 1.5 °C and zone9 ≤ idle + 4 °C; after 15 min the block starts anyway as a warm start. The smoke run skips the
wait. A block ends early at the first limit: Android status ≥ 4 (CRITICAL), battery ≥ 45.0 °C, or a CPU zone ≥ 110 °C
in 3 consecutive 1 s samples. Readings: every 1 s, CPU zones, battery temperature and `scaling_max_freq` per cpufreq
policy (one root shell); every 5 s, `dumpsys thermalservice` (status, VIRTUAL-SKIN), MemAvailable, swap, PSS and
battery W. After each block's STOP, the runner checks that capture stopped, then force-stops RobotCam and records
`pidof`.

Both runs were started from native Termux with `oneshot.sh` here (SHA-256 `efb3939d…`: no agent running, battery
`Discharging`, screen timeout raised and restored in the DECISIONS #123 form, root thermal logger, 5 min idle,
cache-drop handshake). The commands are in [SMOKE.md](SMOKE.md). Gemma: `gemma-4-E2B-it-Q4_0.gguf` (2841481184 bytes)
on the dotprod b1609 build; MTP drafter `mtp-gemma-4-E2B-it-Q8_0.gguf` (97817664 bytes). Model sizes and server
commands are in `run.json`; weights are not in Git.

## Results: `run_20261001T041743Z` (copied from `runs/run_20261001T041743Z/report.txt`)

Idle skin 31.4 °C. No INCOMPLETE line, no block limit reached, no warm start.

| | B1_mix_nollm | B2_only640_nollm | B3_mix_gemma | B4_only640_gemma |
|---|---:|---:|---:|---:|
| frames processed | 360 | 334 | 350 | 295 |
| failed reads | none | none | none | none |
| detect320 ms median/P95 | 103/140 (n 325) | – | 108/443 (n 315) | – |
| detect640 ms median/P95 | 394/511 (n 35) | 443/715 (n 334) | 376/1338 (n 35) | 460/1356 (n 295) |
| drift640 first/last 30 s | 499 → 376 | 357 → 555 | 501 → 345 | 358 → 566 |
| frame age s median | 0.50 | 0.37 | 0.42 | 0.48 |
| selector ms median/P95 | – | – | 2168/2376 | 2834/4819 |
| selector calls ok/err/correct | 0/0/0 | 0/0/0 | 9/0/9 | 9/0/9 |
| min MemAvailable MiB | 3849 | 3814 | 2532 | 2523 |
| max swap used MiB | 1125 | 1127 | 1063 | 1087 |
| peak PSS llama-server MiB | – | – | 3851 | 3724 |
| LMK lines (kill lines) | 7 (0) | 9 (0) | 6 (0) | 5 (0) |
| survived RobotCam / llama | yes / – | yes / – | yes / yes | yes / yes |
| VIRTUAL-SKIN start/end/max °C | 31.4/37.8/37.8 | 32.8/40.0/40.2 | 32.8/39.4/39.4 | 32.8/40.0/40.3 |
| Android status max | 0 | 1 | 1 | 1 |
| policy capped s (% of block) | 178 (99%) | 179 (99%) | 179 (99%) | 179 (99%) |
| lowest scaling_max MHz 0/4/6 | 1401/2348/2630 | 930/1491/500 | 1328/2253/1745 | 1098/1491/500 |
| gate wait s | 0 | 253 | 71 | 405 |
| mean battery W | 4.26 | 6.26 | 4.08 | 5.77 |

Gemma load: cold 5.44 s (full-cold, page cache dropped by the handshake watcher), warm 3.86 s; gate wait 455 s.
MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 2619 MiB,
swap used 1008 MiB, PSS llama-server 4154 MiB, VmHWM 4211 MiB, load 11.10 s.

"policy capped" is the time with **any** policy below its `cpuinfo_max_freq`; the 1 s samples per policy are in the
block JSONs (`fast`).

## Smoke run `run_20261001T040522Z_smoke`

20 s blocks, no gate wait. Every block processed frames with no failed reads, B3/B4 each had one selector call, the
cold load was full-cold (5.62 s, warm 4.80 s), the MTP snapshot completed, and the report has no INCOMPLETE line. It
shows that the runner and launcher work on the phone; its numbers are not measurements.

## Known limitations (from the review reports in `reports/`)

- **Not approved as code.** CLEANUP1 (`reports/CLEANUP1_REPORT.md`) stopped after round 3 with REQUEST CHANGES, and the
  executed runner `85a6e4cb…` is that round-3 candidate. The three open findings and the Local AI decisions on them:
  - **B5 is not gated** (decision: B5 is a RAM snapshot only, so it needs no cooldown gate).
  - **A pid left after the force-stop is recorded only**, not judged (decision: as the task text says). In both
    archived runs `pids_after_force_stop` is empty after every block and B5.
  - **Capped-time convention:** each in-block 1 s sample counts the time since the previous sample; a gap across a
    skipped sample is credited to the later reading, and the tail after the last sample (< 1 s) is not counted
    (decision: accepted as the convention).
- B5 starts without a gate, so its RAM numbers were taken at whatever temperature B4 left.
- zone9 is still read (gate, `zone9` row, thermal log) but is not a heat signal; the gate's zone9 part has no
  meaning for heat (see HEAT_EVIDENCE.md).
- The gate's idle skin is one reading after a 5 min idle, not a cold-phone baseline.
- Earlier rounds (`reports/CORESIDENCY_CODER_REPORT.md`: 20 rounds, final APPROVE WITH NOTES;
  `reports/CORESIDENCY_FIX1_REPORT.md`: round 3 REQUEST CHANGES) describe older runner versions. The reviewer's sandbox
  failed some of its own shell commands (exit 182) in those rounds; see the reports.
- One full run only. The review artifacts (requests, stdout, stderr per round) are not archived; they stay on the phone
  (see ARTIFACTS.md).

## Contents

| Path | What |
|---|---|
| `coresidency.py`, `oneshot.sh`, `run_coresidency.sh`, `test_coresidency.py`, `test_oneshot.sh`, `SMOKE.md` | runner, launcher, wrapper, offline tests, run instructions |
| `runs/run_20261001T041743Z/` | full run: `report.txt`, `run.json`, `loads.json`, block JSONs, llama-server logs |
| `runs/run_20261001T040522Z_smoke/` | smoke run, same files |
| `runs/oneshot_console_*.log` | the launcher console of each run |
| `runs/thermal.log` | the launcher's root zone9/10/11 logger, 5 s, covering all four co-residency sessions (2026-09-30 14:55Z to 2026-10-01 04:50Z), including the two runs not archived; zone readings only, not a heat state |
| `reports/` | CORESIDENCY_CODER_REPORT.md, CORESIDENCY_FIX1_REPORT.md, CLEANUP1_REPORT.md, verbatim |

[RUN_INDEX.md](RUN_INDEX.md) lists the runs, including the two not archived; [ARTIFACTS.md](ARTIFACTS.md) hashes
every file and lists the redactions.
