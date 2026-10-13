# DRYRUN-1 — HEAT-A motors-off dry run archive

Evidence from the HEAT-A motors-off dry run (adaptive 37/35 °C VIRTUAL-SKIN pauses, camera off while paused). Full analysis: [DRYRUN1_REPORT.md](DRYRUN1_REPORT.md). Every number below is copied from that report.

- Command: `python3 run_mission.py --dry "explore and map the rooms"`, native Termux.
- Commit: `6f81f2a` (HEAT-A).
- Time: 2026-10-13 07:27:48Z (first trace row) to 12:30:28Z (last trace row); 18159.8 s = 5.044 h.
- Setup: motors off (`--dry`), unattended, on battery from 94 % (charger unplugged at 07:27:30Z, 17 s before the
  first trace row; no later plug change before the shutdown).
- End: Android low-battery `SHUTDOWN` at 12:30:25.763Z, battery 0 %. The trace has no `end` event; its last row
  (12:30:28Z, `rc -1`) falls in the shutdown sequence.

## Label (owner decision 2026-10-14)

- **VALID** motors-off acceptance of heat control: 67 pauses, 0 rule violations, 67/67 camera restarts, status 0,
  skin 34.7–37.1 °C after the first pause.
- **NOT VALID** as a complete mission: no `end`; power loss.
- **NOT COMPARABLE** with #132: no Gemma picker, no M2.

No other follow-up changes (owner decision 2026-10-14).

## Key numbers (from the report)

| Item | Value |
|---|---|
| Trace lines | 3899, 0 parse failures; last line not cut |
| dumpsys rows / gap | 3630 rows; gap median 5.002 s, max 5.997 s; 0 gaps > 10 s |
| Phase changes | 134 (67 ACTIVE→PAUSE, 67 PAUSE→ACTIVE); 0 violations |
| Camera | 67 `camera_stop_ok`, 67 `camera_start_ok`, 0 stop/start failures |
| Decide skin at changes | every ACTIVE→PAUSE ≥ 37.002 °C; every PAUSE→ACTIVE ≤ 34.994 °C |
| First active slot | 1847.0 s (30.8 min, from start skin 26.4 °C) |
| Later active slots (66) | median 168.4 s, min 131.5 s, max 500.2 s |
| Pauses (67) | median 61.4 s, min 52.3 s, max 74.9 s |
| Inference duty | 0.767 whole run; 0.740 from the first pause |
| Skin after the first pause | 34.689–37.082 °C (one row below 34.7, none above 37.4) |
| Max skin | 37.0819 °C |
| Android thermal status | max 0; HOLD none |
| Max battery_c | 33.4 °C |
| Max cpu_max_c | 89.0 °C (CPU-zone fault-check value, not a heat state) |
| On battery to SHUTDOWN | 18175.2 s, 94 % → 0 % |

## Caveats

- Single run.
- Ambient temperature not recorded.
- Rest before the start not recorded.
- Ran to 0 % battery; the run ended by power loss, not by the mission.

## Files

- `heat_trace_20261013T072747Z.jsonl` — the trace written by the run.
- `DRYRUN1_REPORT.md`, `analyze.py` — the read-only analysis and its script.
- `android/bootreason.txt`, `android/pstore_ls.txt`, `android/battery_capacity.txt` — owner-collected shutdown
  evidence (report section 8).
- `android/batterystats_excerpt.txt` — only the batterystats history lines quoted in report section 8. The full
  history stays outside the repository; its path, size and SHA-256 are in the excerpt header.

Hashes and sources: [ARTIFACTS.md](ARTIFACTS.md). Run row: [RUN_INDEX.md](RUN_INDEX.md). Heat label:
[HEAT_EVIDENCE.md](../../thermal_char/HEAT_EVIDENCE.md).

## Rerun the analysis

From this folder, with the native Termux Python; it only prints and writes no file:

```bash
/data/data/com.termux/files/usr/bin/python3 -I -B analyze.py heat_trace_20261013T072747Z.jsonl
```
