# heat_control/dryrun1 — run index

One row per run. Times are from the trace (UTC); the shutdown time is from the batterystats excerpt (local time there), converted to UTC as in the report.

| Folder / console | Date UTC | Set / blocks | Label |
|---|---|---|---|
| `heat_trace_20261013T072747Z.jsonl`; no console log | 2026-10-13 07:27:48Z – 12:30:28Z (trace); Android `SHUTDOWN` 12:30:25.763Z | `python3 run_mission.py --dry "explore and map the rooms"`, commit 6f81f2a; motors off, unattended, on battery from 94 %; no blocks | VALID motors-off acceptance of heat control (67 pauses, 0 rule violations, 67/67 camera restarts, status 0, skin 34.7–37.1 °C after the first pause); NOT VALID as a complete mission (no `end`; power loss); NOT COMPARABLE with #132 (no Gemma picker, no M2). |
