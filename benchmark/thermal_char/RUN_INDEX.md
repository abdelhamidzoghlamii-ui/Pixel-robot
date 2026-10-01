# Thermal characterization — run index

Heat labels follow [HEAT_EVIDENCE.md](HEAT_EVIDENCE.md). Both runs executed `thermal_char.py` `cbf73d12…` and
`coresidency.py` `65c572e0…` (copies archived here as `*_executed_*.py`).

| Run | When (UTC) | Output | Status | Heat label |
|---|---|---|---|---|
| smoke, 60 s load + 30 s cooldown (oneshot 23:06:09Z, 5 min idle) | 2026-09-30 23:11–23:13Z | `runs/run_20260930T231117Z_smoke/`, `runs/oneshot_console_20260930T230609Z.log` | stopped at planned 60 s; INCOMPLETE: RobotCam pid after the load stop | VALID AS FUNCTIONAL CHECK ONLY |
| full, 1200 s load + 300 s cooldown (oneshot 23:27:00Z, 5 min idle) | 2026-09-30 23:32–23:57Z | `runs/run_20260930T233208Z/`, `runs/oneshot_console_20260930T232700Z.log` | stopped at planned 1200 s; INCOMPLETE: RobotCam pid after the load stop | **VALID** (skin, status, cpufreq caps) |

`runs/thermal.log` is the launcher's zone9/10/11 log across both sessions (zone readings only, not a heat state).
No other thermal_char run exists on the phone.
