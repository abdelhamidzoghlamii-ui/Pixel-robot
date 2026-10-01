# Co-residency — run index

Heat labels follow [../thermal_char/HEAT_EVIDENCE.md](../thermal_char/HEAT_EVIDENCE.md).

| Run | When (UTC) | Runner | Output | Status | Heat label |
|---|---|---|---|---|---|
| smoke, 20 s blocks (oneshot 04:00:14Z, 5 min idle) | 2026-10-01 04:05–04:07Z | `85a6e4cb…` | `runs/run_20261001T040522Z_smoke/`, `runs/oneshot_console_20261001T040014Z.log` | complete, no INCOMPLETE line | VALID AS FUNCTIONAL CHECK ONLY |
| full run, 4 × 180 s + loads + B5 (oneshot 04:12:35Z, 5 min idle) | 2026-10-01 04:17–04:50Z | `85a6e4cb…` | `runs/run_20261001T041743Z/`, `runs/oneshot_console_20261001T041235Z.log` | complete, no INCOMPLETE line, no limit, no warm start | **VALID** (skin, status, cpufreq caps) |

## Not archived

Both used older runners and are superseded by the two runs above. Their files stay on the phone; ARTIFACTS.md lists
their hashes.

| Run | When (UTC) | Runner | Phone path | Status | Reason |
|---|---|---|---|---|---|
| `run_20260930T150019Z_smoke` | 2026-09-30 15:00–15:02Z | `ef622037…` | `/data/data/com.termux/files/home/coresidency/run_20260930T150019Z_smoke/` and `/data/data/com.termux/files/home/coresidency/oneshot_console_20260930T145512Z.log` | **NOT ARCHIVED, NOT TRUSTED** | selector count bug: the runner expected 2 calls in a 20 s block where only 1 slot can start, so B3/B4 report INCOMPLETE (CORESIDENCY_FIX1_REPORT.md, cause 1) |
| `run_20260930T182618Z` | 2026-09-30 18:26–18:33Z | `65c572e0…` | `/data/data/com.termux/files/home/coresidency/run_20260930T182618Z/` and `/data/data/com.termux/files/home/coresidency/oneshot_console_20260930T182110Z.log` | **NOT ARCHIVED, NOT TRUSTED** | every block stopped by the zone9 > 80 °C heat stop (time to limit 155.9, 3.6, 2.2, 3.8 s); zone9 is a CPU-core sensor, so the stops say nothing about heat (CLEANUP1 removed that stop) |
