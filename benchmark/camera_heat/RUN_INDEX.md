# Camera heat — run index

Heat labels follow [../thermal_char/HEAT_EVIDENCE.md](../thermal_char/HEAT_EVIDENCE.md).

| Run | When (UTC) | Output | Status | Heat label |
|---|---|---|---|---|
| full run, 4 × 180 s (oneshot 02:25:11Z, 5 min idle) | 2026-09-29 02:30–02:43Z | `runs/camera_heat_20260929T022511Z_16984/`, `runs/oneshot_console_20260929T022511Z.log` | complete, all four blocks | temperatures **NOT TRUSTED (zone9)**; power and RAM **STILL VALID WITH CAVEAT** (CPU capping unknown) |

## Not archived

| Run | When (UTC) | Phone path | Status | Reason |
|---|---|---|---|---|
| `camera_heat_20260929T010749Z_10076` | 2026-09-29 01:07Z | `/data/data/com.termux/files/home/ladder/camera_heat_20260929T010749Z_10076/` | **NOT ARCHIVED, NOT TRUSTED** | aborted start: only an empty `sensors.jsonl`, no results |
| `camera_heat_20260929T021119Z_12612` | 2026-09-29 02:11–02:18Z | `/data/data/com.termux/files/home/ladder/camera_heat_20260929T021119Z_12612/` and `/data/data/com.termux/files/home/ladder/oneshot_console_20260929T021119Z.log` | **NOT ARCHIVED, NOT TRUSTED** | 20 s toy (functional check before the full run); zone9 temperatures only |

Also left on the phone: `/data/data/com.termux/files/home/ladder/camera_heat_active.json` (78 bytes, last run name),
`/data/data/com.termux/files/home/ladder/camera_heat_helper.log` (empty), `/data/data/com.termux/files/home/ladder/run_camera_heat.sh`
(same SHA-256 as the archived `run_camera_heat.sh`).
