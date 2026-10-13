# DRYRUN-1 — HEAT-A motors-off dry run (read-only evidence analysis)

Run: `python3 run_mission.py --dry "explore and map the rooms"`, native Termux, commit 6f81f2a (HEAT-A), unattended; phone later found switched off.
Analysis: `analyze.py` in this folder (`python3 -I analyze.py heat_trace_20261013T072747Z.jsonl`), run on the copy only.
All numbers below are computed from the trace unless marked "from #132" or "from code".

## 1. Copy and hash — VALID

| file | sha256 |
|---|---|
| original `/termux-home/robot_runs/heat_trace_20261013T072747Z.jsonl` | `96e65b11885352a18c75846612f0cb56cc746f6a1913271bca60c320e977524d` |
| copy `dryrun1/heat_trace_20261013T072747Z.jsonl` | `96e65b11885352a18c75846612f0cb56cc746f6a1913271bca60c320e977524d` |

Match. Size 810873 bytes.

## 2. Integrity — VALID

- 3899 lines, 3899 parse, 0 fail. The file ends with a newline: the last line is **not** cut.
- First: `2026-10-13T07:27:48Z`, t_mono 250488.328738804. Last: `2026-10-13T12:30:28Z`, t_mono 268648.119067454.
- Duration: 18159.8 s = 302.66 min = 5.044 h (t_mono); UTC span 18160 s (agrees).
- t_mono strictly increasing on every row: yes (0 non-increasing).

## 3. Event counts — VALID

| event | count |
|---|---|
| dumpsys | 3630 |
| start | 1 |
| phase_change | 134 |
| camera_stop_ok | 67 |
| camera_start_ok | 67 |
| hold_start / hold_end | 0 / 0 |
| note | 0 |
| end | **0** |
| camera_stop_fail / camera_start_fail | 0 / 0 |

`distance_cm` is 999 on 3897 rows and null on 2 (no ultrasonic reading in dry mode).

## 4. dumpsys cadence — VALID

- 3630 rows, 3629 gaps: median 5.002 s, min 4.873 s, mean 5.004 s, max 5.997 s (the final gap, 12:30:22Z → 12:30:28Z).
- Gaps > 10 s: 0. Gaps > 60 s: 0.
- Rows with an error: 1, the last row of the file: `12:30:28Z`, `detail: "rc -1"`, status/skin null. In `heat_control.root()` (from code) this is su's return code; Python reports a process ended by signal N as −N, so su was ended by signal 1 (SIGHUP). A timeout would have produced a `TimeoutExpired` detail instead. The Python process was still alive to write this row.

## 5. Phase timeline — VALID (0 violations)

Rule (from code, `heat_control.decide`): ACTIVE→PAUSE at skin ≥ 37.0 and ≥ 60 s active; PAUSE→ACTIVE at skin ≤ 35.0 and ≥ 30 s paused. The "decide skin" is the last valid dumpsys row before the camera operation (the reading `decide()` used); duration is row-to-row between `phase_change` rows (first ACTIVE from `start`). Status was 0 on every change.

| # | t_utc | change | decide skin °C | prev phase s |
|---|---|---|---|---|
| 1 | 07:58:35Z | ACTIVE→PAUSE | 37.047 | 1847.0 |
| 2 | 07:59:30Z | PAUSE→ACTIVE | 34.892 | 55.4 |
| 3 | 08:04:21Z | ACTIVE→PAUSE | 37.002 | 290.5 |
| 4 | 08:05:16Z | PAUSE→ACTIVE | 34.930 | 55.4 |
| 5 | 08:11:15Z | ACTIVE→PAUSE | 37.008 | 358.9 |
| 6 | 08:12:11Z | PAUSE→ACTIVE | 34.940 | 55.4 |
| 7 | 08:18:26Z | ACTIVE→PAUSE | 37.002 | 375.4 |
| 8 | 08:19:21Z | PAUSE→ACTIVE | 34.870 | 55.3 |
| 9 | 08:27:41Z | ACTIVE→PAUSE | 37.002 | 500.2 |
| 10 | 08:28:37Z | PAUSE→ACTIVE | 34.952 | 55.3 |
| 11 | 08:32:11Z | ACTIVE→PAUSE | 37.060 | 214.3 |
| 12 | 08:33:12Z | PAUSE→ACTIVE | 34.887 | 61.3 |
| 13 | 08:36:26Z | ACTIVE→PAUSE | 37.004 | 193.8 |
| 14 | 08:37:28Z | PAUSE→ACTIVE | 34.856 | 61.4 |
| 15 | 08:40:50Z | ACTIVE→PAUSE | 37.048 | 202.8 |
| 16 | 08:41:52Z | PAUSE→ACTIVE | 34.872 | 61.3 |
| 17 | 08:44:51Z | ACTIVE→PAUSE | 37.011 | 179.3 |
| 18 | 08:45:48Z | PAUSE→ACTIVE | 34.965 | 56.9 |
| 19 | 08:48:42Z | ACTIVE→PAUSE | 37.050 | 173.8 |
| 20 | 08:49:37Z | PAUSE→ACTIVE | 34.994 | 55.3 |
| 21 | 08:52:41Z | ACTIVE→PAUSE | 37.055 | 184.3 |
| 22 | 08:53:43Z | PAUSE→ACTIVE | 34.874 | 61.5 |
| 23 | 08:57:02Z | ACTIVE→PAUSE | 37.022 | 199.4 |
| 24 | 08:58:02Z | PAUSE→ACTIVE | 34.903 | 59.9 |
| 25 | 09:01:07Z | ACTIVE→PAUSE | 37.013 | 185.3 |
| 26 | 09:02:03Z | PAUSE→ACTIVE | 34.967 | 55.4 |
| 27 | 09:05:08Z | ACTIVE→PAUSE | 37.060 | 184.9 |
| 28 | 09:06:08Z | PAUSE→ACTIVE | 34.976 | 59.8 |
| 29 | 09:08:53Z | ACTIVE→PAUSE | 37.042 | 165.0 |
| 30 | 09:09:54Z | PAUSE→ACTIVE | 34.850 | 61.3 |
| 31 | 09:12:32Z | ACTIVE→PAUSE | 37.004 | 158.1 |
| 32 | 09:13:33Z | PAUSE→ACTIVE | 34.930 | 61.4 |
| 33 | 09:16:13Z | ACTIVE→PAUSE | 37.006 | 159.8 |
| 34 | 09:17:13Z | PAUSE→ACTIVE | 34.905 | 59.9 |
| 35 | 09:19:48Z | ACTIVE→PAUSE | 37.065 | 154.9 |
| 36 | 09:20:54Z | PAUSE→ACTIVE | 34.870 | 66.0 |
| 37 | 09:23:28Z | ACTIVE→PAUSE | 37.024 | 154.3 |
| 38 | 09:24:30Z | PAUSE→ACTIVE | 34.853 | 61.5 |
| 39 | 09:27:13Z | ACTIVE→PAUSE | 37.046 | 163.3 |
| 40 | 09:28:19Z | PAUSE→ACTIVE | 34.910 | 65.9 |
| 41 | 09:30:48Z | ACTIVE→PAUSE | 37.013 | 149.5 |
| 42 | 09:31:50Z | PAUSE→ACTIVE | 34.910 | 61.4 |
| 43 | 09:36:28Z | ACTIVE→PAUSE | 37.011 | 278.0 |
| 44 | 09:37:20Z | PAUSE→ACTIVE | 34.985 | 52.3 |
| 45 | 09:40:38Z | ACTIVE→PAUSE | 37.033 | 197.9 |
| 46 | 09:41:35Z | PAUSE→ACTIVE | 34.989 | 56.9 |
| 47 | 09:44:23Z | ACTIVE→PAUSE | 37.026 | 168.4 |
| 48 | 09:45:25Z | PAUSE→ACTIVE | 34.939 | 61.5 |
| 49 | 09:47:54Z | ACTIVE→PAUSE | 37.011 | 149.0 |
| 50 | 09:48:55Z | PAUSE→ACTIVE | 34.930 | 61.4 |
| 51 | 09:51:20Z | ACTIVE→PAUSE | 37.002 | 144.5 |
| 52 | 09:52:20Z | PAUSE→ACTIVE | 34.978 | 59.8 |
| 53 | 09:54:31Z | ACTIVE→PAUSE | 37.022 | 131.5 |
| 54 | 09:55:37Z | PAUSE→ACTIVE | 34.853 | 66.0 |
| 55 | 09:58:50Z | ACTIVE→PAUSE | 37.033 | 192.8 |
| 56 | 09:59:51Z | PAUSE→ACTIVE | 34.905 | 61.3 |
| 57 | 10:02:36Z | ACTIVE→PAUSE | 37.002 | 165.3 |
| 58 | 10:03:42Z | PAUSE→ACTIVE | 34.840 | 65.8 |
| 59 | 10:06:36Z | ACTIVE→PAUSE | 37.046 | 174.1 |
| 60 | 10:07:36Z | PAUSE→ACTIVE | 34.972 | 59.8 |
| 61 | 10:10:01Z | ACTIVE→PAUSE | 37.008 | 144.7 |
| 62 | 10:11:02Z | PAUSE→ACTIVE | 34.936 | 61.3 |
| 63 | 10:13:51Z | ACTIVE→PAUSE | 37.015 | 168.8 |
| 64 | 10:14:52Z | PAUSE→ACTIVE | 34.981 | 61.3 |
| 65 | 10:17:22Z | ACTIVE→PAUSE | 37.013 | 149.7 |
| 66 | 10:18:28Z | PAUSE→ACTIVE | 34.856 | 65.9 |
| 67 | 10:21:21Z | ACTIVE→PAUSE | 37.013 | 173.4 |
| 68 | 10:22:23Z | PAUSE→ACTIVE | 34.923 | 61.4 |
| 69 | 10:25:02Z | ACTIVE→PAUSE | 37.006 | 159.5 |
| 70 | 10:26:02Z | PAUSE→ACTIVE | 34.900 | 59.8 |
| 71 | 10:28:32Z | ACTIVE→PAUSE | 37.026 | 150.3 |
| 72 | 10:29:38Z | PAUSE→ACTIVE | 34.878 | 65.8 |
| 73 | 10:32:07Z | ACTIVE→PAUSE | 37.008 | 148.7 |
| 74 | 10:33:08Z | PAUSE→ACTIVE | 34.981 | 61.3 |
| 75 | 10:35:37Z | ACTIVE→PAUSE | 37.062 | 148.8 |
| 76 | 10:36:38Z | PAUSE→ACTIVE | 34.987 | 61.4 |
| 77 | 10:39:07Z | ACTIVE→PAUSE | 37.048 | 148.1 |
| 78 | 10:40:08Z | PAUSE→ACTIVE | 34.965 | 61.3 |
| 79 | 10:42:43Z | ACTIVE→PAUSE | 37.022 | 154.8 |
| 80 | 10:43:44Z | PAUSE→ACTIVE | 34.952 | 61.4 |
| 81 | 10:46:07Z | ACTIVE→PAUSE | 37.006 | 143.2 |
| 82 | 10:47:09Z | PAUSE→ACTIVE | 34.943 | 61.4 |
| 83 | 10:49:59Z | ACTIVE→PAUSE | 37.002 | 169.9 |
| 84 | 10:51:05Z | PAUSE→ACTIVE | 34.903 | 65.9 |
| 85 | 10:53:59Z | ACTIVE→PAUSE | 37.060 | 174.1 |
| 86 | 10:55:04Z | PAUSE→ACTIVE | 34.918 | 65.8 |
| 87 | 10:57:43Z | ACTIVE→PAUSE | 37.007 | 158.5 |
| 88 | 10:58:49Z | PAUSE→ACTIVE | 34.880 | 65.8 |
| 89 | 11:01:34Z | ACTIVE→PAUSE | 37.020 | 165.1 |
| 90 | 11:02:40Z | PAUSE→ACTIVE | 34.916 | 66.0 |
| 91 | 11:05:18Z | ACTIVE→PAUSE | 37.049 | 158.5 |
| 92 | 11:06:20Z | PAUSE→ACTIVE | 34.961 | 61.3 |
| 93 | 11:08:49Z | ACTIVE→PAUSE | 37.007 | 148.9 |
| 94 | 11:09:54Z | PAUSE→ACTIVE | 34.925 | 65.8 |
| 95 | 11:12:34Z | ACTIVE→PAUSE | 37.035 | 159.6 |
| 96 | 11:13:35Z | PAUSE→ACTIVE | 34.987 | 61.3 |
| 97 | 11:16:15Z | ACTIVE→PAUSE | 37.015 | 159.3 |
| 98 | 11:17:14Z | PAUSE→ACTIVE | 34.976 | 59.9 |
| 99 | 11:19:49Z | ACTIVE→PAUSE | 37.007 | 154.6 |
| 100 | 11:20:55Z | PAUSE→ACTIVE | 34.925 | 65.8 |
| 101 | 11:23:40Z | ACTIVE→PAUSE | 37.015 | 165.2 |
| 102 | 11:24:46Z | PAUSE→ACTIVE | 34.976 | 65.9 |
| 103 | 11:27:34Z | ACTIVE→PAUSE | 37.002 | 168.4 |
| 104 | 11:28:40Z | PAUSE→ACTIVE | 34.947 | 65.8 |
| 105 | 11:31:19Z | ACTIVE→PAUSE | 37.024 | 159.4 |
| 106 | 11:32:31Z | PAUSE→ACTIVE | 34.867 | 72.0 |
| 107 | 11:35:26Z | ACTIVE→PAUSE | 37.011 | 174.6 |
| 108 | 11:36:31Z | PAUSE→ACTIVE | 34.947 | 64.5 |
| 109 | 11:39:20Z | ACTIVE→PAUSE | 37.011 | 169.9 |
| 110 | 11:40:26Z | PAUSE→ACTIVE | 34.976 | 66.0 |
| 111 | 11:43:21Z | ACTIVE→PAUSE | 37.015 | 174.7 |
| 112 | 11:44:32Z | PAUSE→ACTIVE | 34.985 | 70.3 |
| 113 | 11:47:15Z | ACTIVE→PAUSE | 37.002 | 163.3 |
| 114 | 11:48:27Z | PAUSE→ACTIVE | 34.865 | 71.9 |
| 115 | 11:51:31Z | ACTIVE→PAUSE | 37.022 | 184.0 |
| 116 | 11:52:41Z | PAUSE→ACTIVE | 34.918 | 70.4 |
| 117 | 11:55:40Z | ACTIVE→PAUSE | 37.013 | 178.8 |
| 118 | 11:56:52Z | PAUSE→ACTIVE | 34.985 | 71.8 |
| 119 | 11:59:36Z | ACTIVE→PAUSE | 37.006 | 163.8 |
| 120 | 12:00:42Z | PAUSE→ACTIVE | 34.923 | 65.8 |
| 121 | 12:03:31Z | ACTIVE→PAUSE | 37.006 | 169.4 |
| 122 | 12:04:43Z | PAUSE→ACTIVE | 34.940 | 71.8 |
| 123 | 12:07:37Z | ACTIVE→PAUSE | 37.015 | 174.1 |
| 124 | 12:08:47Z | PAUSE→ACTIVE | 34.845 | 70.4 |
| 125 | 12:11:42Z | ACTIVE→PAUSE | 37.059 | 174.7 |
| 126 | 12:12:52Z | PAUSE→ACTIVE | 34.979 | 70.3 |
| 127 | 12:15:42Z | ACTIVE→PAUSE | 37.004 | 169.8 |
| 128 | 12:16:52Z | PAUSE→ACTIVE | 34.890 | 70.4 |
| 129 | 12:20:12Z | ACTIVE→PAUSE | 37.008 | 199.9 |
| 130 | 12:21:27Z | PAUSE→ACTIVE | 34.863 | 74.9 |
| 131 | 12:24:18Z | ACTIVE→PAUSE | 37.011 | 170.9 |
| 132 | 12:25:29Z | PAUSE→ACTIVE | 34.949 | 70.4 |
| 133 | 12:28:04Z | ACTIVE→PAUSE | 37.024 | 155.2 |
| 134 | 12:29:13Z | PAUSE→ACTIVE | 34.979 | 68.9 |

Violations: **none**. Every ACTIVE→PAUSE skin ≥ 37.002; every PAUSE→ACTIVE skin ≤ 34.994; shortest active 131.5 s, shortest pause 52.3 s, so measuring durations row-to-row instead of from the exact `since` cannot flip any result.

Durations (row-to-row):
- Active slots (67 complete): median 168.4 s, min 131.5 s, max 1847.0 s (the first). After the first: 66 slots, median 168.4 s, min 131.5 s, max 500.2 s.
- Pauses (67): median 61.4 s, min 52.3 s, max 74.9 s.
- Trailing ACTIVE when the trace stops: 74.8 s.
- Inference duty (ACTIVE time / total): 0.767 whole run; 0.740 from the first pause.

## 6. Extremes — VALID (skin band: within #132's)

- Max skin: 37.0819 °C at 10:54:02Z (1 row). Min skin: 26.116302 °C (07:27:53Z). Start skin 26.378902 °C.
- Max Android status: 0 (all 3629 valid rows; 1 null, the last row). HOLD: none.
- Max battery_c: 33.4 °C (3 rows, first 11:45:59Z, last 12:30:28Z). First 22.1 °C.
- Max cpu_max_c: 89.0 °C at 09:36:12Z (1 row).
- After the first pause (from 07:58:35Z, 3259 skin rows): skin 34.689–37.082 °C. One row below 34.7 (34.688797 at 08:37:30Z, ACTIVE), none above 37.4. **Within** #132 adaptive's "about 34.7–37.4 °C".
- Differences from #132 adaptive (from #132: first active 9.6 min, later active slots about 61 s, pauses about 43–60 s, duty from first pause 0.497): here first active 1847.0 s (30.8 min, from 26.4 °C), later active median 168.4 s, pauses median 61.4 s, duty from first pause 0.740. This suggests the dry mission loop heats the phone more slowly than the #132 L2 campaign load (inference, not measured here). Late pauses lengthen (52–66 s before about 11:30Z, 64.5–74.9 s after).

## 7. Ending — the trace just stops (NOT a mission end)

No `end` event; no `camera_stop_*` after the last `camera_start_ok`; no note. Last 15 lines verbatim:

```
{"t_mono": 268576.878139197, "t_utc": "2026-10-13T12:29:16Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 34.913605, "battery_c": 32.9, "cpu_max_c": 46.0, "distance_cm": 999, "detail": null}
{"t_mono": 268581.852340208, "t_utc": "2026-10-13T12:29:21Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.1094, "battery_c": 33.0, "cpu_max_c": 43.0, "distance_cm": 999, "detail": null}
{"t_mono": 268586.828387208, "t_utc": "2026-10-13T12:29:26Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.1094, "battery_c": 32.8, "cpu_max_c": 44.0, "distance_cm": 999, "detail": null}
{"t_mono": 268591.943764407, "t_utc": "2026-10-13T12:29:31Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.390503, "battery_c": 32.9, "cpu_max_c": 46.0, "distance_cm": 999, "detail": null}
{"t_mono": 268596.915677984, "t_utc": "2026-10-13T12:29:36Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.5797, "battery_c": 33.0, "cpu_max_c": 48.0, "distance_cm": 999, "detail": null}
{"t_mono": 268601.931443367, "t_utc": "2026-10-13T12:29:41Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.8045, "battery_c": 33.2, "cpu_max_c": 51.0, "distance_cm": 999, "detail": null}
{"t_mono": 268606.906576833, "t_utc": "2026-10-13T12:29:46Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.8045, "battery_c": 33.1, "cpu_max_c": 50.0, "distance_cm": 999, "detail": null}
{"t_mono": 268611.874567843, "t_utc": "2026-10-13T12:29:51Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 35.8912, "battery_c": 33.0, "cpu_max_c": 50.0, "distance_cm": 999, "detail": null}
{"t_mono": 268616.954182877, "t_utc": "2026-10-13T12:29:56Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 36.037704, "battery_c": 33.0, "cpu_max_c": 46.0, "distance_cm": 999, "detail": null}
{"t_mono": 268621.950933286, "t_utc": "2026-10-13T12:30:01Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 36.138, "battery_c": 33.0, "cpu_max_c": 47.0, "distance_cm": 999, "detail": null}
{"t_mono": 268626.965312317, "t_utc": "2026-10-13T12:30:06Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 36.138, "battery_c": 32.9, "cpu_max_c": 48.0, "distance_cm": 999, "detail": null}
{"t_mono": 268631.90517556, "t_utc": "2026-10-13T12:30:11Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 36.2182, "battery_c": 33.0, "cpu_max_c": 50.0, "distance_cm": 999, "detail": null}
{"t_mono": 268636.939002141, "t_utc": "2026-10-13T12:30:16Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 36.3361, "battery_c": 33.0, "cpu_max_c": 47.0, "distance_cm": 999, "detail": null}
{"t_mono": 268642.141785998, "t_utc": "2026-10-13T12:30:22Z", "event": "dumpsys", "phase": "ACTIVE", "status": 0, "skin_c": 36.3782, "battery_c": 33.0, "cpu_max_c": 56.0, "distance_cm": 999, "detail": null}
{"t_mono": 268648.119067454, "t_utc": "2026-10-13T12:30:28Z", "event": "dumpsys", "phase": "ACTIVE", "status": null, "skin_c": null, "battery_c": 33.4, "cpu_max_c": 59.0, "distance_cm": 999, "detail": "rc -1"}
```

**The mission did not end itself.** It was mid-ACTIVE (skin 36.38 °C, status 0, battery 33.4 °C — no heat rule near firing). The final dumpsys call's `su` was ended by a signal (rc −1) and nothing was written after: no next dumpsys row about 5 s later, no fail-closed `end` (that needs > 60 s without a reading). The process was killed or the phone lost power/shut down within seconds of 12:30:28Z. Trace mtime 14:30 local = 12:30Z matches. The trace alone does not show why (it holds no battery level); section 8 does: low battery.

## 8. Shutdown cause — VALID: low-battery shutdown

Owner ran the step-8 command in native Termux (files in `~/robot_runs/`, written 2026-10-14 00:26 local). History times are local (UTC+2): the camera start at `14:25:28` in the history matches the trace's PAUSE→ACTIVE at `12:25:29Z`.

- `dryrun1_bootreason.txt`: 0 bytes — no `boot.reason` property set (matches the earlier empty getprop).
- `dryrun1_pstore_ls.txt`: `/sys/fs/pstore` is empty (`total 0`, only `.` and `..`, dir time `2026-10-13 18:47`) — no kernel panic / ramoops record.
- `dryrun1_battery_capacity.txt`: `97` (now, after recharge).
- `dryrun1_batterystats_history.txt` (24439915 bytes; header says the buffer is `101% used`, but it covers the whole run). Quoted lines:

```
  10-13 09:27:30.602 RESET:TIME: 2026-10-13-09-27-30
  10-13 09:27:30.602 094 status=not-charging health=good plug=none temp=224 volt=4092 charge=3400 ...
  10-13 09:27:30.602 094 -plugged
  10-13 09:27:30.620 094 status=discharging -wifi_scan
  10-13 10:00:00.024 083 ...
  10-13 11:00:00.016 063 ...
  10-13 12:00:00.107 044 ...
  10-13 13:00:00.016 025 ...
  10-13 14:00:00.100 006 ...
  10-13 14:18:01.990 001 volt=3359 charge=124
  10-13 14:30:20.324 000 volt=3190 charge=-12
  10-13 14:30:24.223 000 volt=3087
  10-13 14:30:25.535 000 stats=0:"shutdown"
  10-13 14:30:25.763 SHUTDOWN
  10-13 14:30:25.879 START
  10-13 18:47:28.923 006 status=charging health=good plug=ac temp=197 volt=3667 charge=198 ... +plugged +charging
```

Reading: the charger was unplugged at 09:27:30 local (07:27:30Z), 17 s before the trace's first row (07:27:48Z), at 94 %. No later `plug=`/`status=` change appears before the shutdown: the whole run was on battery. The level fell to 1 % at 14:18:01 and 0 % at 14:30:20 (3190 mV, then 3087 mV at 14:30:24), and Android recorded `SHUTDOWN` at 14:30:25.763 local = 12:30:25.763Z. On battery from unplug to SHUTDOWN: 18175.2 s (5 h 02 min 55 s), 94 % → 0 %. The trace's last row (12:30:28Z, su ended by a signal, `rc -1`) falls in the shutdown sequence about 2 s after the marker. The next history entry is the boot at 18:47:28 (6 %, on AC), matching the pstore dir time.

**Verdict: low-battery shutdown** (battery at 0 % with a clean Android `SHUTDOWN`). Not thermal: Android thermal status stayed 0, battery temperature peaked at 33.4 °C, and the heat rules never stopped the mission. Not a kernel crash: pstore empty.

## Verdicts

| check | verdict |
|---|---|
| 1 hash match | VALID |
| 2 integrity | VALID |
| 3 event counts | VALID |
| 4 dumpsys cadence | VALID |
| 5 phase rule | VALID (0 violations) |
| 6 extremes / #132 band | VALID |
| 7 ending | NOT VALID as a complete mission: no `end`; trace stops |
| 8 shutdown cause | VALID — low-battery shutdown (0 %, Android SHUTDOWN 12:30:25.763Z); not thermal |
