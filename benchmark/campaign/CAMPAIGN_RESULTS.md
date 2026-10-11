# Campaign results

Generated offline by `python3 -B benchmark/campaign/summarize_campaign.py`; exact numeric values and full phase/cycle/memory details: [CAMPAIGN_RESULTS.json](CAMPAIGN_RESULTS.json). Raw invocations are cited in each table. No hardware was rerun.

## Caveats

- Start skins differ; direct thermal comparisons are conditional on those recorded starts.
- Picker accuracy NOT EVALUATED; latency, execution and cadence misses are not accuracy.
- Single measured runs at one ambient temperature; no repeatability or ambient sweep. Ambient is not recorded in the P23 JSON.
- Fixed had a short rest: observed monotonic gap below the owner >=30 min rule.
- Monotonic rest gaps use increasing device timestamps; no boot ID is recorded, so cross-invocation clock continuity is UNVERIFIED.
- Phase-1 fixed validity labels are retained, including NOT COMPARABLE — START TEMP and rehearsals.

## Methods

- power: power.weighted on raw current_now receipt samples: clipped trapezoids, <=1.5 s gaps, no extrapolation.
- p95: Nearest rank ceil(.95*n); median is statistics.median.
- m2: Inference-only duration; calls.csv also keeps full wall duration.
- active_minutes: Cumulative active-phase wall time, including camera start and boundary waits; pauses excluded.
- skin_crossings: First recorded skin receipt >= threshold; no interpolation.
- caps: Later sample owns preceding interval; no tail credit, identical to power_map.capped_by_policy.
- scope: P23 measured origin through end; idle, gate and cleanup excluded from mode totals. Phase1 comparison excludes warmup.

## Phase 1 — all archived blocks, original labels

| Run / phase (source) | Layout | Label | Mean W | M2 median / p95 ms | Picker median / p95 ms |
|---|---|---|---|---|---|
| [owner_rehearsal_p1_fix3c / active_01](runs/owner_rehearsal_p1_fix3c.json) | L0 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 4.542860 | null / null | 1827.350000 / 1858.200000 |
| [owner_rehearsal_p1_fix3c / active_02](runs/owner_rehearsal_p1_fix3c.json) | L1 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.143199 | 4900.169355 / 5131.302005 | 2490.450000 / 2602.500000 |
| [owner_rehearsal_p1_fix3c / active_03](runs/owner_rehearsal_p1_fix3c.json) | L2 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.101102 | 886.241232 / 942.910360 | 1781.950000 / 1838.600000 |
| [owner_rehearsal_p1_fix3c / active_04](runs/owner_rehearsal_p1_fix3c.json) | L3 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.462355 | 1380.895061 / 1767.991985 | 2762.700000 / 2877.300000 |
| [owner_rehearsal_p1_fix3c / active_05](runs/owner_rehearsal_p1_fix3c.json) | L4 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.847141 | 798.275798 / 1589.313233 | 2920.300000 / 2924.400000 |
| [owner_rehearsal_p1_fix3c / active_06](runs/owner_rehearsal_p1_fix3c.json) | L0 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 4.546426 | null / null | 1704.850000 / 1813.000000 |
| [owner_rehearsal_p1_fix4d / active_01](runs/owner_rehearsal_p1_fix4d.json) | L0 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 4.355454 | null / null | 1824.350000 / 1874.000000 |
| [owner_rehearsal_p1_fix4d / active_02](runs/owner_rehearsal_p1_fix4d.json) | L1 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.330283 | 5006.541100 / 5187.802940 | 2441.450000 / 2478.000000 |
| [owner_rehearsal_p1_fix4d / active_03](runs/owner_rehearsal_p1_fix4d.json) | L2 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.394394 | 893.528423 / 976.049073 | 2144.050000 / 2605.800000 |
| [owner_rehearsal_p1_fix4d / active_04](runs/owner_rehearsal_p1_fix4d.json) | L3 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 5.493211 | 1337.736817 / 1878.584189 | 2786.300000 / 2865.300000 |
| [owner_rehearsal_p1_fix4d / active_05](runs/owner_rehearsal_p1_fix4d.json) | L4 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 6.102339 | 771.557170 / 1777.721680 | 3064.700000 / 3122.200000 |
| [owner_rehearsal_p1_fix4d / active_06](runs/owner_rehearsal_p1_fix4d.json) | L0 | NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage | 4.197935 | null / null | 1688.050000 / 1759.900000 |
| [owner_session_p1_fix4d / active_01](runs/owner_session_p1_fix4d.json) | L0 | VALID | 4.097380 | null / null | 1733.400000 / 1958.100000 |
| [owner_session_p1_fix4d / active_02](runs/owner_session_p1_fix4d.json) | L1 | VALID | 4.991155 | 4918.740033 / 6451.268233 | 2819.300000 / 2984.700000 |
| [owner_session_p1_fix4d / active_03](runs/owner_session_p1_fix4d.json) | L2 | NOT COMPARABLE — START TEMP | 4.715038 | 1176.106853 / 1323.148764 | 2129.000000 / 2658.700000 |
| [owner_session_p1_fix4d / active_04](runs/owner_session_p1_fix4d.json) | L3 | VALID | 5.095646 | 1313.017946 / 1759.718467 | 2955.600000 / 3484.000000 |
| [owner_session_p1_fix4d / active_05](runs/owner_session_p1_fix4d.json) | L4 | NOT COMPARABLE — START TEMP | 5.411293 | 871.105367 / 2000.549683 | 3686.300000 / 4406.900000 |
| [owner_session_p1_fix4d / active_06](runs/owner_session_p1_fix4d.json) | L0 | NOT COMPARABLE — START TEMP | 4.038427 | null / null | 1774.700000 / 2015.300000 |
| [owner_session_p1a_fix / active_01](runs/owner_session_p1a_fix.json) | L0 | VALID | 3.883226 | null / null | 1770.500000 / 2060.200000 |
| [owner_session_p1a_fix / active_02](runs/owner_session_p1a_fix.json) | L1 | NOT VALID — WARM START | 4.022536 | null / null | 1935.200000 / 2245.700000 |
| [owner_session_p1a_fix / active_03](runs/owner_session_p1a_fix.json) | L2 | NOT VALID — WARM START | 3.956889 | null / null | 1817.300000 / 2062.600000 |

## owner_adaptive_p23_01 — adaptive

Label: **VALID**; checks: VALID; results: VALID. [Raw invocation](runs/owner_adaptive_p23_01.json).

| Metric | Value |
|---|---|
| duration s | 2160.000000 |
| start skin C | 24.599300 |
| idle start skin C | 25.356400 |
| max skin C | 37.434700 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | 417.071527 / 572.915693 / null |
| whole / active / pause mean W | 3.541867 / 4.724322 / 1.348299 |
| active_s | 1403.458451 |
| pause_s | 756.541549 |
| camera_start_s | 37.391834 |
| inference_s | 1358.646809 |
| camera_stop_s | 66.365261 |
| camera_off_s | 689.112090 |
| camera_on_without_inference_s | 112.241101 |
| pause_count | 14 |
| mean_pause_s | 54.038682 |
| camera_restarts | 13 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.629003 |
| memory available min MiB | 1715 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 13} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 1105 | 151.555380 | 214.504720 |
| overall | yolo_640 | 264 | 566.602702 | 726.537151 |
| overall | m2 | 70 | 1000.873800 | 1187.435670 |
| overall | picker | 70 | 2500.250000 | 2915.900000 |
| first_5_active_minutes | yolo_320 | 239 | 150.082764 | 223.230550 |
| first_5_active_minutes | yolo_640 | 59 | 565.571167 | 709.135946 |
| first_5_active_minutes | m2 | 15 | 951.473633 | 1070.628174 |
| first_5_active_minutes | picker | 15 | 2429.000000 | 3039.300000 |
| last_5_active_minutes | yolo_320 | 235 | 152.747884 | 204.590861 |
| last_5_active_minutes | yolo_640 | 55 | 571.328654 | 762.921835 |
| last_5_active_minutes | m2 | 15 | 1089.364014 | 1386.365683 |
| last_5_active_minutes | picker | 15 | 2432.700000 | 2775.600000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 63.295976 | 95.776419 | 930.000000 | 930.000000 |
| policy4 | 0.093045 | 0.143201 | 1197.000000 | 1197.000000 |
| policy6 | 6.868997 | 9.728901 | 1426.000000 | 1426.000000 |

YOLO due / done: `{'320': 1105, '640': 264}` / `{'320': 1105, '640': 264}`.

Steady-state duty from first pause: `{'from_s': 573.7470937859907, 'duration_s': 1586.2529062140093, 'inference_s': 787.7911773880769, 'fraction': 0.49663655417239755}`.

| Phase | State | Start s | Duration s | Inference s | Mean W | Skin min C | Skin max C |
|---|---|---|---|---|---|---|---|
| active_01 | active | 0.000000 | 573.747094 | 570.855632 | 4.732183 | 24.599300 | 37.126400 |
| pause_01 | pause | 573.747094 | 43.266338 | 0.000000 | 1.386683 | 34.849304 | 37.126400 |
| active_02 | active | 617.013432 | 93.654749 | 90.632581 | 4.720771 | 34.711605 | 37.099804 |
| pause_02 | pause | 710.668181 | 47.596056 | 0.000000 | 1.314358 | 34.989304 | 37.292603 |
| active_03 | active | 758.264237 | 60.942446 | 57.165887 | 4.692379 | 34.797802 | 37.099304 |
| pause_03 | pause | 819.206683 | 46.205473 | 0.000000 | 1.430413 | 34.900303 | 37.071102 |
| active_04 | active | 865.412156 | 61.103879 | 58.175967 | 4.668696 | 34.831400 | 37.254200 |
| pause_04 | pause | 926.516035 | 50.932137 | 0.000000 | 1.286603 | 34.987400 | 37.254200 |
| active_05 | active | 977.448172 | 61.111730 | 58.169056 | 4.643078 | 34.829304 | 37.278603 |
| pause_05 | pause | 1038.559902 | 50.695307 | 0.000000 | 1.355103 | 34.989304 | 37.276104 |
| active_06 | active | 1089.255208 | 61.106416 | 58.176461 | 4.688723 | 34.929200 | 37.380900 |
| pause_06 | pause | 1150.361624 | 55.936459 | 0.000000 | 1.366762 | 34.969200 | 37.298904 |
| active_07 | active | 1206.298083 | 61.562818 | 58.178651 | 4.772649 | 34.853500 | 37.295605 |
| pause_07 | pause | 1267.860901 | 60.098827 | 0.000000 | 1.265724 | 34.875504 | 37.262302 |
| active_08 | active | 1327.959728 | 61.593696 | 58.182288 | 4.772384 | 34.715800 | 37.317200 |
| pause_08 | pause | 1389.553425 | 60.113463 | 0.000000 | 1.385672 | 34.867000 | 37.317200 |
| active_09 | active | 1449.666887 | 61.630649 | 58.176456 | 4.793917 | 34.720500 | 37.215400 |
| pause_09 | pause | 1511.297537 | 55.268430 | 0.000000 | 1.376461 | 34.902203 | 37.215400 |
| active_10 | active | 1566.565967 | 61.153452 | 58.201821 | 4.781127 | 34.869503 | 37.318604 |
| pause_10 | pause | 1627.719419 | 55.817795 | 0.000000 | 1.447023 | 34.987400 | 37.318604 |
| active_11 | active | 1683.537214 | 61.484549 | 58.177644 | 4.693137 | 34.856003 | 37.434700 |
| pause_11 | pause | 1745.021763 | 60.253174 | 0.000000 | 1.289791 | 34.933700 | 37.346905 |
| active_12 | active | 1805.274937 | 61.444166 | 58.174399 | 4.782823 | 34.779900 | 37.290400 |
| pause_12 | pause | 1866.719103 | 60.239333 | 0.000000 | 1.365155 | 34.873604 | 37.290400 |
| active_13 | active | 1926.958436 | 61.518863 | 58.183076 | 4.503478 | 34.742500 | 37.316303 |
| pause_13 | pause | 1988.477299 | 60.384807 | 0.000000 | 1.314440 | 34.953804 | 37.276104 |
| active_14 | active | 2048.862106 | 61.403944 | 58.196889 | 4.830483 | 34.816105 | 37.374000 |
| pause_14 | pause | 2110.266050 | 49.733950 | 0.000000 | 1.310665 | 35.386105 | 37.374000 |

## owner_endurance_p23_01 — endurance

Label: **VALID**; checks: VALID; results: VALID. [Raw invocation](runs/owner_endurance_p23_01.json).

| Metric | Value |
|---|---|
| duration s | 1800.000000 |
| start skin C | 21.919401 |
| idle start skin C | 21.333601 |
| max skin C | 40.757202 |
| max Android status | 1 |
| time to 35 / 37 / 39 C s | 519.236567 / 675.092516 / 996.521710 |
| whole / active / pause mean W | 4.399329 / 4.399329 / null |
| active_s | 1800.000000 |
| pause_s | 0 |
| camera_start_s | 2.665148 |
| inference_s | 1794.226963 |
| camera_stop_s | 0.000000 |
| camera_off_s | 0.023719 |
| camera_on_without_inference_s | 5.749318 |
| pause_count | 0 |
| mean_pause_s | 0.000000 |
| camera_restarts | 0 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.996793 |
| memory available min MiB | 1677 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 0} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 1436 | 169.339579 | 239.934164 |
| overall | yolo_640 | 358 | 635.536865 | 844.901205 |
| overall | m2 | 89 | 874.195435 | 1208.382040 |
| overall | picker | 89 | 3186.800000 | 5485.500000 |
| first_5_active_minutes | yolo_320 | 239 | 155.080404 | 212.484456 |
| first_5_active_minutes | yolo_640 | 59 | 578.409627 | 844.901205 |
| first_5_active_minutes | m2 | 15 | 820.192424 | 937.772909 |
| first_5_active_minutes | picker | 15 | 1963.800000 | 3129.800000 |
| last_5_active_minutes | yolo_320 | 238 | 197.369263 | 294.377116 |
| last_5_active_minutes | yolo_640 | 59 | 744.586223 | 993.045614 |
| last_5_active_minutes | m2 | 14 | 1172.056356 | 1353.062012 |
| last_5_active_minutes | picker | 14 | 4527.100000 | 5997.300000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 99.721472 | 99.721472 | 1328.000000 | 1328.000000 |
| policy4 | 48.607875 | 48.607875 | 1836.000000 | 1836.000000 |
| policy6 | 56.553695 | 56.553695 | 1426.000000 | 1426.000000 |

YOLO due / done: `{'320': 1436, '640': 358}` / `{'320': 1436, '640': 358}`.

## owner_fixed_p23_01 — fixed

Label: **VALID**; checks: VALID; results: VALID. [Raw invocation](runs/owner_fixed_p23_01.json).

| Metric | Value |
|---|---|
| duration s | 2160.000000 |
| start skin C | 27.390300 |
| idle start skin C | 28.623602 |
| max skin C | 38.022205 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | 451.474169 / 816.718734 / null |
| whole / active / pause mean W | 3.303409 / 4.294048 / 1.322132 |
| active_s | 1440.000000 |
| pause_s | 720.000000 |
| camera_start_s | 32.230949 |
| inference_s | 1358.498307 |
| camera_stop_s | 58.268251 |
| camera_off_s | 659.931323 |
| camera_on_without_inference_s | 141.570369 |
| pause_count | 12 |
| mean_pause_s | 60.000000 |
| camera_restarts | 11 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.628934 |
| memory available min MiB | 1797 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 11} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 1104 | 151.227397 | 203.005493 |
| overall | yolo_640 | 264 | 568.417216 | 710.789185 |
| overall | m2 | 60 | 963.407858 | 1223.578899 |
| overall | picker | 60 | 2473.350000 | 3028.100000 |
| first_5_active_minutes | yolo_320 | 231 | 150.104737 | 212.499878 |
| first_5_active_minutes | yolo_640 | 55 | 563.754639 | 753.169841 |
| first_5_active_minutes | m2 | 13 | 938.317343 | 999.325643 |
| first_5_active_minutes | picker | 13 | 2558.000000 | 3028.100000 |
| last_5_active_minutes | yolo_320 | 230 | 151.608724 | 203.971680 |
| last_5_active_minutes | yolo_640 | 55 | 568.462768 | 702.166504 |
| last_5_active_minutes | m2 | 12 | 977.402527 | 1095.605754 |
| last_5_active_minutes | picker | 12 | 2035.900000 | 2849.800000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 65.379280 | 96.612869 | 1328.000000 | 1328.000000 |
| policy4 | 1.622092 | 2.433138 | 2253.000000 | 2253.000000 |
| policy6 | 13.236491 | 19.787522 | 1745.000000 | 1745.000000 |

YOLO due / done: `{'320': 1104, '640': 264}` / `{'320': 1104, '640': 264}`.

Per-cycle table:

| Cycle | Active s | Pause s | Inference s | Duty | Skin min C | Skin max C | Camera start s | Camera stop s |
|---|---|---|---|---|---|---|---|---|
| 1 | 120.000000 | 60.000000 | 113.182759 | 0.628793 | 27.374200 | 32.037903 | 2.732756 | 4.865865 |
| 2 | 120.000000 | 60.000000 | 113.230717 | 0.629060 | 29.907303 | 33.760900 | 2.618724 | 5.165620 |
| 3 | 120.000000 | 60.000000 | 113.180381 | 0.628780 | 31.463600 | 35.281400 | 2.666825 | 4.708204 |
| 4 | 120.000000 | 60.000000 | 113.202048 | 0.628900 | 32.730103 | 36.314304 | 2.674152 | 4.792347 |
| 5 | 120.000000 | 60.000000 | 113.174142 | 0.628745 | 33.662900 | 37.076300 | 2.618663 | 4.734635 |
| 6 | 120.000000 | 60.000000 | 113.229872 | 0.629055 | 34.375202 | 37.471000 | 2.673688 | 5.469727 |
| 7 | 120.000000 | 60.000000 | 113.180763 | 0.628782 | 34.915900 | 37.616802 | 2.676518 | 4.896594 |
| 8 | 120.000000 | 60.000000 | 113.243888 | 0.629133 | 35.125400 | 37.582800 | 2.606703 | 4.809752 |
| 9 | 120.000000 | 60.000000 | 113.185783 | 0.628810 | 35.227900 | 38.022205 | 2.616230 | 4.684791 |
| 10 | 120.000000 | 60.000000 | 113.231873 | 0.629066 | 35.370400 | 37.938602 | 2.612395 | 4.741659 |
| 11 | 120.000000 | 60.000000 | 113.272941 | 0.629294 | 35.377304 | 37.849200 | 3.162852 | 4.727649 |
| 12 | 120.000000 | 60.000000 | 113.183140 | 0.628795 | 35.332302 | 37.803700 | 2.571442 | 4.671409 |

## owner_rehearsal_p23_01_endurance — endurance

Label: **NOT VALID — OWNER REHEARSAL**; checks: NOT VALID — power coverage; results: NOT VALID — OWNER REHEARSAL. [Raw invocation](runs/owner_rehearsal_p23_01.json).

| Metric | Value |
|---|---|
| duration s | 180.000000 |
| start skin C | 25.954502 |
| idle start skin C | 26.263899 |
| max skin C | 31.198402 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | null / null / null |
| whole / active / pause mean W | 4.407829 / 4.407829 / null |
| active_s | 180.000000 |
| pause_s | 0 |
| camera_start_s | 3.268890 |
| inference_s | 173.177821 |
| camera_stop_s | 0.000000 |
| camera_off_s | 0.023846 |
| camera_on_without_inference_s | 6.798334 |
| pause_count | 0 |
| mean_pause_s | 0.000000 |
| camera_restarts | 0 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.962099 |
| memory available min MiB | 1738 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': ['power coverage'], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 0} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 140 | 147.098002 | 204.039673 |
| overall | yolo_640 | 34 | 554.808391 | 706.137980 |
| overall | m2 | 8 | 815.718527 | 861.414958 |
| overall | picker | 8 | 2095.900000 | 2271.300000 |
| first_5_active_minutes | yolo_320 | 140 | 147.098002 | 204.039673 |
| first_5_active_minutes | yolo_640 | 34 | 554.808391 | 706.137980 |
| first_5_active_minutes | m2 | 8 | 815.718527 | 861.414958 |
| first_5_active_minutes | picker | 8 | 2095.900000 | 2271.300000 |
| last_5_active_minutes | yolo_320 | 140 | 147.098002 | 204.039673 |
| last_5_active_minutes | yolo_640 | 34 | 554.808391 | 706.137980 |
| last_5_active_minutes | m2 | 8 | 815.718527 | 861.414958 |
| last_5_active_minutes | picker | 8 | 2095.900000 | 2271.300000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 97.772439 | 97.772439 | 1401.000000 | 1401.000000 |
| policy4 | 0.000000 | 0.000000 | 2348.000000 | 2348.000000 |
| policy6 | 2.793378 | 2.793378 | 2630.000000 | 2630.000000 |

YOLO due / done: `{'320': 140, '640': 34}` / `{'320': 140, '640': 34}`.

## owner_rehearsal_p23_01_fixed — fixed

Label: **NOT VALID — OWNER REHEARSAL**; checks: VALID; results: NOT VALID — OWNER REHEARSAL. [Raw invocation](runs/owner_rehearsal_p23_01.json).

| Metric | Value |
|---|---|
| duration s | 180.000000 |
| start skin C | 29.683000 |
| idle start skin C | 30.713402 |
| max skin C | 31.476803 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | null / null / null |
| whole / active / pause mean W | 2.982913 / 3.738337 / 1.472067 |
| active_s | 120.000000 |
| pause_s | 60.000000 |
| camera_start_s | 8.210276 |
| inference_s | 99.537677 |
| camera_stop_s | 14.029541 |
| camera_off_s | 45.618690 |
| camera_on_without_inference_s | 34.843633 |
| pause_count | 3 |
| mean_pause_s | 20.000000 |
| camera_restarts | 2 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.552987 |
| memory available min MiB | 1722 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 2} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 84 | 147.016011 | 170.723063 |
| overall | yolo_640 | 18 | 551.235149 | 614.554444 |
| overall | m2 | 3 | 712.912232 | 713.041545 |
| overall | picker | 3 | 1536.900000 | 1695.400000 |
| first_5_active_minutes | yolo_320 | 84 | 147.016011 | 170.723063 |
| first_5_active_minutes | yolo_640 | 18 | 551.235149 | 614.554444 |
| first_5_active_minutes | m2 | 3 | 712.912232 | 713.041545 |
| first_5_active_minutes | picker | 3 | 1536.900000 | 1695.400000 |
| last_5_active_minutes | yolo_320 | 84 | 147.016011 | 170.723063 |
| last_5_active_minutes | yolo_640 | 18 | 551.235149 | 614.554444 |
| last_5_active_minutes | m2 | 3 | 712.912232 | 713.041545 |
| last_5_active_minutes | picker | 3 | 1536.900000 | 1695.400000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 62.792573 | 89.211362 | 1401.000000 | 1401.000000 |
| policy4 | 0.000000 | 0.000000 | 2348.000000 | 2348.000000 |
| policy6 | 0.000000 | 0.000000 | 2850.000000 | 2850.000000 |

YOLO due / done: `{'320': 84, '640': 18}` / `{'320': 84, '640': 18}`.

Per-cycle table:

| Cycle | Active s | Pause s | Inference s | Duty | Skin min C | Skin max C | Camera start s | Camera stop s |
|---|---|---|---|---|---|---|---|---|
| 1 | 40.000000 | 20.000000 | 33.187857 | 0.553131 | 29.480501 | 30.693703 | 2.980179 | 4.654349 |
| 2 | 40.000000 | 20.000000 | 33.179198 | 0.552987 | 30.017002 | 31.078703 | 2.612997 | 4.674844 |
| 3 | 40.000000 | 20.000000 | 33.170623 | 0.552844 | 30.435303 | 31.476803 | 2.617100 | 4.700348 |

## owner_rehearsal_p23_01_adaptive — adaptive

Label: **NOT VALID — OWNER REHEARSAL**; checks: VALID; results: NOT VALID — OWNER REHEARSAL. [Raw invocation](runs/owner_rehearsal_p23_01.json).

| Metric | Value |
|---|---|
| duration s | 480.000000 |
| start skin C | 29.965403 |
| idle start skin C | 30.660200 |
| max skin C | 34.374900 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | null / null / null |
| whole / active / pause mean W | 2.894082 / 4.457740 / 1.274957 |
| active_s | 244.182623 |
| pause_s | 235.817377 |
| camera_start_s | 10.868028 |
| inference_s | 231.688507 |
| camera_stop_s | 18.572186 |
| camera_off_s | 216.480235 |
| camera_on_without_inference_s | 31.831258 |
| pause_count | 4 |
| mean_pause_s | 58.954344 |
| camera_restarts | 3 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.482684 |
| memory available min MiB | 1720 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 3} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 191 | 145.645671 | 202.962077 |
| overall | yolo_640 | 44 | 557.502523 | 647.893107 |
| overall | m2 | 12 | 777.048341 | 873.042115 |
| overall | picker | 12 | 1975.550000 | 2471.500000 |
| first_5_active_minutes | yolo_320 | 191 | 145.645671 | 202.962077 |
| first_5_active_minutes | yolo_640 | 44 | 557.502523 | 647.893107 |
| first_5_active_minutes | m2 | 12 | 777.048341 | 873.042115 |
| first_5_active_minutes | picker | 12 | 1975.550000 | 2471.500000 |
| last_5_active_minutes | yolo_320 | 191 | 145.645671 | 202.962077 |
| last_5_active_minutes | yolo_640 | 44 | 557.502523 | 647.893107 |
| last_5_active_minutes | m2 | 12 | 777.048341 | 873.042115 |
| last_5_active_minutes | picker | 12 | 1975.550000 | 2471.500000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 48.963429 | 92.970772 | 1401.000000 | 1401.000000 |
| policy4 | 0.000000 | 0.000000 | 2348.000000 | 2348.000000 |
| policy6 | 2.503738 | 4.921702 | 2630.000000 | 2630.000000 |

YOLO due / done: `{'320': 191, '640': 44}` / `{'320': 191, '640': 44}`.

Steady-state duty from first pause: `{'from_s': 60.421672758995555, 'duration_s': 419.57832724100444, 'inference_s': 174.52160551400448, 'fraction': 0.41594523402005895}`.

| Phase | State | Start s | Duration s | Inference s | Mean W | Skin min C | Skin max C |
|---|---|---|---|---|---|---|---|
| active_01 | active | 0.000000 | 60.421673 | 57.166901 | 4.410029 | 29.885200 | 32.512000 |
| pause_01 | pause | 60.421673 | 71.374087 | 0.000000 | 1.224146 | 30.462002 | 32.532303 |
| active_02 | active | 131.795759 | 61.151916 | 58.162547 | 4.387464 | 30.443700 | 33.175200 |
| pause_02 | pause | 192.947676 | 65.378103 | 0.000000 | 1.253987 | 31.172102 | 33.175200 |
| active_03 | active | 258.325779 | 61.186714 | 58.181947 | 4.610436 | 31.145302 | 33.905100 |
| pause_03 | pause | 319.512492 | 65.508443 | 0.000000 | 1.327539 | 31.764303 | 33.905100 |
| active_04 | active | 385.020936 | 61.422320 | 58.177111 | 4.422530 | 31.764303 | 34.355300 |
| pause_04 | pause | 446.443256 | 33.556744 | 0.000000 | 1.321234 | 33.233204 | 34.374900 |

## owner_rehearsal_p23_02_endurance — endurance

Label: **NOT VALID — OWNER REHEARSAL**; checks: VALID; results: NOT VALID — OWNER REHEARSAL. [Raw invocation](runs/owner_rehearsal_p23_02.json).

| Metric | Value |
|---|---|
| duration s | 180.000000 |
| start skin C | 24.859100 |
| idle start skin C | 25.065802 |
| max skin C | 30.713402 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | null / null / null |
| whole / active / pause mean W | 4.369746 / 4.369746 / null |
| active_s | 180.000000 |
| pause_s | 0 |
| camera_start_s | 2.730662 |
| inference_s | 173.172621 |
| camera_stop_s | 0.000000 |
| camera_off_s | 0.019151 |
| camera_on_without_inference_s | 6.808228 |
| pause_count | 0 |
| mean_pause_s | 0.000000 |
| camera_restarts | 0 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.962070 |
| memory available min MiB | 1849 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 0} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 140 | 150.311971 | 200.200846 |
| overall | yolo_640 | 34 | 572.095093 | 690.418783 |
| overall | m2 | 8 | 798.371217 | 893.905925 |
| overall | picker | 8 | 2024.000000 | 2278.700000 |
| first_5_active_minutes | yolo_320 | 140 | 150.311971 | 200.200846 |
| first_5_active_minutes | yolo_640 | 34 | 572.095093 | 690.418783 |
| first_5_active_minutes | m2 | 8 | 798.371217 | 893.905925 |
| first_5_active_minutes | picker | 8 | 2024.000000 | 2278.700000 |
| last_5_active_minutes | yolo_320 | 140 | 150.311971 | 200.200846 |
| last_5_active_minutes | yolo_640 | 34 | 572.095093 | 690.418783 |
| last_5_active_minutes | m2 | 8 | 798.371217 | 893.905925 |
| last_5_active_minutes | picker | 8 | 2024.000000 | 2278.700000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 97.782043 | 97.782043 | 1401.000000 | 1401.000000 |
| policy4 | 0.000000 | 0.000000 | 2348.000000 | 2348.000000 |
| policy6 | 2.218891 | 2.218891 | 2704.000000 | 2704.000000 |

YOLO due / done: `{'320': 140, '640': 34}` / `{'320': 140, '640': 34}`.

## owner_rehearsal_p23_02_fixed — fixed

Label: **NOT VALID — OWNER REHEARSAL**; checks: VALID; results: NOT VALID — OWNER REHEARSAL. [Raw invocation](runs/owner_rehearsal_p23_02.json).

| Metric | Value |
|---|---|
| duration s | 180.000000 |
| start skin C | 29.130602 |
| idle start skin C | 30.304000 |
| max skin C | 31.256802 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | null / null / null |
| whole / active / pause mean W | 3.049220 / 3.847224 / 1.453212 |
| active_s | 120.000000 |
| pause_s | 60.000000 |
| camera_start_s | 7.790868 |
| inference_s | 100.527841 |
| camera_stop_s | 14.442382 |
| camera_off_s | 45.105770 |
| camera_on_without_inference_s | 34.366389 |
| pause_count | 3 |
| mean_pause_s | 20.000000 |
| camera_restarts | 2 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.558488 |
| memory available min MiB | 1777 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 2} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 85 | 147.679281 | 182.308798 |
| overall | yolo_640 | 18 | 548.738038 | 602.981201 |
| overall | m2 | 3 | 718.796061 | 760.816529 |
| overall | picker | 3 | 1735.300000 | 2111.300000 |
| first_5_active_minutes | yolo_320 | 85 | 147.679281 | 182.308798 |
| first_5_active_minutes | yolo_640 | 18 | 548.738038 | 602.981201 |
| first_5_active_minutes | m2 | 3 | 718.796061 | 760.816529 |
| first_5_active_minutes | picker | 3 | 1735.300000 | 2111.300000 |
| last_5_active_minutes | yolo_320 | 85 | 147.679281 | 182.308798 |
| last_5_active_minutes | yolo_640 | 18 | 548.738038 | 602.981201 |
| last_5_active_minutes | m2 | 3 | 718.796061 | 760.816529 |
| last_5_active_minutes | picker | 3 | 1735.300000 | 2111.300000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 62.240253 | 90.042931 | 1401.000000 | 1401.000000 |
| policy4 | 0.000000 | 0.000000 | 2348.000000 | 2348.000000 |
| policy6 | 0.556440 | 0.834661 | 2802.000000 | 2802.000000 |

YOLO due / done: `{'320': 85, '640': 18}` / `{'320': 85, '640': 18}`.

Per-cycle table:

| Cycle | Active s | Pause s | Inference s | Duty | Skin min C | Skin max C | Camera start s | Camera stop s |
|---|---|---|---|---|---|---|---|---|
| 1 | 40.000000 | 20.000000 | 34.171190 | 0.569520 | 29.014902 | 30.265802 | 2.614299 | 5.076475 |
| 2 | 40.000000 | 20.000000 | 33.172646 | 0.552877 | 29.674002 | 30.837801 | 2.615000 | 4.669941 |
| 3 | 40.000000 | 20.000000 | 33.184005 | 0.553067 | 30.114302 | 31.256802 | 2.561569 | 4.695966 |

## owner_rehearsal_p23_02_adaptive — adaptive

Label: **NOT VALID — OWNER REHEARSAL**; checks: VALID; results: NOT VALID — OWNER REHEARSAL. [Raw invocation](runs/owner_rehearsal_p23_02.json).

| Metric | Value |
|---|---|
| duration s | 480.000000 |
| start skin C | 29.678102 |
| idle start skin C | 30.328302 |
| max skin C | 34.283604 |
| max Android status | 0 |
| time to 35 / 37 / 39 C s | null / null / null |
| whole / active / pause mean W | 2.876505 / 4.373016 / 1.320633 |
| active_s | 244.667327 |
| pause_s | 235.332673 |
| camera_start_s | 10.313997 |
| inference_s | 232.769505 |
| camera_stop_s | 18.696567 |
| camera_off_s | 216.522342 |
| camera_on_without_inference_s | 30.708154 |
| pause_count | 4 |
| mean_pause_s | 58.833168 |
| camera_restarts | 3 |
| camera_failures | 0 |
| capping_while_active | True |
| duty_fraction | 0.484936 |
| memory available min MiB | 1785 |
| rereads | 0 |
| thermal rereads | 0 |
| issues | {'issues': [], 'error': None, 'cleanup_errors': [], 'camera_failures': 0, 'camera_restarts': 3} |

Call latencies (M2 inference only):

| Slice | Kind | Calls | Median ms | p95 ms |
|---|---|---|---|---|
| overall | yolo_320 | 192 | 147.582072 | 208.654501 |
| overall | yolo_640 | 44 | 565.746583 | 685.547079 |
| overall | m2 | 12 | 776.494995 | 887.047323 |
| overall | picker | 12 | 1557.300000 | 2443.200000 |
| first_5_active_minutes | yolo_320 | 192 | 147.582072 | 208.654501 |
| first_5_active_minutes | yolo_640 | 44 | 565.746583 | 685.547079 |
| first_5_active_minutes | m2 | 12 | 776.494995 | 887.047323 |
| first_5_active_minutes | picker | 12 | 1557.300000 | 2443.200000 |
| last_5_active_minutes | yolo_320 | 192 | 147.582072 | 208.654501 |
| last_5_active_minutes | yolo_640 | 44 | 565.746583 | 685.547079 |
| last_5_active_minutes | m2 | 12 | 776.494995 | 887.047323 |
| last_5_active_minutes | picker | 12 | 1557.300000 | 2443.200000 |

| Policy | Whole cap % | Active cap % | Whole lowest MHz | Active lowest MHz |
|---|---|---|---|---|
| policy0 | 48.967758 | 93.224760 | 1401.000000 | 1401.000000 |
| policy4 | 0.000000 | 0.000000 | 2348.000000 | 2348.000000 |
| policy6 | 1.441875 | 2.828738 | 2630.000000 | 2630.000000 |

YOLO due / done: `{'320': 192, '640': 44}` / `{'320': 192, '640': 44}`.

Steady-state duty from first pause: `{'from_s': 60.969505401008064, 'duration_s': 419.03049459899194, 'inference_s': 174.55424312801915, 'fraction': 0.4165669214482012}`.

| Phase | State | Start s | Duration s | Inference s | Mean W | Skin min C | Skin max C |
|---|---|---|---|---|---|---|---|
| active_01 | active | 0.000000 | 60.969505 | 58.215262 | 4.387250 | 29.678102 | 32.218300 |
| pause_01 | pause | 60.969505 | 67.270401 | 0.000000 | 1.277802 | 30.186104 | 32.233902 |
| active_02 | active | 128.239907 | 61.217628 | 58.189248 | 4.461936 | 30.194502 | 33.021503 |
| pause_02 | pause | 189.457535 | 65.403510 | 0.000000 | 1.354074 | 30.980400 | 33.099400 |
| active_03 | active | 254.861045 | 61.361048 | 58.185385 | 4.311969 | 30.962803 | 33.511700 |
| pause_03 | pause | 316.222093 | 84.852317 | 0.000000 | 1.247935 | 31.445600 | 33.558100 |
| active_04 | active | 401.074411 | 61.119145 | 58.179610 | 4.331043 | 31.445600 | 34.247902 |
| pause_04 | pause | 462.193556 | 17.806444 | 0.000000 | 1.706034 | 33.807200 | 34.283604 |

## Rest gaps

| Previous | Next | Minutes | >=30 min | Basis |
|---|---|---|---|---|
| owner_endurance_p23_01 | owner_fixed_p23_01 | 14.006511 | False | Recorded device monotonic end to next idle origin. No boot ID; continuity UNVERIFIED. |
| owner_fixed_p23_01 | owner_adaptive_p23_01 | 31.415212 | True | Recorded device monotonic end to next idle origin. No boot ID; continuity UNVERIFIED. |

## Planner comparison only — every difference

These approximate hand numbers never feed generation, labels or checks. Differences below retain full precision in JSON.

| Mode | Metric | Planner | Recomputed | Difference |
|---|---|---|---|---|
| endurance | mean_w | 4.400000 | 4.399329 | -0.000671 |
| endurance | duty_fraction | 0.997000 | 0.996793 | -0.000207 |
| endurance | max_skin_c | 40.760000 | 40.757202 | -0.002798 |
| endurance | picker_calls | 89 | 89 | 0 |
| fixed | mean_w | 3.300000 | 3.303409 | 0.003409 |
| fixed | duty_fraction | 0.629000 | 0.628934 | -0.000066 |
| fixed | max_skin_c | 38.020000 | 38.022205 | 0.002205 |
| fixed | picker_calls | 60 | 60 | 0 |
| adaptive | mean_w | 3.540000 | 3.541867 | 0.001867 |
| adaptive | duty_fraction | 0.629000 | 0.629003 | 0.000003 |
| adaptive | max_skin_c | 37.430000 | 37.434700 | 0.004700 |
| adaptive | picker_calls | 70 | 70 | 0 |
| owner_fixed_p23_01 | rest_minutes | 14 | 14.006511 | 0.006511 |
| owner_adaptive_p23_01 | rest_minutes | 31 | 31.415212 | 0.415212 |
