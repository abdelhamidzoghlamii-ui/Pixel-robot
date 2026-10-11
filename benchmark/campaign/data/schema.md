# Campaign CSV schema

All CSVs use UTF-8, comma separators, a header and LF endings. Empty fields mean null, never zero. Source JSON is unchanged. Times are recorded device monotonic offsets: P23 `clock=run` uses measured mode origin (idle times can be negative); Phase 1 `clock=phase` uses each phase origin because older rest records do not record a shared origin. Do not join different phase clocks by timestamp alone.

P23 windows contain one row per 60 s measured-run window, plus separately labelled idle windows; final windows may be shorter. Adaptive windows can contain both states (`mixed`), with exact active_s/pause_s. P23 due/done and latency statistics retain archived scheduler window definitions: YOLO window assignment uses scheduled slot, M2/picker uses started_s. Calls use actual starts. Older Phase 1 window due/skipped are null because only aggregate due counts are recorded; done and latencies are recomputed from calls. Warmup is labelled NOT VALID; probe/preflight/incomplete invocations remain in cards even if no timed data exists.

Power downsampling: each phase-relative [k,k+1) second is a clipped trapezoid mean of raw current_now watts over receipt timestamps, using power.weighted. Partial final seconds use their actual duration. Gaps >1.5 s or missing brackets yield null watts, and covered_s states coverage. Skin rows retain every recorded thermal receipt in idle, gate, active and pause spans; setup and post-run cleanup are outside the dataset. Camera state is the runner-requested/checked state, not independent camera telemetry. Transitions use recorded camera_start_at/camera_off_at.

Median uses statistics.median; p95 uses nearest-rank ceil(0.95*n). M2 duration is inference_ms (not preprocessing); wall_duration_ms retains full call wall time. Nonexecuted insufficient-box M2 slots are not inference calls. No picker accuracy is claimed. Labels are never combined automatically.

## Columns

### windows.csv

| Column | Unit/type | Definition |
|---|---|---|
| `run` | text | Run-card stem; rehearsal mode suffix disambiguates shared invocations. |
| `label` | text | Original result validity; rehearsal/warmup labels override block success. |
| `phase` | text | 1, 2/3 or probe. |
| `mode` | text | endurance, fixed, adaptive or layout_screen. |
| `layout` | text | Recorded L0–L4; null when not applicable. |
| `phase_id` | text | Stable state/index within invocation; measured identifies mode-wide windows; gate is premeasurement cooling. |
| `state` | text | active, pause, idle, mixed, warmup or probe; mixed windows include state durations. |
| `start_s` | s | Window start offset in declared clock. |
| `phase_start_s` | s | Window offset in its phase; measured windows use mode origin. |
| `active_s` | s | Intersection of window with active phases. |
| `pause_s` | s | Intersection of window with pause phases. |
| `idle_s` | s | Intersection of window with idle phase. |
| `duration_s` | s | Window/power interval duration; thermal skin rows null. |
| `mean_w` | W | Exact raw-sample power.weighted mean; null if incomplete coverage. |
| `skin_min_c` | C | Minimum recorded VIRTUAL-SKIN in window. |
| `skin_max_c` | C | Maximum recorded VIRTUAL-SKIN in window. |
| `android_status_max` | enum | Largest recorded Android thermal status in window. |
| `memory_available_min_mib` | MiB | Minimum recorded MemAvailable within window. |
| `rereads` | count | Recorded empty-answer retries plus incomplete thermal retries in window. |
| `clock` | text | run or phase, as defined above. |
| `missing_reason` | text | Why older window fields are missing. |
| `policy0_cap_percent` | % | Time below recorded cpuinfo maximum; later sample owns preceding interval, no last-sample tail credit. |
| `policy0_lowest_mhz` | MHz | Lowest recorded policy maximum in window. |
| `policy4_cap_percent` | % | Time below recorded cpuinfo maximum; later sample owns preceding interval, no last-sample tail credit. |
| `policy4_lowest_mhz` | MHz | Lowest recorded policy maximum in window. |
| `policy6_cap_percent` | % | Time below recorded cpuinfo maximum; later sample owns preceding interval, no last-sample tail credit. |
| `policy6_lowest_mhz` | MHz | Lowest recorded policy maximum in window. |
| `yolo_320_due` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `yolo_320_done` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `yolo_320_skipped` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `yolo_320_median_ms` | ms | Latency statistic for named kind/YOLO size; selector means picker; M2 is inference-only. |
| `yolo_320_p95_ms` | ms | Latency statistic for named kind/YOLO size; selector means picker; M2 is inference-only. |
| `yolo_640_due` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `yolo_640_done` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `yolo_640_skipped` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `yolo_640_median_ms` | ms | Latency statistic for named kind/YOLO size; selector means picker; M2 is inference-only. |
| `yolo_640_p95_ms` | ms | Latency statistic for named kind/YOLO size; selector means picker; M2 is inference-only. |
| `m2_calls` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `m2_median_ms` | ms | Latency statistic for named kind/YOLO size; selector means picker; M2 is inference-only. |
| `selector_calls` | count | Named calls or YOLO scheduled/completed/skipped slots; selector means picker. |
| `selector_median_ms` | ms | Latency statistic for named kind/YOLO size; selector means picker; M2 is inference-only. |
| `picker_misses` | count | Picker cadence misses in window. |

### calls.csv

| Column | Unit/type | Definition |
|---|---|---|
| `run` | text | Run-card stem; rehearsal mode suffix disambiguates shared invocations. |
| `label` | text | Original result validity; rehearsal/warmup labels override block success. |
| `phase` | text | 1, 2/3 or probe. |
| `mode` | text | endurance, fixed, adaptive or layout_screen. |
| `layout` | text | Recorded L0–L4; null when not applicable. |
| `phase_id` | text | Stable state/index within invocation; measured identifies mode-wide windows; gate is premeasurement cooling. |
| `state` | text | active, pause, idle, mixed, warmup or probe; mixed windows include state durations. |
| `kind` | text | calls: YOLO/M2/picker; thermal: skin/power_1s. |
| `size` | pixels | YOLO input edge (320/640); null for M2/picker. |
| `call_index` | index | Zero-based raw call index within phase/kind, before excluding nonexecuted M2 slots. |
| `started_s` | s | Actual call start offset in declared clock. |
| `phase_started_s` | s | Actual call start offset from its phase origin. |
| `duration_ms` | ms | YOLO/picker recorded ms or M2 inference_ms; failed missing latency stays null. |
| `wall_duration_ms` | ms | (ended_s-started_s)*1000, including M2 preprocessing. |
| `skin_c` | C | Recorded VIRTUAL-SKIN reading. |
| `skin_age_s` | s | Call start minus preceding skin receipt; null if none. |
| `frequency_age_s` | s | Call start minus preceding frequency receipt; null if none. |
| `context_type` | text | live/fallback for M2 and contextual picker; none for YOLO and context-free picker. |
| `miss` | boolean | Recorded picker cadence miss, not accuracy; null for other kinds. |
| `clock` | text | run or phase, as defined above. |
| `source` | path | Raw invocation JSON basename under runs/. |
| `policy0_max_mhz` | MHz | Last recorded policy scaling maximum at or before call start; policy0 LITTLE, policy4 MID, policy6 BIG. |
| `policy4_max_mhz` | MHz | Last recorded policy scaling maximum at or before call start; policy0 LITTLE, policy4 MID, policy6 BIG. |
| `policy6_max_mhz` | MHz | Last recorded policy scaling maximum at or before call start; policy0 LITTLE, policy4 MID, policy6 BIG. |

### thermal.csv

| Column | Unit/type | Definition |
|---|---|---|
| `run` | text | Run-card stem; rehearsal mode suffix disambiguates shared invocations. |
| `label` | text | Original result validity; rehearsal/warmup labels override block success. |
| `phase` | text | 1, 2/3 or probe. |
| `mode` | text | endurance, fixed, adaptive or layout_screen. |
| `layout` | text | Recorded L0–L4; null when not applicable. |
| `phase_id` | text | Stable state/index within invocation; measured identifies mode-wide windows; gate is premeasurement cooling. |
| `state` | text | active, pause, idle, mixed, warmup or probe; mixed windows include state durations. |
| `clock` | text | run or phase, as defined above. |
| `kind` | text | calls: YOLO/M2/picker; thermal: skin/power_1s. |
| `time_s` | s | Skin receipt or power interval start offset in declared clock. |
| `duration_s` | s | Window/power interval duration; thermal skin rows null. |
| `skin_c` | C | Recorded VIRTUAL-SKIN reading. |
| `android_status` | enum | Android thermal status 0 NONE through 6 SHUTDOWN. |
| `mean_w` | W | Exact raw-sample power.weighted mean; null if incomplete coverage. |
| `covered_s` | s | Trapezoid coverage within power interval. |
| `camera_on` | boolean | Requested/checked camera state at receipt or power interval start. |
| `segment_type` | text | Heating with recorded layout, camera-stop transition, camera-OFF before start/cooling, or camera-ON idle. |

