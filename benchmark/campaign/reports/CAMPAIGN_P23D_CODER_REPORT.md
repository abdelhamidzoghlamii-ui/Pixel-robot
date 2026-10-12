# CAMPAIGN_P23D_CODER_REPORT

EXECUTION PROFILE: Coder, Codex CLI, gpt-6.1-sol, reasoning high, Ponytail lite; proot Debian. Reviewer: Claude Code headless claude-opus-5-5, medium, Ponytail off; same dontAsk restricted read-only tool configuration, stdin request, isolated exact candidate; max3 rounds. Base c0ba141dc1f6abce558f578cbae361a8d9cc16d1. Offline only; motors OFF. No hardware/root/camera/model loads/live or timed runs; no commit or push. No DOC DIFF; docs/ untouched.

## Archive and sweep

Actual per-stem counts: owner_rehearsal_p23_02=48; owner_endurance_p23_01=45; owner_fixed_p23_01=73; owner_adaptive_p23_01=78. All244 files copied unchanged; runs/archive_sha256_p23_sessions.json records source, destination, bytes and SHA256. Four new llama-server.log files were explicitly force-added and git ls-files verified (checks/p23d/force_added_logs.json), as directly requested. R1 found ignored required artifacts missing from the candidate; 14 additional task-required ignored files (datasets/inventory/recovered evidence/robotcam metadata) were force-added, with proof in force_added_required_files.json. No commit was made.

Full sweep: 2104 file decisions; {'inventory only': 693, 'already archived': 1117, 'archived into folder': 294}. Source hashes and destination matches: benchmark/campaign/checks/p23d/sweep_decisions.json. 252 raw files and 42 public-safe report/final-response files copied. Source-copy equality verified, original archived non-documentation bytes preserved; proof summary in checks/p23d/archive_integrity.json. Full source safety scan in source_safety_scan.json.

Scope: Downloads benchmark folders/review-artifact folders/loose text; home loose JSON/MD/TSV/CSV/logs and bench_results; README/RUN output roots duty_cycle,power_map,thermal_char,camera_power,coresidency,ladder,v3-runs,jevlike,qwen_vision_bench.json. Model weights, venvs, caches, review copies and Android trash skipped. ~/bench_photos does not exist. Actual ignored repo bench_photos39 files are hashed and inventory-only; existing public subsets remain unchanged. Identical ignored/untracked local v3 cases.py was not mistaken for an archived file. Patches, identity lists and requests remain outside repo with names/hashes indexed.

Qwen finding: /termux-home/qwen_vision_bench.json/results.json (5826 bytes) and summary.txt (1182 bytes) recovered unchanged into benchmark/qwen_vision/runs. Attempt2026-09-15T00:03:34Z..00:03:35Z, rows=[], battery refusal91% (required50..80), not a successful vision run. summary is an empty-result grid. /termux-home/qwen_vision_bench.out is byte-identical to that summary; qwen_vision_check.out is a recorded configuration/preflight output and inventory-only. Missing successful #105 Gemma/Qwen experiments and the stopped4B sweep remain UNVERIFIED.

## Public safety

Regex-scanned 416 candidate new/modified UTF-8 text files; new findings=0. Compared exact email/credential/phone-identifier matches with committed HEAD text; checks/p23d/public_safety_scan.json records all scanned paths/patterns/findings. No secret values are retained in scan output. Pattern scan is not proof of absence; reviewer must independently verify it. Frozen manifest and scan output are included once present; finding-only metadata avoids hash cycles.

Six external source files have new email patterns and are omitted unchanged, never silently redacted:
- /termux-home/storage/downloads/ARCHIVE1_REPORT.md; [{'kind': 'email', 'line': 526}]
- /termux-home/storage/downloads/CAMPAIGN_P1_FIX3_CODER_REPORT.md; [{'kind': 'email', 'line': 339}]
- /termux-home/storage/downloads/CODER_REPORT_archive_0928.md; [{'kind': 'email', 'line': 113}, {'kind': 'email', 'line': 288}]
- /termux-home/storage/downloads/campaign/commit_campaign_p23b.json; [{'kind': 'email', 'line': 110}]
- /termux-home/storage/downloads/campaign/review_request_campaign_p1.txt; [{'kind': 'email', 'line': 1309}]
- /termux-home/storage/downloads/campaign/review_request_campaign_p1_r2.txt; [{'kind': 'email', 'line': 1270}]

This includes CAMPAIGN_P1_FIX3_CODER_REPORT.md; its path/hash and omission are indexed, while the other public-safe campaign reports and reviewer outputs are copied unchanged. Reviewer raw JSON and final text are both preserved where present. Final current-task report and review lifecycle artifacts remain external beside earlier reports, as in the P23/P23B/P23C workflow.

## Derived reusable data and findings

33 generated outputs: 28 invocation/mode cards, three CSVs, Markdown and JSON summary. R1 publication omissions are corrected; the revised self-check requires every derived file and sweep archive destination in the frozen candidate and rejects ignored-untracked required files. calls.csv8833 rows, windows.csv317 rows, thermal.csv19503 rows at the frozen candidate. Loader/csv schema/example added. P23 due/done per valid mode match; M2 ms is inference-only, full wall ms retained. P23 windows are mode-wide60s plus labelled idle; adaptive mixed windows retain state seconds. Phase1 windows are phase-local; unavailable per-window due/skips are null with a reason. Rehearsals/warmup/incomplete/preflight evidence remain explicitly labelled. Phase1 invocations lack recorded code hashes, so commit and exact owner command remain unknown/null rather than inferred from current code. All card null paths have reasons. P23 code hashes are searched through the pinned P23D base history, ensuring later tuning commits cannot silently rewrite historical provenance.

The summary preserves all21 archived Phase1 block labels and all9 P23 mode results, including both NOT VALID rehearsals. Validity labels are not inferred from phase files that still say IN PROGRESS. Whole/active/pause mean W use existing power.weighted on raw receipt samples; no planner targets feed the calculations. First/last5 active minutes use cumulative active-phase wall time, including starts/boundary waits and excluding pauses. Raw logs remain authoritative.

| Mode | Mean W | Duty | Max skin C | Picker calls |
|---|---:|---:|---:|---:|
| endurance | 4.399329228069399 | 0.9967927570605469 | 40.757202 | 89 |
| fixed | 3.3034091444579277 | 0.6289344015671102 | 38.022205 | 60 |
| adaptive | 3.541867349720145 | 0.6290031522648524 | 37.4347 | 70 |

Rest gaps: [(14.006511110216282, False), (31.415211727933396, True)]. End to next idle origin uses recorded monotonic timestamps; no boot ID=>continuity UNVERIFIED. Fixed gap is below the owner>=30min rule. Caveats: different start skins; picker accuracy NOT EVALUATED; single runs at one ambient (ambient value absent in P23 JSON); fixed short rest. CAMPAIGN_RESULTS.json/.md reports every exact difference from planner approximations4.40/3.30/3.54W,.997/.629/.629,40.76/38.02/37.43C,89/60/70calls,~14/~31min.

Older conversion candidates, not converted: power_map/duty_cycle/camera_power (sampling/call/phase adapters and common-time coverage), coresidency/thermal_char (no campaign phases/M2 contexts), relate_anything speed1/desk2 (missing campaign YOLO/picker/windows/power schema), qwen_vision (no campaign timeseries), llm_objective_setting/strategic_selector/yolo_speed (continuous skin/caps/commonclock/power coverage UNVERIFIED). See UNARCHIVED_INVENTORY.md for exact source inventory and limits.

## Index

Extended existing benchmark/INDEX.md in its original table conventions, replaced stale campaign candidate status, added absent relate_anything/desk2/speed1/logs rows and recovered Qwen status. Rerun/raw-data information is in each folder README to keep the table readable; absent exact commands/dates/outcomes say UNVERIFIED. Campaign has README.md and ARTIFACTS.md; missing RUN_INDEX/ARTIFACTS entries supplied for indexed folders. Existing documentation updated; every historical ARTIFACTS byte is preserved as its original prefix, with a separate current inventory appended. Raw evidence is unchanged. No new catalog.

## Checks and frozen candidate

### self_check_p23

```text
PASS Part B: same final reading identity for gate and comparability; reached, hot timeout, below-band; P23 1 C / 900 s cold gate; parser failure read once
PASS real reused scheduler: SizePolicy 46/11 complete slots in 60 s (640 replaces 320; bounded terminal tail); L2 M2-before-selector; H4 one miss skips / three stop; latched controlled end
PASS SEVERE / 30 min / emergency precedence, fixed timing, adaptive hysteresis and minimum 60/30 s; burst-limited 3 re-reads per 180 s
PASS shared controller: full 30 min endurance and 36 min fixed/adaptive virtual sessions, 12 fixed cycles, camera OFF pauses/restarts, atomic per-phase files, SEVERE valid and emergency invalid with files retained
PASS 60 s aggregation: due/done/skipped, timings, receipt-weighted power, sensor/status/caps/PSS/memory/re-read counts and event detection
PASS OWNER M3: full2160s fixed grid/12cycles with3s starts+2s stops inside slots; phase/cycle/totals camera/inference/OFF fields partition wall time; duty uses inference_s; adaptive minimums include start/confirmedOFF
PASS REAL fixed rehearsal scheduler with3s starts:40s slots leave one selector each; first active forced fallback, next two liveM2; all three pauses/restarts run
main loop heartbeat 152068.1 s old: SIGTERM, normal cleanup
PASS LiveSession camera state serialized with PSS read, startup failure counted, heartbeat SIGTERM and memory failure propagate
PASS CLI mock dry run all three modes, adaptive high/low switch and pause restart; mock NOT VALID, mandatory live/current-code rehearsal proof, redirection files allowed and evidence overwrite refused
PASS power (power.sample): 8 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS fast sample (rt.fast_check over cr.fast_sample): 8 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS root mask (verify_monitor over sb.root_mask): 9 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS monitor setup (rt.prepare over sb.prepare_monitor): 9 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS pgrep (clear_processes): 8 non-empty answers read once; 4 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS battery (rt.battery_sample over sb.battery_sample): 6 non-empty answers read once; 4 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS root_sample (rt.memory_sample over cr.root_sample): 6 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS pidof (camera-OFF app absence in rt.memory_sample): 5 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS battery capacity (diagnostics.battery): 9 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS diagnostics nodes (diagnostics.snapshot): 4 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS dumpsys package (camera_version): 4 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS logcat (phase1.lmk_window over cr.lmk_lines): 8 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS settings (rt.screen over sb.Screen.setting): 4 non-empty answers read once; 4 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS dumpsys power/activity (screen_state): 5 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS am start RobotCam (camera_start over pm.camera_start): 3 non-empty answers read once; 1 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS am broadcast STOP (stop_camera): 3 non-empty answers read once; 1 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS transient mask readback (root_mask_scope): 5 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
PASS am force-stop / pidof (stop_camera over the real camera_end_check): non-empty answers read once (rc 1 with output, malformed pidof, timeout); a force-stop that did not run is re-run once with its pidof; an empty pidof is read once more; a remaining pid is re-checked once after 0.5 s; capture_stopped=False is kept
PASS /health: one failure read once more (2 of 2 rule), two fail; thermal reader identical to 5629699 (AST), a partial status-4 answer fails on the first read
PASS every wrapped reader swept (19 re-read call names): /health, RobotCam pidof after force-stop, am broadcast STOP, am force-stop, am start RobotCam, battery, battery capacity, diagnostics, dumpsys package, dumpsys power/activity, fast sample, logcat, monitor setup, pgrep, pidof, power, root mask, root_sample, settings
PASS battery (rt.battery_sample): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS memory/PSS (rt.memory_sample, cr.root_sample): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS camera-OFF pidof (rt.memory_sample): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS package version (camera_version): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS logcat (phase1.lmk_window, cr.lmk_lines): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS capacity (diagnostics.battery): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS diagnostics nodes (diagnostics.snapshot): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS screen state (screen_state): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
PASS settings (rt.screen, sb.Screen.setting): stderr-only su failure read once; blank re-read once
PASS force-stop / pidof (rt.stop_camera, real camera_end_check): stderr-only failures read once and fail; blank ones re-read once
PASS transient readback (root_mask_scope over the real cr.root): with su stderr read once; blank re-read once
PASS power (power.sample over the real RootShell): stderr-only answer read once and fails; blank re-read once
PASS mask (verify_monitor, sb.root_mask over the real RootShell): stderr-only answer read once and fails
PASS launch failures only (FIX4D R1): a local OSError after a complete battery answer and a cleanup PermissionError after an unsafe setup mask stand (1 read); su/RootShell/am that could not be started are re-read once; camera_start re-reads only a launch failure of its first `am start` (a later one fails closed)
| Finding | Probe (healthy re-read queued) | FIX4D result |
|---|---|---|
| FIX4 R1-2 | fast: battery 45.0 C next to a missing policy, healthy re-read queued | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading |
| FIX4 R1-3 | pgrep rc 0 with empty stdout (twice) | 2 reads, 1 re-read(s): ReadFailed: pgrep: Blip: process enumeration failed with no output: rc 0, '' (re-read once; first: Blip: process enumera |
| FIX4 R1-3 | pgrep rc 1 with a listed line | 1 read(s), no re-read: RuntimeError: agents/robot/other runners resident: 2001 codex |
| FIX4 R1-3 | pgrep rc 0 with a malformed line | 1 read(s), no re-read: ValueError: invalid literal for int() with base 10: 'garbage' |
| FIX4 R1-5 | screen state: su answer (1, empty), then nothing again | 2 reads, 1 re-read(s): returned {'error': 'ReadFailed: dumpsys power/activity: Blip: empty screen state answer: rc 1 (re-read once; first: Blip |
| FIX4 R1-5 | screen-timeout settings get answering nothing, twice | 2 reads, 1 re-read(s): ReadFailed: settings: Blip: empty answer to 'settings get system screen_off_timeout' (re-read once; first: Blip: empty a |
| FIX4 R2-1 | power average ['-1000000','0','Discharging','garbage'] | 1 read(s), no re-read: RuntimeError: invalid battery voltage |
| FIX4 R2-1 | power ['', '0', 'Discharging'] | 1 read(s), no re-read: ValueError: invalid literal for int() with base 10: '' |
| FIX4 R2-2 | settings get answering '' (then '') | 2 reads, 1 re-read(s): ReadFailed: settings: Blip: empty answer to 'settings get system screen_off_timeout' (re-read once; first: Blip: empty a |
| FIX4 R2-2 | screen state (1, '===\n') | 1 read(s), no re-read: returned {'rc': 1, 'error': "unparsable: '===\\n'"} |
| FIX4 R2-2 | transient readback empty actual= (twice) | 2 reads, 1 re-read(s): ReadFailed: dumpsys package: Blip: transient root shell pin/readback failed: actual='CAMPAIGN_ROOT_MASK pid=77 actual=\n |
| FIX4 R2-2 | transient readback mismatch | 1 read(s), no re-read: RuntimeError: transient root shell pin/readback failed: actual="CAMPAIGN_ROOT_MASK pid=77 actual=pid 77's current affini |
| FIX4 R2-2 | setup discover answer without the required policies | 1 read(s), no re-read: RuntimeError: missing required CPU frequency policies; not started |
| FIX4 R3-1 | fast: 7-field layout, battery 450, final policy line missing | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['50000', '51000', '52000', '450', '180 |
| FIX4 R3-2 | power ['-1000000','0'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '0'] |
| FIX4 R3-2 | power average ['-1000000','0','Discharging'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '0', 'Discharging'] |
| FIX4 R3-3 | capture not stopped, force-stop ok, pidof readback blip | 3 reads, 1 re-read(s): RuntimeError: camera cleanup: capture not shown stopped 15.1 s after STOP (151 new frames, 0 unreadable reads) |
| FIX4B R1-1 | fast: BIG 110000 with one blank policy | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading |
| FIX4B R1-1 | fast: BIG 110000, shortened answer | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['110000', '51000', '52000', '300', '18 |
| FIX4B R1-2 | su battery (0, '-1000000\n0\n') | 1 read(s), no re-read: RuntimeError: root battery read failed: rc=0, '-1000000\n0\n' |
| FIX4B R1-2 | root_sample voltage 0 with provider PSS missing | 1 read(s), no re-read: returned {'t': 151971.481562648, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {'runner': 100, 'llama_server': 100, 'robotcam_a |
| FIX4B R1-3 | capacity '101\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R1-3 | capacity '-1\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R1-4 | setup mask '3-0' plus a malformed line | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R1-4 | setup mask CPU 8 on 8 cores | 1 read(s), no re-read: RuntimeError: invalid root CPU range |
| FIX4B R1-5 | after put: readback (1, '60000') | 1 read(s), no re-read: RuntimeError: screen setting failed: 60000 |
| FIX4B R1-5 | after put: readback '60000\ngarbage' | 1 read(s), no re-read: RuntimeError: screen timeout set/readback failed |
| FIX4B R2-1 | root_sample voltage 0, complete PSS (base parser has no voltage check) | 1 read(s), no re-read: returned {'t': 151971.48881924, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {'runner': 100, 'llama_server': 100, 'robotcam_ap |
| FIX4B R2-2 | logcat ok=False, rc 1, in-window kill line | 1 read(s), no re-read: returned {'logcat_rc': 1, 'raw_head': 'logcat_rc=1\nerr\n\n', 'answer_blank': False, 'lines': ['0.5 lmkd: Kill com.examp |
| FIX4B R2-3 | power ['-1000000','4000000','123'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '4000000', '123'] |
| FIX4B R2-3 | su battery status '123' | 1 read(s), no re-read: RuntimeError: charger/battery status: 123 |
| FIX4B R2-4 | mask '0-5,' | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R2-4 | mask '3-0,' | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R3-1 | power ['-1000000','+0','Discharging'] | 1 read(s), no re-read: RuntimeError: invalid battery voltage |
| FIX4B R3-1 | capacity '+20\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R3-1 | capacity '+101\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R3-1 | root_sample voltage +0, complete PSS (base parser has no voltage check) | 1 read(s), no re-read: returned {'t': 151971.491960841, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {'runner': 100, 'llama_server': 100, 'robotcam_a |
| FIX4B R3-2 | force-stop (1, 'Error: unknown package'), then pidof readback blip | 2 read(s), no re-read: RuntimeError: camera cleanup: am force-stop failed (rc 1: 'Error: unknown package') |
| FIX4B R3-3 | logcat partial header with a kill, separator missing | 1 read(s), no re-read: returned {'logcat_rc': 1, 'raw_head': 'logcat_rc=1\n0.5 lmkd: Kill com.example\n', 'answer_blank': False, 'lines': [], ' |
| FIX4B R3-4 | fast: battery 1001, shortened answer | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['50000', '51000', '52000', '1001', '18 |
| FIX4B R3-4 | fast: CPU 200001, shortened answer | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['200001', '51000', '52000', '300', '18 |
| FIX4B R3-5 | lag mask config: '0-4,' then '0-3' | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R3-6 | power average ['-1000000','4000000','123'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '4000000', '123'] |
| FIX4B R3-6 | su battery four tokens with status '123' | 1 read(s), no re-read: RuntimeError: root battery read failed: rc=0, '-1\n4000000\n123\nx\n' |
| FIX4C R1-1 | root_sample: 200 newlines, then Charging, PSS missing (blank 200-char excerpt) | 1 read(s), no re-read: returned {'t': 151971.496706854, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {}, 'pss_pids': {}, 'answer_blank': False, 'batt |
| FIX4C R1-2 | logcat: 400 newlines, then logcat_rc=1 and a kill, no separator | 1 read(s), no re-read: returned {'logcat_rc': 1, 'raw_head': '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\ |
| FIX4C R1-2 | logcat '=== lmk lines\ngarbage\n' | 1 read(s), no re-read: returned {'logcat_rc': None, 'raw_head': '', 'answer_blank': False, 'lines': [], 'n_lines': 0, 'n_kills': 0, 'outside_bl |
| FIX4C R1-3 | force-stop ok, pidof (1, 'su: error 123\npidof_rc=2\n') | 2 read(s), no re-read: RuntimeError: camera cleanup: RobotCam still running after force-stop (pids ['123']) |
| FIX4C R1-4 | readback rc 98: empty marker line, then a different mask line | 1 read(s), no re-read: RuntimeError: transient root shell pin/readback failed: actual="CAMPAIGN_ROOT_MASK pid=77 actual=\npid 77's current affi |
| FIX4C R2-1 | force-stop (1, ''), pidof (0, 'pidof_rc=2\n'), re-run force-stop (0, ''), healthy pidof queued | 3 reads, 1 re-read(s): RuntimeError: camera cleanup: RobotCam absence after force-stop unconfirmed (root rc 0, pidof rc 2) |
| FIX4C R1-5 | pgrep rc 2, stdout empty, stderr 'pgrep: permission denied' | 1 read(s), no re-read: RuntimeError: process enumeration failed: pgrep: permission denied |
| FIX4C R4-1 | force-stop rc 1, stdout '', stderr 'su: permission denied'; pidof ok; healthy force-stop queued | 2 read(s), no re-read: RuntimeError: camera cleanup: am force-stop failed (rc 1: '') |
| FIX4C R4-1 | pidof rc 1, stdout '', stderr 'su: permission denied'; healthy pidof queued | 2 read(s), no re-read: RuntimeError: camera cleanup: RobotCam absence after force-stop unconfirmed (root rc 1, pidof rc None) |
| FIX4C R4-1 | battery: the same stderr-only failure; healthy answer queued | 1 read(s), no re-read: RuntimeError: root battery read failed: rc=1, '' |
| FIX4C R4-1 | package version: the same stderr-only failure; healthy answer queued | 1 read(s), no re-read: RuntimeError: installed RobotCam version unreadable |
| FIX4D R1-2 | setup: mask 4-7 (+ su stderr) with measured {4,5}, then the cleanup kill raises PermissionError; healthy setup queued | 1 read(s), no re-read: PermissionError: kill |
| FIX4D R1-1 | battery: complete 2 W answer + su stderr, then meminfo raises OSError; healthy 4 W queued | 1 read(s), no re-read: OSError: meminfo unreadable |
PASS every earlier probe (FIX4, FIX4B, FIX4C R1-R2 from self_check_fix4c, FIX4C R4) on this code; FIX4C R3 had no findings (sandbox blocker)
PASS FIX4D sweep extended to P23: every hardware reader reused unchanged; stdout AND su stderr rule, launch-only, HTTP timeout, /health 2-of-2, FIX3D thermal rule; no new transport/parser/re-read call
PASS slow640 1.01/1.5/1.9s without lost frames remains cadence-valid; terminal budget only
PASS review regressions: REAL LiveSession.cycle/run_cycle fractional tails and 1800.3 s / 1.0007 s camera drift; stop predicate main-only; missing selector remains invalid; post-gate LMK window; evidence interrupt propagates
PASS R2: complete-slot mock/live counts,18 proof hashes,bracket emergency stop reason,incremental work checkpoints,recovered launch failure count,adaptive restart reserve,main-thread drain heartbeat,BatteryStop worker type
PASS P23B-R2: fixed/adaptive measured FULL controller outputs and all JSON bytes identical to 49bc13b; boundary decisions/thresholds unchanged; overshoot rehearsal trigger40->low38 vs measured high37->low35; per-pause coverage
Adaptive feasibility ESTIMATE from shifted curves; NOT VALID as a hardware result.
Step-held dumps, first dump at t=0; G=30 C; mock 3 s start / 2 s stop; OFF curve begins at confirmed OFF, shifted to high-switch skin; no extrapolation. Archived heating starts with camera already on.
ASSERT only L2 heating x six pauses, new rule, 480 s. All other matrix rows REPORT ONLY.
| Heating | Pause | Rule | Window s | High-switch s | Low-restart s | Remaining s | Result |
|---|---|---|---:|---:|---:|---:|---|
| block_01_L0 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_01_L0 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_01_L0 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_01_L0 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_01_L0 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_01_L0 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_03_L2 | old G-1 | 480 | 60.0 | 247.9 | 232.1 | PASS |
| block_01_L0 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_01_L0 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_01_L0 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_04_L3 | old G-1 | 480 | 60.0 | 330.7 | 149.3 | PASS |
| block_01_L0 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_01_L0 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_01_L0 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_05_L4 | old G-1 | 480 | 60.0 | 257.7 | 222.3 | PASS |
| block_01_L0 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_01_L0 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_01_L0 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_06_L0 | old G-1 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| block_02_L1 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_02_L1 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_02_L1 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_02_L1 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_02_L1 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_02_L1 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_03_L2 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_02_L1 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_02_L1 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_04_L3 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_02_L1 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_02_L1 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_05_L4 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_02_L1 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_02_L1 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_06_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_03_L2 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_03_L2 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_03_L2 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_03_L2 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_03_L2 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_03_L2 | old G-1 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| block_03_L2 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_03_L2 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_03_L2 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_04_L3 | old G-1 | 480 | 60.0 | 403.7 | 76.3 | PASS |
| block_03_L2 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_03_L2 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_03_L2 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_05_L4 | old G-1 | 480 | 60.0 | 311.3 | 168.7 | PASS |
| block_03_L2 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_03_L2 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_03_L2 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_06_L0 | old G-1 | 480 | 60.0 | 389.1 | 90.9 | PASS |
| block_04_L3 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_04_L3 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_04_L3 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_04_L3 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_04_L3 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_04_L3 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_03_L2 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_04_L3 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_04_L3 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_04_L3 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_04_L3 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_04_L3 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_05_L4 | old G-1 | 480 | 60.0 | 418.5 | 61.5 | PASS |
| block_04_L3 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_04_L3 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_04_L3 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_06_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_05_L4 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_05_L4 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_05_L4 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_05_L4 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_05_L4 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_03_L2 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_05_L4 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_05_L4 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_04_L3 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_05_L4 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_05_L4 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_05_L4 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_05_L4 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_05_L4 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_06_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_06_L0 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_06_L0 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_06_L0 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_06_L0 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_06_L0 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_03_L2 | old G-1 | 480 | 60.0 | 242.9 | 237.1 | PASS |
| block_06_L0 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_06_L0 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_06_L0 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_04_L3 | old G-1 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| block_06_L0 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_06_L0 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_06_L0 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_05_L4 | old G-1 | 480 | 60.0 | 247.9 | 232.1 | PASS |
| block_06_L0 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_06_L0 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_06_L0 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_06_L0 | old G-1 | 480 | 60.0 | 306.3 | 173.7 | PASS |
MATRIX REPORT ONLY: trigger-2, 240 s: 30/36 restart; worst successful restart 165.1399999999942 s
MATRIX REPORT ONLY: trigger-2, 480 s: 36/36 restart; worst successful restart 184.73999999999307 s
MATRIX REPORT ONLY: old G-1, 240 s: 0/36 restart; worst successful restart None s
MATRIX REPORT ONLY: old G-1, 480 s: 13/36 restart; worst successful restart 418.5399999999819 s
PASS shifted-curve feasibility: REAL run_mode/switch and restart reserve; all six L2 480 s estimates restart with >=61 s left; remaining matrix report only
PASS CAMPAIGN_P23 self-check (offline mocks only; agents resident; proot; NOT VALID for timing)

/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
usage: self_check_p23.py [-h] [--mode {endurance,fixed,adaptive}]
                         [--t-hi T_HI] [--dry-run | --preflight] [--mock]
                         [--rehearsal REHEARSAL] --output OUTPUT
self_check_p23.py: error: evidence exists; choose a fresh stem

```
### self_check_p23c

```text
PASS start-up gap 0.37 settle_s 3.3499999999998096 first idle full coverage
PASS start-up gap 2.47 settle_s 5.8499999999996675 first idle full coverage
PASS start-up gap 2.874435548 settle_s 6.250000000363798 first idle full coverage
PASS steady cadence: no delay beyond tenth receipt + <=50 ms polling; stale/absent/slow/short streams timeout at 60 s
PASS main: 60 s timeout is setup failure, no idle/active measurement, cleanup, NOT VALID
PASS differential d5d1e87: all three modes FULL and SHORT, run objects and every phase/checkpoint JSON byte identical at common origin; setup gate executed
PASS exact setup mark: retries after snapshot and during settle retained; prior invocation excluded; cap4>3 preserved
PASS CAMPAIGN_P23C (offline mocks only; NOT VALID for hardware timing)

/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(

```
### self_check_p23d

```text
PASS P23D: 33 byte-identical derived files; 8833 calls; 317 windows; 3954 artifact entries; cards, validity/counts, phase totals, durations, preceding samples, power coverage.


```

Initial Debian attempts for the existing P23/P23C checks failed because numpy is absent; full *.debian.* logs retained. Existing checks passed with installed native Termux Python, still offline mocks only. No dependency installed. R1 found 14 required ignored files missing from publication; those files are now force-added, included in the frozen candidate, and independently cross-checked against the current inventory. Round2 preparation also caught an overly strict check for unchanged tracked metadata and a failed adjacent-inventory formatter; both are fixed, with diagnostics retained. An initial new-check failure correctly identified external omission rows in a historical ARTIFACTS table; check now validates archive-relative entries and leaves those historical external-source omission records separate. Initial logs retained.

Frozen manifest: benchmark/campaign/frozen_manifest_campaign_p23d.json, SHA256 84a552a3cbe34498b07f8c7ebb8b36e81571b4fe3b80ee505f0027172e2ad043; 415 candidate files. New-file bytes 45123008; net byte growth 45456163 (manifest itself excluded from those totals). Largest candidate file: (4245630, 'benchmark/campaign/runs/owner_adaptive_p23_01.json'). No single added file exceeds50MB. Final totals including manifest: 45204818 new bytes; 45537973 net growth.

## Independent review round 1 — final response verbatim

REVIEW_BEGIN
# CAMPAIGN_P23D review, round 1 — CHANGES REQUESTED

The archive copy, run cards and headline numbers are right: I recomputed all of them from raw JSON and they match. But the frozen candidate is missing files that its own code, docs and self-check depend on. Committed as frozen, the archive would not reproduce on a fresh checkout and would contain broken links.

## Findings

### Blocking

**B1. The CSV datasets and the unarchived-files inventory are not in the candidate.**
- **What is missing:** `benchmark/campaign/data/{windows,calls,thermal}.csv` are ignored by `.gitignore:12` (`*.csv`). `benchmark/UNARCHIVED_INVENTORY.md` is ignored by `.gitignore:8` (`benchmark/*`).
- **Where it shows:** none of the four files is in `frozen_manifest_campaign_p23d.json` or the review patch. That patch has 401 diffs over the same 396 files plus the manifest. None is in the live git index either, though all four exist in the live working tree (calls.csv is 2,338,048 bytes, thermal.csv 2,944,216, windows.csv 112,553, the inventory 177,206).
- **What breaks on a clean checkout:**
  - The loaders `campaign_data.py:32-46` (`iter_windows`, `iter_calls`, `iter_thermal`).
  - The example in `data/README.md:13`.
  - `self_check_p23d.py:16-18`, which compares the rebuild with the files on disk.
  - The links in `data/README.md:5`, `campaign/README.md:7` and `UNARCHIVED_INVENTORY`.
- **Why the check passed:** the "33 byte-identical derived files" PASS only holds in the coder's working tree. No manifest records a hash for the CSVs.
- **Fix:** force-add these four files (as was done for the `.log` files), add them to `ARTIFACTS.md` and the frozen manifest, and re-freeze.

**B2. Eight "archived into folder" decisions point to files that are not in the candidate.**
- **The eight destinations:**
  - `benchmark/coresidency/runs/block_B1_mix_nollm.json`, `block_B2_only640_nollm.json`, `block_B3_mix_gemma.json`
  - `benchmark/coresidency/runs/coresidency_run_20260930T150019Z_smoke_report.txt` and `coresidency_run_20260930T182618Z_report.txt`
  - `benchmark/camera_heat/runs/camera_heat_active.json`
  - `benchmark/qwen_vision/runs/results.json` and `summary.txt`
- **Also affected:** the "already archived" decision for `/termux-home/qwen_vision_bench.out` points to that same untracked `summary.txt`.
- **Empty inventory tables:** the "P23D additions" tables in `coresidency/ARTIFACTS.md`, `camera_heat/ARTIFACTS.md` and `qwen_vision/ARTIFACTS.md` contain only a header row, in both the candidate and the live repo.
- **Broken links:** `benchmark/INDEX.md` (qwen row) and the "P23D recovery" section of `qwen_vision/README.md` link to the missing files. The README also says "hashes are in ARTIFACTS.md", which is false.
- **Why the check missed it:** `self_check_p23d.py:66-67` skips absolute-path rows and never cross-checks sweep destinations against the tree.

### Medium

- **M1. Historical hash records were rewritten in `coresidency/ARTIFACTS.md`.**
  - The rows for `coresidency.py` (67834 → 69973 bytes) and `test_coresidency.py` (49281 → 49598) are edited in the table and in the sha256sums block.
  - That section says the hashes "equal CLEANUP1_REPORT.md". The edit makes that claim false, and P23D did not touch either file.
  - This edits existing archive evidence. Add a separate current-hash note instead.
  - The `README.md` rows in the "before this task" sections of `camera_heat` and `coresidency` were edited the same way (lower severity).
- **M2. The qwen row in `benchmark/INDEX.md` overstates the result.**
  - It says "Recovered raw results … partial results recovered", but the raw file has `rows=[]` and a battery-precondition refusal at 91%.
  - The README states this correctly. The INDEX outcome should say "no successful results recovered".

### Minor

- **m3. Matching by content hash alone produces misleading "already archived" decisions.**
  - 118 empty source files are marked "already archived". For example, `review_campaign_p23_r2.stderr` "matches" 178 unrelated stderr files.
  - 31 more files under 64 bytes match the same way.
  - Empty or trivial files should get their own decision, and a destination should be required to be tracked in the repo.
- **m4. `matching_code` (`campaign_data.py:234-254`) returns the newest commit whose files match, not the commit the run executed.**
  - So all P23 cards show the base `c0ba141`.
  - Label it "newest matching commit".
- **m5. Phase 1 cards leave `exact_owner_command` null, although RUN.md has the exact command.**
  - For example, the `owner_session_p1_fix4d` command is at `RUN.md:177`.
  - This is acceptable under the stated known limit, but the card could cite that command marked UNVERIFIED.
- **m6. `reports/INDEX.md` has weak Task and Date columns.** For review rows the Task column holds filenames (e.g. `p1_fix2_r1.json`), and every Date is UNKNOWN.
- **m7. Some loose benchmark code in Downloads has no sweep decision or skip reason:** `powermap_power_map.py`, `powermap_oneshot.sh`, `oneshot.sh`. Personal documents, APKs and images were rightly skipped, but the inventory's skip list doesn't name those categories.
- **m8. Per-mode rehearsal cards are inaccurate.** They carry `t_hi_c` for all three modes and the parent invocation's command.

## Independent computations (my own code, raw JSON only)

None of the new scripts were imported or run. Power is my own trapezoid integration matching `power.py:45-64`: clip to the window, skip gaps over 1.5 s, no extrapolation. Windows run from measured origin to origin + duration.

| Mode | Mean W whole / active / pause | Coverage, gaps | Inference duty (inference_s) | Max skin °C | YOLO 320/640 due = done, skipped | Picker calls / misses | M2 executed |
|---|---|---|---|---|---|---|---|
| endurance | 4.399329 / 4.399329 / – | 1800.0 s, 0 | 0.996793 (1794.226963) | 40.757202 | 1436/358 = 1436/358, 0 | 89 / 0 | 89 |
| fixed | 3.303409 / 4.294048 / 1.322132 | 2160.0 s, 0 | 0.628934 (1358.498307) | 38.022205 | 1104/264 = 1104/264, 0 | 60 / 0 | 60 |
| adaptive | 3.541867 / 4.724322 / 1.348299 | 2160.0 s, 0 | 0.629003 (1358.646809) | 37.434700 | 1105/264 = 1105/264, 0 | 70 / 0 | 70 |

- **Skin crossings, first receipt ≥ threshold (35 / 37 / 39 °C):**
  - endurance: 519.237 / 675.093 / 996.522 s
  - fixed: 451.474 / 816.719 s / none
  - adaptive: 417.072 / 572.916 s / none
- **Max Android status:** 1 / 0 / 0 for endurance / fixed / adaptive.
- **Rest gaps:** 14.006511 min (endurance → fixed) and 31.415212 min (fixed → adaptive).

All of these match `CAMPAIGN_RESULTS.json`/`.md` and the README.

## Spot checks

**Run cards**
- `owner_fixed_p23_01` (P23, VALID) is correct:
  - all 73 evidence files match by bytes and SHA256;
  - recorded fields match the raw invocation: llama-server command, model hashes, M2 cluster [6,7]×2, monitor mask, ORT mid 2/2, battery 81%, gate skin 27.3903, `t_hi_c` 37;
  - the schedule (FULL) and the exact owner command line from RUN.md are right.
- `owner_session_p1_fix4d` (Phase 1) is correct:
  - all 20 evidence files match by hash;
  - block labels match raw: L0 VALID, L1 VALID, L2 NOT COMPARABLE, L3 VALID, L4 NOT COMPARABLE, final L0 NOT COMPARABLE;
  - battery 80%, server command identical, no `code_hashes`, so commit unknown and command null with a reason.
- `owner_rehearsal_p23_02_adaptive` is correct apart from m8: 48 evidence files OK, SHORT schedule, `--dry-run` command matches `RUN.md:382`, labelled NOT VALID.

**calls.csv rows**
- **Source:** the live-repo `calls.csv`, read-only, since it isn't in the candidate. It has 8833 rows.
- **Main sample:** seed 20261011, `random.Random(seed).sample(range(8833), 20)`. Row indexes: 59, 239, 431, 609, 655, 898, 2100, 3558, 3771, 3965, 4570, 5537, 5657, 5678, 5867, 7734, 7863, 8058, 8122, 8694.
- **Extra sample:** seed 7, 10 M2/picker rows: 620, 853, 1052, 3317, 3722, 4221, 4593, 6176, 6779, 7642.
- **Raw sources covered:** `owner_{adaptive,endurance,fixed}_p23_01.json`, `owner_rehearsal_p23_0{1,2}.json`, `owner_session_p1_fix4d.json`, `owner_session_p1a_fix.json`.
- **Result:** all 30 rows match on every field I checked:
  - absolute start = phase `inference_origin` (P23) or cycle origin (Phase 1) + `started_s`;
  - run-clock offset and phase offset;
  - the preceding fast/dump sample (t ≤ start), skin value and age, frequency age;
  - policy0/4/6 maximum MHz;
  - M2 `inference_ms` versus wall time (for example row 853: 758.14 vs 772.39 ms);
  - live/fallback/none context, and the miss flag.

**Sweep decisions** (seed 4242, stratified)
- **Already archived:**
  - `downloads/campaign/review_campaign_p23_r2.stderr` — empty file, see m3
  - `ladder/camera_heat_20260929T022511Z_16984/response_1.json`
  - `camera_power/run_20261003T062831Z_smoke/baseline_first_end.jpg`
  - `downloads/campaign/owner_session_p1_fix2.stderr`
- **Archived into folder:** `owner_fixed_p23_01_fixed_samples_0025.json`, `owner_fixed_p23_01_fixed_pause_07.json`, `owner_fixed_p23_01.json`.
- **Inventory only:** `p23b_round1_checks/coder_mock_p23b_adaptive_idle.json`, `coresidency_review_artifacts/review16/stderr.txt`, `coresidency_review_artifacts/review13/request.md`.
- **Result:** source hashes and destination hashes all match.
- **Full pass over all 2101 decisions:** 286 archive destinations OK; 8 archive destinations missing (B2); 1 "already archived" destination missing (qwen).
- **Per-stem counts:** 48 / 45 / 73 / 78 = 244, all byte-identical to the Downloads source.
- **Coverage of required Downloads folders:** every file in campaign, relate_speed1/fix1/desk3, thermal_char_run_*, power_map_*, coresidency_*, robotcam and *_review_artifacts has a decision. The only exceptions are the current P23D review files, created after the sweep.

## Public safety

- **Candidate scan:** I ran the 6 provided patterns over all 396 candidate files plus the frozen manifest. One email value matches, in `reports/CAMPAIGN_P1_FIX_CODER_REPORT.md` at lines 336–337. The same value already appears in two unchanged committed reports, `camera_power/reports/CAMPOWER{1,2}_CODER_REPORT.md`, so it is allowlisted. No new findings.
- **Live-only files:** the 12 files from B1/B2 (CSVs, inventory, the 8 archive files) have zero matches.
- **Omitted external sources:** emails not in committed text appear at:
  - `ARCHIVE1_REPORT.md:526`
  - `CAMPAIGN_P1_FIX3_CODER_REPORT.md:339` — this file was correctly omitted and listed, not redacted
  - `CODER_REPORT_archive_0928.md:113,288`
  - `commit_campaign_p23b.json:110`
  - `review_request_campaign_p1.txt:1309`
  - `review_request_campaign_p1_r2.txt:1270`

  None of them was copied.
- **Report coverage:** all other CAMPAIGN coder reports and public review final texts in Downloads were copied. AUDIT1 was already archived, and P23D's own report stays external.
- **Limits:**
  - A whole-repo allowlist walk and a walk of the home-directory roots were refused by the tool permissions. I used a targeted allowlist check and walked only the Downloads folders instead.
  - The scan is regex-only, binary files are not scanned, and I could not run git, so I could not confirm which matching files are tracked.

## Checks I could confirm

- Frozen manifest: all 396 files match by bytes and hash; total 39,810,176 bytes; nothing over 50 MB.
- `self_check_p23`, `p23c` and `p23d` outputs are as recorded. The initial `p23d` failure on an absolute path was fixed by skipping those rows, which is part of why B2 slipped through.
- The `.log` files are in the live index.

I didn't run the self-checks myself; per the instructions, I didn't import or execute the new scripts.

**Verdict: CHANGES REQUESTED** (round 1). Blocking: B1 and B2. Should be fixed: M1 and M2. This is a review finding only, not permission to commit.
REVIEW_END

## Identity proof round 1

```json
{
  "command": "cd /termux-home/review_campaign_p23d_r1 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete independent review instructions are on stdin. Ponytail OFF. Follow them and return a final verdict.' --model claude-opus-5-5 --effort medium --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{\"mcpServers\":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23d_r1.txt\n",
  "copy_byte_identity": true,
  "date": "2026-10-11T16:50:44.062697+00:00",
  "effort": "medium",
  "exit": 0,
  "live_byte_identity": true,
  "model": "claude-opus-5-5",
  "ponytail": "off"
}

```

/termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r1_before.json SHA256 e941d6e81514194bf14536edf4ba713504886bea50ebb415ee6da329fffbb17c
copy: 3086 files; canonical inventory SHA256 c73d8cd66ecd1173ebe2adde301609ead57025bc92aed1faea8d9d694202ab13
live: 4963 files; canonical inventory SHA256 6869fba342855ce085e873de69b2e67608905b3fc532c106b41a05637a2a50a6
/termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r1_after.json SHA256 e941d6e81514194bf14536edf4ba713504886bea50ebb415ee6da329fffbb17c
copy: 3086 files; canonical inventory SHA256 c73d8cd66ecd1173ebe2adde301609ead57025bc92aed1faea8d9d694202ab13
live: 4963 files; canonical inventory SHA256 6869fba342855ce085e873de69b2e67608905b3fc532c106b41a05637a2a50a6

Review-copy/patch/identity/request files remain external:
- /termux-home/storage/downloads/campaign/review_campaign_p23d_r1.json; bytes 19841; SHA256 81d12a2cde8ca4a6dc8f585e71807353da1f056a7d65a2d9cbead579c10c91b6
- /termux-home/storage/downloads/campaign/review_campaign_p23d_r1.stderr; bytes 0; SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- /termux-home/storage/downloads/campaign/review_campaign_p23d_r1_result.txt; bytes 11749; SHA256 c0b92ad7462dee7a3be29a9c024beb4bba37087562896d01b493d43183040162
- /termux-home/storage/downloads/campaign/review_command_campaign_p23d_r1.txt; bytes 668; SHA256 95ce019d864d75e7914caa025db05d9842ad696a36bf597d33e4e17d844efc0f
- /termux-home/storage/downloads/campaign/review_full_diff_campaign_p23d_r1.patch; bytes 41077276; SHA256 4a27b8ae539ef37a0d4b72cb7e87be93dca70056f0c1177570f1c4f65101690e
- /termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r1_after.json; bytes 1584948; SHA256 e941d6e81514194bf14536edf4ba713504886bea50ebb415ee6da329fffbb17c
- /termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r1_before.json; bytes 1584948; SHA256 e941d6e81514194bf14536edf4ba713504886bea50ebb415ee6da329fffbb17c
- /termux-home/storage/downloads/campaign/review_request_campaign_p23d_r1.txt; bytes 40323; SHA256 29aa6cad2b909dca3354e3860f2791655830463a7883e32c9406c581c461de34
- /termux-home/storage/downloads/campaign/review_status_campaign_p23d_r1.json; bytes 883; SHA256 13702c12fb3bc6dbdba0dd13551993c24395f59b4a591dabb3a15a4685880cac

## Independent review round 2 — final response verbatim

REVIEW_BEGIN
# CAMPAIGN_P23D review, round 2: APPROVED

Both round-1 blockers are fixed in the frozen candidate. The four ignored dataset/inventory files and the eight recovered raw files are now in the candidate, the manifest and the live git index. The two round-1 medium findings are also fixed. I recomputed every headline number from raw JSON with my own code, and all of them match. What remains are minor wording and link problems that don't block.

## Findings

### Round-1 items: status

- **B1 — fixed.** `data/{calls,windows,thermal}.csv` and `benchmark/UNARCHIVED_INVENTORY.md` are now in `frozen_manifest_campaign_p23d.json`.
  - I parsed the live `.git/index` read-only. All 41 manifest files that `.gitignore` would exclude are in the index.
  - For all 18 force-added files, the staged blob SHA-1 equals the candidate bytes.
- **B2 — fixed.** All 294 "archived into folder" destinations exist in the candidate, are in the manifest, and match their source SHA-256. This includes the coresidency, camera_heat and qwen files.
  - All 1117 "already archived" decisions have at least one tracked destination with the same hash.
  - The "P23D current inventory" tables are no longer empty.
- **M1 — fixed.** In the review patch, every pre-existing ARTIFACTS.md and README.md receives added lines only and loses none. For example, `coresidency/ARTIFACTS.md` gets +46/−0. The historical sha256 blocks are byte-preserved, and current hashes now sit in a separate "P23D current inventory" section. The only file that loses lines is `benchmark/INDEX.md` (two rows rewritten; see n2).
- **M2 — fixed in the outcome column.** `benchmark/INDEX.md:25` now says "no successful results recovered" and names the battery-precondition refusal.
- **m3–m8 — fixed, or kept with a stated reason:**
  - m3: all 149 files under 64 bytes carry the note "does not establish run identity or provenance".
  - m4: cards now say "newest matching snapshot".
  - m5: Phase 1 owner commands stay null with an explicit reason. This was accepted as a known limit.
  - m6: the Task column now holds canonical task names.
  - m7: the three loose scripts now have decisions (already archived), and `UNARCHIVED_INVENTORY.md:715` names the skipped categories.
  - m8: child rehearsal cards now carry only their own mode's values, plus `parent_owner_command`.

### New minor findings (non-blocking)

- **n1. Two broken links.** The added rerun paragraph links `[runs/](runs/)`, but these folders have no `runs/` directory:
  - `benchmark/llm_objective_setting/conversation/conv_mtp/README.md:72`
  - `benchmark/strategic_selector/von12_rerun/README.md:58`

  All 403 relative links in new or changed Markdown outside `reports/` resolve except these two. The broken links inside `reports/` are in verbatim historical reports and are expected.
- **n2. Leftover "partial" wording for Qwen.**
  - The INDEX description still says "harness and partial recovered results" (`benchmark/INDEX.md:25`).
  - `qwen_vision/RUN_INDEX.md:17` says "These are partial #105 evidence".
  - Since `rows=[]`, these should say "aborted attempt".
  - The rewritten INDEX row also dropped the original date "2026-09-14/16" and the list of archived items (projector report, AGY review, probe). `qwen_vision/README.md` still records them.
- **n3. Generic rerun text even where scripts exist.** The repeated text "Rerun command: UNVERIFIED: no single command identified in [existing instructions](README.md)" links each README to itself. It also appears in folders that do contain rerun scripts, such as `coresidency/run_coresidency.sh`. This is conservative, not false.
- **n4. Commit hygiene (process only).** 282 new files that aren't ignored are not yet staged. That is expected before a "yes commit", but the commit needs to add exactly the manifest list. `self_check_p23d.py:91-95` only checks membership in the frozen manifest or in the tracked set.

## Independent computations (my own code, raw JSON only)

I did not import or run `campaign_data.py`, `summarize_campaign.py` or `self_check_p23d.py`. Power uses my own in-memory trapezoid integration with the semantics of `power.py:45-64`: clipped to the window, gaps over 1.5 s skipped, no extrapolation. Windows run from `run.origin` to `run.end`.

| Mode (label) | Mean W whole / active / pause | Coverage, gaps | Inference s, duty | Max skin °C, max status | YOLO due 320/640 = done, skipped | Picker calls / misses | M2 executed |
|---|---|---|---|---|---|---|---|
| endurance (VALID) | 4.399329228069399 / 4.399329228069399 / – | 1800.0 s, 0 | 1794.2269627089845, 0.9967927570605469 | 40.757202, 1 | 1436/358 = 1436/358, 0 | 89 / 0 | 89 |
| fixed (VALID) | 3.3034091444579277 / 4.294047793525703 / 1.322131846322393 | 2160.0 s, 0 | 1358.498307384958, 0.6289344015671102 | 38.022205, 0 | 1104/264 = 1104/264, 0 | 60 / 0 | 60 |
| adaptive (VALID) | 3.541867349720145 / 4.724321938096641 / 1.3482986213311676 | 2160.0 s, 0 | 1358.646808892081, 0.6290031522648524 | 37.4347, 0 | 1105/264 = 1105/264, 0 | 70 / 0 | 70 |

- **First skin reading at or above 35 / 37 / 39 °C:**
  - endurance: 519.2366 / 675.0925 / 996.5217 s
  - fixed: 451.4742 / 816.7187 s / none
  - adaptive: 417.0715 / 572.9157 s / none
- **Rest gaps:** 14.006511 min (endurance → fixed, below the 30 min rule) and 31.415212 min (fixed → adaptive).
- **Match:** all of these equal `CAMPAIGN_RESULTS.json` and `.md`, `README.md` and `RUN_INDEX.md`.
- **Windows:** the measured windows reproduce the mode totals. The duration-weighted mean of window W gives 4.399329 / 3.303409 / 3.541867 W. Window counts are 30, 36 and 36, and the window due, done, M2, picker and miss counts sum to the figures above.
- **Rows I integrated myself:** four random adaptive windows (seed 31: indexes 0, 7, 25, 30, including two mixed windows) and three random 1 s `thermal.csv` power rows. W, max skin and active_s all match to within 1e-8.

## Spot checks

### Run cards (3)

Evidence for each card was checked by bytes and SHA-256.

- **`owner_adaptive_p23_01` (P23, VALID): correct.** 78/78 evidence files OK.
  - These match raw: code_hashes, model hashes, server command, `m2_cluster` [6,7]×2, monitor mask [0–3], ORT mid, battery 81%, gate skin 24.5993, idle start skin 25.3564, `t_hi_c` 37.0.
  - FULL schedule taken from phase23.py at c0ba141. Exact owner command equals `RUN.md:447`. Rest gap 1884.91 s.
- **`owner_session_p1a_fix` (Phase 1): correct.** 7/7 evidence files OK.
  - Label "SESSION COMPLETE; inspect individual validity"; result validity per block is L0 VALID, L1 and L2 NOT VALID — WARM START.
  - Server command and model hashes are identical to raw. RobotCam version 0.1.
  - No code_hashes, so the commit is unknown and the command, battery and schedule are null, each with a reason.
- **`owner_rehearsal_p23_01_fixed` (child rehearsal card): correct.** 48/48 evidence files OK.
  - Labelled NOT VALID — OWNER REHEARSAL, with child checks VALID.
  - `t_hi_c` [37.0], gate skin 29.683 and idle start skin 30.713402 are the fixed-mode values. Battery 86%.
  - Commit d5d1e87 with its SHORT schedule. `exact_owner_command` is null with a reason, and `parent_owner_command` holds the `--dry-run` line.
- **Coverage:** all 22 invocations that have a stdout file have a card. The 6 extra cards are the per-mode rehearsal children.

### calls.csv rows

- **Main sample:** seed 20261013, `random.Random(seed).sample(range(8833), 20)`.
  - Row indexes: 306, 764, 874, 1206, 1588, 1711, 2112, 2280, 2508, 3168, 3291, 3624, 3883, 4094, 5372, 6798, 7402, 7489, 7759, 8018.
  - Raw sources: `owner_{adaptive,endurance,fixed}_p23_01.json`, `owner_rehearsal_p1_fix4d.json`, `owner_session_p1_fix4d.json`.
- **Extra sample:** seed 99, 10 M2/picker rows drawn from 777: 985, 1505, 3346, 3367, 3398, 3417, 3420, 4464, 4716, 7228.
- **Result: all 30 rows match every field I checked:**
  - absolute start = phase `inference_origin` (P23; origin confirmed at `phase23.py:267`) or block clock (Phase 1) + `started_s`;
  - run offset and phase offset;
  - the preceding skin reading (t ≤ start, Phase 1 including `skin_start`/`skin_end`), its value and age, and the frequency age;
  - policy0/4/6 maximum MHz;
  - M2 `inference_ms` versus wall time (e.g. row 4094: 967.67 vs 983.60 ms; row 7228: 4560.02 vs 4613.86 ms);
  - live/fallback/none context, and the miss flag.

### Sweep decisions

- **Stratified sample:** seed 5150, 4 + 3 + 3 drawn by category.
  - Already archived:
    - [33] `camera_power/run_20261003T004418Z_smoke/block_baseline_last.json`
    - [134] `camera_power/run_20261003T073546Z_smoke/manual_1000_start.jpg`
    - [246] `duty_cycle/run_20261002T201443Z_PARTS/block_P0U.json`
    - [1780] `downloads/duty_cycle_run_20261003T090742Z_CYCLE_report.txt`
  - Archived into folder:
    - [198] `coresidency/run_20261001T041743Z/block_B3_mix_gemma.json` (a B2 file)
    - [875] `owner_adaptive_p23_01_adaptive_idle.json`
    - [968] `owner_endurance_p23_01_endurance_samples_0028.json`
  - Inventory only:
    - [1409] `p23c_round1_checks/coder_mock_p23c_fixed_active_02.json`
    - [1630] `coresidency_review_artifacts/notes20.txt`
    - [1744] `coresidency_review_artifacts/review6/stdout.txt`
  - Every source and destination hash matches.
- **Full pass over all 2104 decisions:**
  - every source still exists with the recorded bytes and hash;
  - all 294 archive destinations OK;
  - all 1117 "already archived" decisions have a tracked destination with the same hash;
  - all 149 matches under 64 bytes carry the provenance note.
- **Matching policy:** matching purely by hash is acceptable now that tiny files are annotated and destinations must be tracked or in the candidate.
- **Report coverage:** every CAMPAIGN coder report and review final in Downloads is copied. The exceptions are FIX3 (unsafe, inventoried), AUDIT1 (already archived) and P23D's own files (external).
- **Report index:** `reports/INDEX.md` hashes match all 42 files. The verdicts I checked match the text (p23_r3, p1_fix3_r1, p23c_r2).

## Public safety

- **Candidate scan:** I ran the 6 provided patterns over all 415 candidate files plus the frozen manifest; all decode as UTF-8, so none were skipped. There are 2 email matches, at `reports/CAMPAIGN_P1_FIX_CODER_REPORT.md:336-337`. Both values already appear in unchanged tracked files, so they are allowlisted. **New findings: 0.**
- **Omitted sources:** I confirmed email matches at:
  - `ARCHIVE1_REPORT.md:526`
  - `CAMPAIGN_P1_FIX3_CODER_REPORT.md:339`
  - `CODER_REPORT_archive_0928.md:113,288`
  - `campaign/commit_campaign_p23b.json:110`
  - `review_request_campaign_p1.txt:1309`
  - `review_request_campaign_p1_r2.txt:1270`

  None of them was copied; all are inventory-only. FIX3 is listed as "PUBLIC SAFETY: new email; omitted unchanged, not redacted" at `UNARCHIVED_INVENTORY.md:105`. `CODER_REPORT_archive_0928.md:274` also matches, but that value is already in committed files.
- **Limits:**
  - The scan is regex-only.
  - The allowlist uses unchanged index-tracked files in the copy, not a HEAD tree walk.
  - I couldn't run git.
  - The source scan covered 2062 of the 2104 swept files; the rest are presumably binary.

## Checks I could confirm

- **Frozen manifest:** 415 files, 0 hash mismatches, 46,028,333 bytes, nothing over 50 MB. The patch is 47,107,755 bytes with SHA-256 `1023f4c3…0780`, matching the stated value.
- **Protected areas untouched:** the manifest has no changes to `phase23`, `phase1`, `runtime`, `power`, `diagnostics`, `coresidency.py`, readers or `docs/`.
- **Recorded self-check outputs:** they match the request. `results.json` shows exit 0 for all three. `self_check_p23d.r2_initial.stderr` records a failed first attempt and the fix that followed.
- **Not rerun after round 1:** the P23/P23C self-checks. Their code is unchanged, so I accept that.
- **Not run by me:** I didn't run any self-check, per the instructions.

**Verdict: APPROVED** (round 2). n1–n4 are optional cleanups. This is a review finding only; it does not authorize a commit or a push.
REVIEW_END

## Identity proof round 2

```json
{
  "command": "cd /termux-home/review_campaign_p23d_r2 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete independent review instructions are on stdin. Ponytail OFF. Follow them and return a final verdict.' --model claude-opus-5-5 --effort medium --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{\"mcpServers\":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23d_r2.txt\n",
  "copy_byte_identity": true,
  "date": "2026-10-11T17:22:07.609360+00:00",
  "effort": "medium",
  "exit": 0,
  "live_byte_identity": true,
  "model": "claude-opus-5-5",
  "ponytail": "off"
}

```

/termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r2_before.json SHA256 fc3e625974469791ba3fde36d4b43341f8696631b7d2594f8c7823a6ba386208
copy: 3105 files; canonical inventory SHA256 4ea0b4aa2e6dc540dfb1041fd37175bc1a2f3da372b611df589862fcdbcf24a5
live: 4984 files; canonical inventory SHA256 202fe4c842d749c070bde2eb72f28394c4e70601157687e849d09566a66de007
/termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r2_after.json SHA256 fc3e625974469791ba3fde36d4b43341f8696631b7d2594f8c7823a6ba386208
copy: 3105 files; canonical inventory SHA256 4ea0b4aa2e6dc540dfb1041fd37175bc1a2f3da372b611df589862fcdbcf24a5
live: 4984 files; canonical inventory SHA256 202fe4c842d749c070bde2eb72f28394c4e70601157687e849d09566a66de007

Review-copy/patch/identity/request files remain external:
- /termux-home/storage/downloads/campaign/review_campaign_p23d_r2.json; bytes 21845; SHA256 a4914ecd9484b2f3e3a056e373bf412a63f882e4fdac2475a2f55f0a0a5cd8b6
- /termux-home/storage/downloads/campaign/review_campaign_p23d_r2.stderr; bytes 0; SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- /termux-home/storage/downloads/campaign/review_campaign_p23d_r2_result.txt; bytes 12040; SHA256 fffcf1aeaba54f88620254cc960eb678e366595332aca3eb2fae71fb8793fb91
- /termux-home/storage/downloads/campaign/review_command_campaign_p23d_r2.txt; bytes 668; SHA256 d5fc192d028af851c1ca77ca8e141e9a7e1be9d70781c859eb807a5f06146d49
- /termux-home/storage/downloads/campaign/review_full_diff_campaign_p23d_r2.patch; bytes 47107755; SHA256 1023f4c3c17082a26a3da2a5c2cdfe103aefd223b4e09e4f59f2321eebb70780
- /termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r2_after.json; bytes 1592445; SHA256 fc3e625974469791ba3fde36d4b43341f8696631b7d2594f8c7823a6ba386208
- /termux-home/storage/downloads/campaign/review_identity_campaign_p23d_r2_before.json; bytes 1592445; SHA256 fc3e625974469791ba3fde36d4b43341f8696631b7d2594f8c7823a6ba386208
- /termux-home/storage/downloads/campaign/review_request_campaign_p23d_r2.txt; bytes 55069; SHA256 45969a6cae978d9d7ca40b30ac0b3e34dae51c77b3cf3f227283638e303fe59c
- /termux-home/storage/downloads/campaign/review_status_campaign_p23d_r2.json; bytes 883; SHA256 2c3cc4ee8a631a9f197366b6056ee3c3d67254b4ac225383ee5353d439ab4600

## Disposition

Last review round 2: APPROVED. copy/live byte identity=True/True.

Round-2 optional findings n1-n4 are retained in the approved frozen candidate: two folder README links point to absent runs/ directories; Qwen has residual partial wording and its INDEX row omits historical detail still preserved in README; generic rerun paragraphs conservatively say UNVERIFIED; ordinary new files remain unstaged until the commit gate. The reviewer explicitly judged these non-blocking. Commit preparation must stage exactly the frozen candidate and verify its hashes.

No commit/push. Four requested logs and 14 required ignored artifacts are staged; all changes await the human gate. STOP for owner “yes commit”; a separate “yes push” is required after committing.

Skipped: hardware/root/camera/model loads/live or timed runs, older-folder dataset conversion, unrequested tuning. Risks: recorded evidence gaps and public-safety regex limits; no new repeatability or deployment validity established.
