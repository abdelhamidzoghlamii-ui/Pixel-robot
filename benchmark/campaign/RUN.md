# CAMPAIGN_P1 owner steps

M2 fp32 is a benchmark candidate, not deployed. This screens speed, memory,
heat and power; **selector accuracy with relation context is NOT evaluated**.
Motors OFF, phone on a table, lights ON, charger unplugged, screen ON, native
Termux foreground. No motors, USB or serial API is imported or commanded.
Exit Codex, Claude, AGY, node and every robot/benchmark/model-server process
before steps 1, 2 and 4. The runner starts and owns one Gemma server itself.
Never rerun a command with the same stem. Even a refused rerun can overwrite
`.stdout`/`.stderr` because the shell opens redirections before Python checks
evidence. Every invocation below uses a fresh `_p1_fix` stem and runs in a
subshell with `set -C` (noclobber): an existing redirect file refuses before
Python starts. On another attempt, change the stem in **every** filename; never
use `>|`, truncate or remove previous evidence. The runner still refuses existing
JSON/block/server evidence. Checking non-empty `.stdout`/`.stderr` inside Python
would be too late to prevent truncation, and normal startup itself fills stdout;
therefore shell noclobber is the protection, rather than a Python log-size check.

Historical owner lag/preflight/setup-refusal evidence is in [RUN_INDEX.md](RUN_INDEX.md).
The completed lag probe supports `current_now`; `current_avg` responds too slowly.
No lag probe rerun is needed for this fix. Commands below are native Termux;
Downloads `campaign/` already exists. Exit all agents before owner execution.

1. Phase-1 preflight with a fresh stem (all layouts; no idle/gates/timed blocks):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --preflight --output ~/storage/downloads/campaign/owner_preflight_p1_fix.json > ~/storage/downloads/campaign/owner_preflight_p1_fix.stdout 2> ~/storage/downloads/campaign/owner_preflight_p1_fix.stderr)
```

Send `owner_preflight_p1_fix.json`, `owner_preflight_p1_fix.stdout`,
`owner_preflight_p1_fix.stderr`, `owner_preflight_p1_fix_llama-server.log`.
Success: `PREFLIGHT ONLY — NO TIMING`, `setup_complete: true`, successful cleanup.
Preflight and session run exactly the same setup function and order, without a
mode argument, then branch at the first idle boundary. Both check each selected
layout's root/su/pump mask, required policies, YOLO/M2 workers and caller,
charger/processes/cores/skin/status/sensors, fixed artifact hashes and installed
RobotCam version; own a screen/wake lock and server, perform S1O warmup, start
camera mode B at rate 1 and verify a frame, then stop camera and run pre-idle
checks. Cleanup closes the server and restores screen/wake lock.

Before every campaign monitor setup, reset only the calling thread's requested
mask to the configured device CPUs, then read back and require cores 4–7. Android
cpuset restrictions remain enforced by the kernel. This repairs a stale narrowed
thread mask after a transient restriction; a continuing restriction fails closed.
Keep Termux foreground and screen on throughout. A preflight pass verifies that
invocation's start path, not future Android CPU availability. Layout evidence
records `setup_affinity.before/requested/actual`; refusal errors show masks,
process and read method. Do not proceed until preflight and cleanup succeed.

2. Session part A (L0,L1,L2; 14–38 minutes plus loads/setup/cleanup):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --blocks L0,L1,L2 --output ~/storage/downloads/campaign/owner_session_p1a_fix.json > ~/storage/downloads/campaign/owner_session_p1a_fix.stdout 2> ~/storage/downloads/campaign/owner_session_p1a_fix.stderr)
```

Send `owner_session_p1a_fix.json`, `owner_session_p1a_fix.stdout`,
`owner_session_p1a_fix.stderr`, `owner_session_p1a_fix_llama-server.log`,
`owner_session_p1a_fix_block_01_L0.json`, `owner_session_p1a_fix_block_02_L1.json`,
`owner_session_p1a_fix_block_03_L2.json` (only files actually created).

3. Session part B (L3,L4,L0; 14–38 minutes plus loads/setup/cleanup):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --blocks L3,L4,L0 --output ~/storage/downloads/campaign/owner_session_p1b_fix.json > ~/storage/downloads/campaign/owner_session_p1b_fix.stdout 2> ~/storage/downloads/campaign/owner_session_p1b_fix.stderr)
```

Send `owner_session_p1b_fix.json`, `owner_session_p1b_fix.stdout`,
`owner_session_p1b_fix.stderr`, `owner_session_p1b_fix_llama-server.log`,
`owner_session_p1b_fix_block_01_L3.json`, `owner_session_p1b_fix_block_02_L4.json`,
`owner_session_p1b_fix_block_03_L0.json` (only files actually created).
Each split has one 300 s idle with Gemma loaded and camera off, then 180 s blocks.
Skin gates require idle +1.5 °C, with 480 s maximum; timed-out gates label blocks
`NOT VALID — WARM START`. Session completion does not establish block validity.
Each split has its own baseline; cross-session comparisons need care. Default
unsplit order is L0,L1,L2,L3,L4,L0 (23–71 minutes plus setup), which exceeds
60 minutes at maximum gates.

Optional portable scheduling dry run (36 s; fully mocked, **NOT VALID**):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --dry-run --output ~/storage/downloads/campaign/owner_dry_p1_fix.json > ~/storage/downloads/campaign/owner_dry_p1_fix.stdout 2> ~/storage/downloads/campaign/owner_dry_p1_fix.stderr)
```

Files: `owner_dry_p1_fix.{json,stdout,stderr}` and
`owner_dry_p1_fix_block_01_L0.json`, `_block_02_L1.json`, `_block_03_L2.json`,
`_block_04_L3.json`, `_block_05_L4.json`, `_block_06_L0.json` (same stem).
Label: `NOT VALID — DRY RUN (all hardware mocked)`. Six 6 s blocks use the
same scheduler; no models/camera/server/root/idle/gates/sensors; null watts.
Can run in Debian with `/usr/bin/python3` and `/termux-home` paths.

## Reuse and layout details

Reused imported paths (desk2 mask diagnostics extended; behavior unchanged): `power_map.build_detector('mid')`,
`power_map.camera_start(1)`, `coresidency.read_frame`, `coresidency.Server`,
`coresidency.make_selector/select`, `variants.build('fp32', cluster)`,
`speed_block.prepare_monitor/root_mask/check_pinning/Screen`,
`coresidency.fast_sample/block_limit`, `power_map.capped_by_policy`.
No runner was fork-copied. The small campaign skin gate adapts power_map's
condition/bound because power_map also requires an externally maintained CPU
thermal log; this campaign reads CPU fault sensors directly instead.

Exact reused detector settings:

```python
opts.intra_op_num_threads = 2 if setting == 'mid' else 4
opts.inter_op_num_threads = 1
# MID workers/caller pinned to {4, 5}
```

Exact reused RobotCam command inside camera_start:

```python
cr.am('start', '-n', 'com.pixelrobot.robotcam/.StartActivity', '--es', 'mode', 'B', '--ei', 'rate', str(rate))
```

Frame path: `/storage/emulated/0/Download/robotcam/frame.jpg`. Same decoded
reader with JPEG session/frame/boottime validation and advancing frame numbers. Repeated frames retry for up to 1.35 s (one
publication period plus tolerance); a remaining repeat fails closed.
Installed versionCode/versionName are recorded by preflight; manual-exposure
options are not passed. Root shell/su/pump and sampler exclude the union of
YOLO MID and each layout's M2 cluster: L0/L3 safe 0–3,6–7; L1 safe 6–7;
L2/L4 safe 0–3. Nonempty masks and live readback required. Legacy transient root service shells
are adapted by a campaign-only wrapper: `taskset -p MASK $$` plus readback
before the original service command. Preflight exercises this for every layout.
The original `cr.root` function is restored after block/probe cleanup.

Exact reused server launch (`coresidency.server_cmd`, PORT=8080):

```python
return [server_manager.LLAMA_SERVER, '-m', MODEL, '--port', str(PORT), '--ctx-size', '2048',
        '--threads', server_manager.THREADS, '--threads-batch', server_manager.BATCH_THREADS,
        '--parallel', '1', '--swa-full', '--cache-ram', '0', '--host', '127.0.0.1', *extra]
```

MODEL is `~/models/gemma-4-E2B-it-Q4_0.gguf`, threads/batch threads both 4;
server binary is the existing `server_manager.LLAMA_SERVER`. No server CPU
pinning or MTP. `expected_hashes.json` fixes seven external artifacts.
Selector uses unchanged S1O prompt/scoring via make_selector/select every
20 s; only transport timeout is bounded to 15 s per HTTP request. Successful calls
finishing past the next selector slot count as misses as well as cadence failures. Prompt
is the existing situation/options/instruction, latest M2 text appended to
situation. No accuracy grading, no robot action is executed.

**Explicit detection instruction differs from historical SizePolicy**:
#128 selects one size per frame (640 replaces 320 at large slots). This task
explicitly requires 320 on every frame PLUS 640 every 5 s, so the campaign
runs both on the same frame at large slots (0,5,...,175). Detector sessions,
threads, pins, camera, selector and server are reused unchanged. This adds
36 small detections per full block compared with the historical mix.

L0 has no M2/context. L1: LITTLE 0–3 / four threads after each 640.
L2: BIG 6–7 / two threads immediately before each selector, using latest
completed 640 and its boxes; selection waits for M2. At later selector slots
this can be the 640 frame from five seconds earlier, while the current
YOLO-640 runs concurrently on MID. Context freshness/overlap are recorded,
and context accuracy is not evaluated. L3: MID 4–5 / two
threads after each 640; due YOLO slots while M2 runs are skipped, counted,
never queued. L4: BIG 6–7 / two threads after each 640. One M2 request at a
time; cadence overload is recorded. M2 requires 2–32 boxes: fewer than two
produces empty context and an insufficient-box slot count, not fabricated
inference; more than 32 fails closed rather than silently truncating boxes.

Context uses top five score-sorted rows with pass=true and score>=threshold:

```text
M2 relations:
- person [0] next to chair [1].
```

Both wall-clock M2 call stats and ORT `inference_ms` stats are recorded for
comparison with SPEED1. Skin/gate/end timestamps are relative to the common
block origin; caps and memory summary exclude post-180-s tails. Logcat read
failure invalidates the block. Existing server logs are refused rather than appended.

Underscores become spaces; indices distinguish same-class detections. Empty:
`M2 relations:\n- none above threshold.` L0 appends nothing. An initial
L1/L3/L4 selector may receive empty context before the first relation finishes.
All actual appended strings are retained per selector call.

Power uses 0.37 s nominal slots (overrun skips slots), root read-start and
receipt times, signed current_now and voltage_now, time-weighted mean over
receipt timestamps, full boundary brackets, inside/outside M2 call sample
counts and read-boundary crossings. Coverage gaps >1.5 s => null watts and
invalid block. De-phasing fixes cadence aliasing; gauge lag remains a method
limitation until the owner probe is inspected. CPU/battery/caps sampled every
1 s (start offset 0.11 s); the nominal five-second skin/status series uses 4.87 s (offset 0.37 s),
and memory/PSS uses 5.13 s (offset 1.31 s). These periods deliberately drift
relative to 1 s and 5 s inference slots; all actual receipt timestamps and
configured periods are recorded. Per-frame guard uses only in-process state/core checks;
process/HTTP health checks use a pinned monitor every 5.23 s (offset 0.71 s),
and before/after the block. The lag probe also uses pinned 5.23 s process checks,
not 10 Hz pgrep in its quiet baseline. All forked monitor clients inherit
the verified safe thread mask; Magisk-spawned command shells explicitly set
and read back their mask. Binder service work in existing Android processes
and Magisk daemon/control startup work remain outside campaign affinity
control. These instrumentation costs remain method limits shared by all layouts;
the non-divisor periods prevent a fixed phase at each 640 slot, but
do not eliminate measurement overhead. The LMK window is exactly the 180 s
block, excluding model builds/gates and cleanup; unparseable epoch timestamps
invalidate the logcat accounting. Per-policy cap accounting reuses the
existing policy0/4/6 method. Logcat LMK rows retained per block.

Continuation matches SPEED1: only a local cadence failure may continue with
successful cleanup and no monitor errors. Root/sensors, thermal stops,
charger/processes, cores/affinity, inference/camera/server and cleanup failures
stop later blocks. Warm starts remain invalid and the next block is gated.

## Verification limits

`self_check.py` exercises new scheduling/power/lag/refusal/lifecycle paths with
mocks. Original direct-script test results are in `checks/results.json`; fix checks are in `checks/p1_fix/results.json`.
Coder ran no real timed block, session or root preflight. Proot cannot reach
phone root; agents resident would invalidate timings. Live battery gauge,
root/su affinity/readback, Android services/policies/caps/thermal/logcat/PSS,
installed RobotCam startup/settings/frames, real M2/YOLO overlap and skips,
Gemma HTTP/health/timeouts/context, cooling gates, idle and native cleanup
remain owner validation paths. Portable dry-run covers scheduling with mocks.

Signals during a block are recorded as shared failures; cleanup runs and later
blocks stop. The outer session exits 1 rather than preserving 128+signal.
