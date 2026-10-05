# CAMPAIGN_P1_FIX2 owner steps

M2 fp32 is a benchmark candidate, not deployed. This screens speed, memory,
heat and power; **selector accuracy with relation context is NOT evaluated**.
Motors OFF. Phone upright in its mount facing a cluttered table with 3–4
objects, lights ON, nothing moving, battery **at least 80%**, charger unplugged.
Keep the screen ON and stay in native Termux in the foreground throughout.
Exit Codex, Claude, AGY, node and every robot/benchmark/model-server process
before either command. The runner owns one Gemma server. No motion API is used.

Use a fresh stem for every invocation, changing it in all three filenames.
Commands use shell noclobber (`set -C`) to protect stdout/stderr; the runner
refuses existing JSON/block/warm-up/server evidence. Never truncate, remove
or reuse evidence. `owner_preflight_p1_fix2` already exists, so this preflight
uses `owner_preflight_p1_fix3`.

1. Preflight, no idle/warm-up/pauses/timed blocks:

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --preflight --output ~/storage/downloads/campaign/owner_preflight_p1_fix3.json > ~/storage/downloads/campaign/owner_preflight_p1_fix3.stdout 2> ~/storage/downloads/campaign/owner_preflight_p1_fix3.stderr)
```

Send `owner_preflight_p1_fix3.json`, `owner_preflight_p1_fix3.stdout`,
`owner_preflight_p1_fix3.stderr`, `owner_preflight_p1_fix3_llama-server.log`
(only files actually created). Require `PREFLIGHT ONLY — NO TIMING`,
`setup_complete: true` and successful cleanup before the session.

Preflight and session share setup: native/no-agent/charger/80% battery checks,
cores/affinity, root/su/pump masks and policies, thermal/sensor checks,
artifact/fallback hashes, decoded fallback photo/boxes and scoped diagnostic reads,
installed RobotCam version, YOLO/M2 workers,
screen/wake lock, owned server and S1O warmup, camera mode B at 1/s and a
verified live frame, camera stop, pre-idle checks. Each selected layout is checked.
Calling-thread affinity is refreshed/read back before monitor setup and must
include cores 4–7; Android cpuset restrictions still fail closed.

2. Full session, one command, approximately **85–90 minutes** (86 minutes
scheduled plus setup, model builds, camera transitions and cleanup):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --output ~/storage/downloads/campaign/owner_session_p1_fix2.json > ~/storage/downloads/campaign/owner_session_p1_fix2.stdout 2> ~/storage/downloads/campaign/owner_session_p1_fix2.stderr)
```

Send these exact files, only those actually created:

- `owner_session_p1_fix2.json` (includes idle sub-phases and all pauses)
- `owner_session_p1_fix2.stdout`, `owner_session_p1_fix2.stderr`
- `owner_session_p1_fix2_llama-server.log`
- `owner_session_p1_fix2_warmup_L0.json`
- `owner_session_p1_fix2_block_01_L0.json`
- `owner_session_p1_fix2_block_02_L1.json`
- `owner_session_p1_fix2_block_03_L2.json`
- `owner_session_p1_fix2_block_04_L3.json`
- `owner_session_p1_fix2_block_05_L4.json`
- `owner_session_p1_fix2_block_06_L0.json`

Plan: 300 s idle with Gemma loaded, no inference; first 180 s camera OFF,
then 60 s camera-OFF diagnostics and 60 s camera-ON without inference.
Camera stops again. ONE 180 s L0 warm-up is recorded as
**WARM-UP — NOT A RESULT**. Measured order: L0,L1,L2,L3,L4,L0.
Every measured block has a preceding **fixed 600 s pause**, camera OFF,
YOLO/M2/selector stopped, Gemma loaded. No skin gate. Skin/status, caps,
power and memory remain sampled through idle phases and pauses.

First measured start skin sets `T_ref_c`. Later blocks more than 1.5 °C
away are **NOT COMPARABLE — START TEMP**; other validity rules remain.
Comparability is retained even if another failure already invalidates a block.
Completion does not establish block validity. `--blocks L0,L1,L2` (or another
subset) retains the L0 warm-up and all preceding pauses. Below 80% battery
refuses setup. Below 25% before any warm-up, block, idle phase or pause
stops cleanly and leaves remaining layouts in `unrun_blocks`.
Thermal/charger/cores/process/sensor/camera/inference/server/cleanup failures
stop later blocks. Only local cadence failures may continue after successful
cleanup and no monitor errors.

Optional portable scheduling dry run, about 38 s, **NOT VALID**:

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --dry-run --output ~/storage/downloads/campaign/owner_dry_p1_fix2.json > ~/storage/downloads/campaign/owner_dry_p1_fix2.stdout 2> ~/storage/downloads/campaign/owner_dry_p1_fix2.stderr)
```

Send `owner_dry_p1_fix2.{json,stdout,stderr}`, `owner_dry_p1_fix2_warmup_L0.json`
and six block files with that stem, from `_block_01_L0.json` to `_block_06_L0.json`.
All hardware is mocked: short idle/pause phases, 1 s warm-up and six 6 s blocks.
Watts are null; no root/models/camera/server start. Debian can use
`/usr/bin/python3` and `/termux-home` paths.

## Load and diagnostics

YOLO MID uses cpus 4–5, two intra-op threads and one inter-op thread, at
320 EVERY frame plus 640 every 5 s on the same live RobotCam frame. Camera
mode B publishes 1/s. Selector is unchanged S1O every 20 s, 15 s bounded
HTTP requests, no accuracy grading or action execution. Gemma keeps ctx 2048,
threads/batch 4, parallel 1, swa-full, cache-ram 0; no MTP. Seven external
artifact hashes remain in `expected_hashes.json`.

L0 has no M2/context, including warm-up. L1: LITTLE 0–3 / four threads after
each 640. L2: BIG 6–7 / two threads before each selector using latest completed
640. L3: MID 4–5 / two threads after each 640; skipped YOLO slots while M2 runs
are counted, never queued. L4: BIG 6–7 / two threads after each 640.
At most one M2 request runs at a time; overload is recorded.

All M2 layouts use live 640 input with at least two boxes. With fewer, every
slot performs real inference on committed
`benchmark/relate_anything/desk2/speed_photo.jpg` and precomputed yolo11s boxes
in `speed_input.json`. Both SHA-256 values are hard-coded and verified in
setup and fallback loading; the existing PIL loader checks image size/box count.
More than 32 live boxes fails closed, never truncated. Per call: `scene`,
`live_boxes`, `input_boxes`, wall/ORT inference times and context. Per block:
`live_calls`, `fallback_calls`, calls and slots. Fallback context starts with
**(fallback scene)**, including later selector appends. Threshold-passing
top-five relation formatting stays unchanged. Context accuracy is not graded.

Each block/pause records start/end battery level/temperature, every discovered
thermal-zone type/temp, cooling-device type/cur_state, policy0 scaling_max_freq,
readable boost min/max and power-hint nodes, camera state and Android status.
Optional discovery is best effort: unreadable nodes are explicit; absent nodes
do not prove absent hints. Discovery is limited to depth five within the stated
sysfs/cpuctl roots, and `read_elapsed_s` records each snapshot's duration.
`camera_on` records the requested state; runner startup/frame and stop checks
verify transitions, but snapshots do not independently read camera state back.
Diagnostics only read, never alter caps. Idle
OFF/ON phases retain separate policy cap summaries to test camera-only capping.

Power uses current_now, nominal 0.37 s slots with overrun skips, root
read-start/receipt timestamps, time-weighted means, boundary brackets and
gap rejection >1.5 s. The owner's lag probe supports current_now, while
current_avg is too slow. Watts are not comparable with #128's method.
CPU/battery/caps: 1 s; skin/status: 4.87 s; memory/PSS: 5.13 s; block shared
checks: 5.23 s. Verified root/su/pump masks exclude inference cores; transient
service shells are pinned/read back. Binder/Magisk daemon work remains outside
affinity control. LMK windows exclude setup/tails; logcat failures invalidate.

## Verification limits

`self_check.py` retains existing lifecycle/refusal/scheduling/power/affinity
coverage, adapted for fixed pauses. `self_check_fix2.py` covers fallback
selection/counting/labels, hashes, temperature boundaries, battery refusal/stop,
diagnostics parsing, warm-up/subset scheduling and pause read/cleanup wiring
with mocks. Outputs and **NOT VALID** dry run: `checks/p1_fix2/`.
No live timed block/session/root preflight was run by the Coder; agents
resident invalidate timing and proot cannot reach root. Root affinity and
optional-node discovery, live battery/thermal/cooling/policy/logcat/PSS,
camera-only cap response, camera restarts, real fallback M2/YOLO overlap,
selector HTTP/context/timeouts, six 600 s cooling pauses and native cleanup
still require owner validation.

Historical owner evidence and source discrepancies: [RUN_INDEX.md](RUN_INDEX.md).
