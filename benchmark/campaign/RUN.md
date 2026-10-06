# CAMPAIGN_P1_FIX3 owner steps

M2 fp32 is a benchmark candidate, not deployed. This screens speed, memory,
heat and power; **selector accuracy with relation context is NOT evaluated**.
Motors OFF. Phone upright in its mount facing a cluttered table with 3–4
objects, lights ON, nothing moving, battery **at least 80%**, charger unplugged.
Keep the screen ON and stay in native Termux in the foreground throughout.
Exit Codex, Claude, AGY, node and every robot/benchmark/model-server process
before either command. The runner owns one Gemma server. No motion API is used.

Fix3: with the camera OFF the RobotCam app is force-stopped, so its PSS is now
recorded as `robotcam_app: "not running (camera OFF)"` (PSS null) instead of
stopping the session, but only after a separate status-checked root `pidof`
confirms no RobotCam process (`pidof_rc=1`, no pid); any other answer fails. A missing app while the camera is ON, root rc ≠ 0,
missing battery fields/MemAvailable, and missing runner/llama-server/camera
provider PSS still stop it. `owner_session_p1_fix2` stopped on exactly this.

Fix3b: thermal reads are checked. A `dumpsys thermalservice` read with root
rc ≠ 0, a reader error, or missing/unparsable skin or status stops the idle
phase, pause or block (also a read completing while workers are joined);
diagnostics use the same checked reader. Camera cleanup keeps `pidof`'s own
status: a RobotCam pid left after `am force-stop`, or an unconfirmed absence,
now fails cleanup (shared `coresidency.camera_end_failed`, used by every
runner). Archived owner evidence had no such case (2,271 thermal rows, 86
camera end checks).

Use a fresh stem for every invocation, changing it in all three filenames.
Commands use shell noclobber (`set -C`) to protect stdout/stderr; the runner
refuses existing JSON/block/warm-up/server evidence. Never truncate, remove
or reuse evidence. Both steps run the full preflight setup first (the same
setup as `--preflight`, which already passed as `owner_preflight_p1_fix3`).

1. Owner rehearsal, **NOT VALID**, about **12 minutes** (estimate, not
measured: about 4 min of shortened phases plus setup, model builds, camera
transitions, 37 real diagnostics snapshots of 4–15 s each and cleanup; 10–15 min). Same setup as the
session (battery ≥ 80%, charger unplugged, screen ON, Termux in front, no
agents):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --dry-run --output ~/storage/downloads/campaign/owner_rehearsal_p1_fix3.json > ~/storage/downloads/campaign/owner_rehearsal_p1_fix3.stdout 2> ~/storage/downloads/campaign/owner_rehearsal_p1_fix3.stderr)
```

It runs every live path of the session with real root, battery, sensors,
camera, models and server: full preflight setup; idle 6 s camera OFF, 6 s
camera OFF, 6 s camera ON (no inference); 20 s L0 warm-up; six 6 s pauses
(camera OFF, Gemma loaded); blocks L0,L1,L2,L3,L4,L0 of 25 s each (camera ON);
selector; live M2 slots and the real fallback path, forced on the M2 slot the
20 s selector reads (L2 at 0 s, L1/L3/L4 at 15 s) so every M2 layout also sends
a fallback-scene context to the selector; diagnostics snapshots, power/caps/memory/PSS/LMK;
cleanup and screen restore. Everything is labelled **NOT VALID — REHEARSAL**.
Short phases make cadence misses likely in L2/L3; that is expected and does
not stop it. The stdout ends with a `Coverage:` line (also
`rehearsal_coverage` in the JSON). Pass = final label exactly
`NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage`
and `"rehearsal_pass": true`. That flag requires setup, 3 idle sub-phases,
the warm-up, 6 pauses and all six blocks without errors, and for each of L1–L4
at least one live M2 call, one fallback M2 call and one selector call with a
fallback-scene context. Live M2 calls 0 in a layout means the camera saw
fewer than two objects: fix the scene before the session. Any other failure:
send the files and do not start the session.

Send these exact files, only those actually created:
`owner_rehearsal_p1_fix3.json`, `.stdout`, `.stderr`,
`owner_rehearsal_p1_fix3_llama-server.log`,
`owner_rehearsal_p1_fix3_warmup_L0.json`, and
`owner_rehearsal_p1_fix3_block_01_L0.json`, `_block_02_L1.json`,
`_block_03_L2.json`, `_block_04_L3.json`, `_block_05_L4.json`,
`_block_06_L0.json`.

2. Full session, only after the rehearsal passed. Recharge to **≥ 80%** if
needed, then unplug the charger. One command, approximately **85–90 minutes**
(86 minutes scheduled plus setup, model builds, camera transitions and cleanup):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --output ~/storage/downloads/campaign/owner_session_p1_fix3.json > ~/storage/downloads/campaign/owner_session_p1_fix3.stdout 2> ~/storage/downloads/campaign/owner_session_p1_fix3.stderr)
```

Send these exact files, only those actually created:

- `owner_session_p1_fix3.json` (includes idle sub-phases and all pauses)
- `owner_session_p1_fix3.stdout`, `owner_session_p1_fix3.stderr`
- `owner_session_p1_fix3_llama-server.log`
- `owner_session_p1_fix3_warmup_L0.json`
- `owner_session_p1_fix3_block_01_L0.json`
- `owner_session_p1_fix3_block_02_L1.json`
- `owner_session_p1_fix3_block_03_L2.json`
- `owner_session_p1_fix3_block_04_L3.json`
- `owner_session_p1_fix3_block_05_L4.json`
- `owner_session_p1_fix3_block_06_L0.json`

Preflight, rehearsal and session share setup: native/no-agent/charger/80% battery checks,
cores/affinity, root/su/pump masks and policies, thermal/sensor checks,
artifact/fallback hashes, decoded fallback photo/boxes and scoped diagnostic reads,
installed RobotCam version, YOLO/M2 workers,
screen/wake lock, owned server and S1O warmup, camera mode B at 1/s and a
verified live frame, camera stop, pre-idle checks. Each selected layout is checked.
Calling-thread affinity is refreshed/read back before monitor setup and must
include cores 4–7; Android cpuset restrictions still fail closed.

Session plan: 300 s idle with Gemma loaded, no inference; first 180 s camera OFF,
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

Portable mock (Coder, proot), about 40 s, **NOT VALID**: `--dry-run --mock`.
All hardware is mocked: short idle/pause phases, 1 s warm-up and six 6 s
blocks; watts are null; no root/models/camera/server start. Debian can use
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
with mocks. `self_check_fix3.py` replays the 37 owner camera-OFF memory samples
(pre-fix fail, fixed pass, camera ON fail), reproduces the base-commit
`rest()` failure, keeps root/battery/provider/server failures fatal, and
checks the rehearsal schedule, forced selector-feeding fallback slot, per-layout
rehearsal_pass, the pidof absence confirmation, late idle/pause reader failures
(now fatal after the worker joins), NOT VALID labels and
the unchanged full-session schedule. Outputs and the **NOT VALID** mock dry run:
`checks/p1_fix3/`. `self_check_fix3b.py` covers the checked thermal reader (rc, error,
skin/status failures, late completion during joins, camera ON/OFF, block and
rehearsal_pass) and the camera cleanup pidof checks; FIX3B outputs:
`checks/p1_fix3b/`.
No live timed block/session/root preflight/rehearsal was run by the Coder;
agents resident invalidate timing and proot cannot reach root. The owner
rehearsal is the first live run of the fix3 code; all live paths (root
affinity, PSS with the camera off and on, camera restarts, real fallback
M2/YOLO overlap, selector HTTP/context/timeouts, pauses and native cleanup)
remain unvalidated until it passes.

Historical owner evidence and source discrepancies: [RUN_INDEX.md](RUN_INDEX.md).
