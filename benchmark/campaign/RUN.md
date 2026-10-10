# CAMPAIGN_P1_FIX4 / FIX4D owner steps

M2 fp32 is a benchmark candidate, not deployed. This screens speed, memory,
heat and power; **selector accuracy with relation context is NOT evaluated**.
Motors OFF. No motion API is used. The runner owns one Gemma server.

## OWNER CHECKLIST (before the rehearsal and again before the session)

- [ ] Airplane mode ON (the server is 127.0.0.1; no calls, no network updates).
- [ ] Do Not Disturb ON, **alarms and timers OFF**, and no alarm due in the next 2.5 h.
- [ ] Automatic system and app updates paused (Play Store, system update).
- [ ] Magisk superuser notifications (toasts) set to none.
- [ ] 3–10 objects in view of the camera, no shelves, lights ON, nothing moving.
- [ ] Phone upright in its mount, facing the table.
- [ ] Battery ≥ 80 %, charger unplugged.
- [ ] Exit Codex, Claude, AGY, node, editors and every robot/benchmark/model-server process.
- [ ] Screen ON, native Termux in the foreground; then **hands off** until the final label prints.

## How to start (M9)

Open **one** native Termux session and paste **exactly** the command below — **no wrapper**: not
`oneshot.sh`, not `script`, not `timeout`, not `sh -c "…"`, not a `run_*.sh` file. Do not open another
Termux session, `less`, `vim` or any editor on the campaign files while it runs: the process guard refuses
any matching process outside the runner's own tree (a wrapper or an open `phase1.py` matches), and a
refusal mid-session ends it. Use a fresh stem for every invocation, changing it in all filenames.
Commands use shell noclobber (`set -C`) to protect stdout/stderr; the runner refuses existing
JSON/block/pause/idle/warm-up/server evidence. Never truncate, remove or reuse evidence.

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

Fix3c: the process guard no longer flags the runner's own processes. `owner_rehearsal_p1_fix3` stopped
after the warm-up because its own `su -c` diagnostics client (pid 1312, a direct child) contained
`read_node`, which matched the guard alternative `node`. Matches are now excluded only when
`/proc/PID/stat` ppid ancestry, read at check time, reaches the runner pid (or it is the owned
server pid); command text is never trusted. A matching process outside our tree, even with an
identical command line, is still refused, and unreadable ancestry is refused too (fail closed).
A matched pid that has already exited when its ancestry is read is no longer resident and is skipped.

Fix3d: `owner_session_p1_fix3c2` stopped in the camera-ON idle sub-phase because one checked thermal read
returned rc 0 with no skin and no status. `owner_session_p1_fix3c` stopped 5 s into the first idle
sub-phase with the same signature, from the thermal worker. The likely cause: the thermal worker and the
main thread's checks wrote the same root output file (`coresidency_thermalservice.txt`), so one read
could `cat` the file just after the other had truncated it. Each thread now uses its own file
(`campaign_thermal_<tid>`). Also, a read with rc 0 but missing or unparsable skin or status is retried:
at most 3 attempts, 0.5 s apart. A retry is started and accepted only within 2 s of the end of the first
read. Its `su` timeout is the time left, and a retry that ends later (a slow retry, or a slow process
launch the timeout does not cover) counts as incomplete and fails closed. At the measured 0.2 s per
read, 3 attempts end 1.6 s after the start. Retries add at most 2 s to an accepted read. A failing read
can take longer, by at most the launch/cleanup time of the last retry and an OS oversleep of the
last pause. The first read keeps its existing 30 s timeout. Worker slots stay on their 4.87 s grid;
any read longer than the period skips a slot (a 4.87 s gap, never bunched), which is the existing
behaviour for slow reads. The 0.12–0.20 s reads measured so far never come close.
A partial read showing a stop status (≥ 4) is never retried. rc ≠ 0, a reader error, or a read
still incomplete after the attempts or the window stop the phase (fail closed).
Each thermal row records `attempts`. Each incomplete attempt keeps its raw output (1 KB): in the row
(`incomplete_attempts`), in the JSON's `thermal_retries` for every caller, or, when no read completes,
in the error. The lag probe's thermal worker now uses the same checked reader and also saves
`thermal_retries`.


Fix4 (audit `AUDIT1_REPORT.md`, owner decisions D1/D2):
- H1: in every idle phase and pause the second root shell is built before the power bracket; the
  bracket, the time origin and all monitor threads then start together (as in blocks), so the first
  power gap is no longer about 1 s of the 1.5 s budget.
- H2: every root/sysfs/dumpsys/pidof/ps/pgrep/battery/power/mask/fast/PSS/logcat/am/health read gets
  at most **one** re-read 0.3 s later, and only when the answer carried nothing (FIX4C/FIX4D): no output at all
  (stdout AND su's stderr blank), a su/am/pgrep launch failure (or a failure exit with no output on either stream), a transient mask readback with an empty mask, or
  an HTTP timeout. Every answer with output — complete, partial, malformed or bad — is judged once by the reader's
  own checks exactly as before FIX4 and is never re-read. `/health` fails only on 2 consecutive failures; a
  RobotCam pid still present after force-stop is re-checked once after 0.5 s. If the re-read carries nothing again
  the read fails (or, for evidence-only reads such as the screen state, an error is
  recorded, never raised). Every re-read is recorded (`read_retries` in the JSON, per phase including
  `setup` and the capacity check just before each pause or block, and `retries` per row); more than 3 in
  one phase makes that phase or block **NOT VALID — READ RETRIES** (the session continues; the rehearsal
  fails; the lag probe is incomplete). Cleanup failures keep their first two causes and fail rehearsal_pass. The thermal reader keeps its FIX3D rule (3 attempts
  within 2 s for an incomplete read; rc ≠ 0 fails at once).
- H3: more than 32 live boxes: M2 uses the 32 highest-confidence boxes (detector order kept) and
  records `live_boxes` (n live) and `truncated_boxes`; each block records `max_live_boxes`
  (largest YOLO-640 box count) and prints it.
- H4: a missing (or repeated) frame is polled until 1.35 s; if none arrives the slot is skipped
  (`camera_misses`, counted late, a cadence miss) and the block goes on. 3 failed slots in a row,
  or a `bad`/`other_session` frame, stop the session.
- H5: every diagnostics snapshot and every failing phase records the screen state
  (`dumpsys power` wakefulness/display lines and the resumed activity), so a lock or a lost
  foreground is visible in the files.
- M1: JSON files are written atomically (temp file, fsync, rename). The session JSON says
  `IN PROGRESS — NOT VALID` until the final label. Each idle sub-phase and pause also has its own file.
- M3: the pause loop no longer makes its own battery and thermal reads; the power sampler, memory rows
  and thermal worker (with the block limits) cover them.
- M4/L3: errors carry their first causes; stdout prints one line at each phase start and end
  (elapsed time, re-read and thermal-retry counts; for blocks camera lateness, skipped camera slots,
  live/fallback M2 calls and `max_live_boxes`).
- M5: a main-loop heartbeat older than 15 s during a block makes the runner SIGTERM itself (normal
  cleanup runs); pgrep (10 s) and termux-wake-lock/unlock (30 s) have timeouts; the selector's 15 s
  transport bound applies from its first tokenize request.
- L4/L5: a second Ctrl-C/signal during the final cleanup is ignored; a cleanup error never replaces
  a battery-stop or cores-lost cause.
- L7: after setup the main thread is pinned to the monitor cores; cores 4–7 are still checked (on a
  dedicated unpinned thread).
- FIX4C (narrowed re-read): the FIX4B per-value judging is removed; each reader is its pre-FIX4 parser plus the
  empty-answer re-read above. The camera end check still runs once (a "capture not stopped" result stays) and a
  force-stop that ran keeps its own result; only a force-stop that did not run, or a pidof that answered nothing,
  is redone.
- FIX4D (su stderr kept): su's own stderr (e.g. "su: permission denied") is part of the answer. `coresidency.root`
  keeps it per thread (`root_stderr()`) and `RootShell` keeps each command's stderr (`last_stderr`); an answer with
  any stdout or stderr is judged once by the reader's own checks and never re-read. Only an OSError raised while
  starting su/am/pgrep counts as a launch failure (for the camera start: only its first `am start`); any other error
  stands as the reader's own.
- D1: L1 runs M2 after every **second** 640 frame (every 10 s, 18 calls per 180 s block); other layouts unchanged.
- D2 (P23 start-temperature fix): after the fixed 600 s camera-OFF pause, block setup completes
  (including model builds and camera start). With inference still idle, monitoring runs and a fresh
  checked skin reading gates the measured interval at T_ref + 1.5 °C, for at most 300 s more.
  The block records `cooling.extra_s`; its **same final gate reading** is `skin_start` and decides
  comparability. Setup/cooling heat is excluded from measured power and cadence by the common origin.
  The camera is ON during this post-setup extension. Still outside: the block runs and is labelled
  **NOT COMPARABLE — START TEMP**. A skin more than 1.5 °C below T_ref is labelled NOT COMPARABLE
  without waiting.

1. Owner rehearsal — **MANDATORY before the session**, **NOT VALID**, about **12 minutes** (estimate;
up to 30 s more of D2 cooling). Same checklist and start rule as the session:

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --dry-run --output ~/storage/downloads/campaign/owner_rehearsal_p1_fix4d.json > ~/storage/downloads/campaign/owner_rehearsal_p1_fix4d.stdout 2> ~/storage/downloads/campaign/owner_rehearsal_p1_fix4d.stderr)
```

It runs every live path of the session with real root, battery, sensors,
camera, models and server: full preflight setup; idle 6 s camera OFF, 6 s
camera OFF, 6 s camera ON (no inference); 20 s L0 warm-up; six 6 s pauses
(camera OFF, Gemma loaded, D2 extension at most 6 s); blocks L0,L1,L2,L3,L4,L0 of 25 s each (camera ON);
selector; live M2 slots and the real fallback path, forced on the M2 slot the
20 s selector reads (L2 at 0 s, L1 at 10 s, L3/L4 at 15 s) so every M2 layout also sends
a fallback-scene context to the selector; diagnostics snapshots, power/caps/memory/PSS/LMK;
cleanup and screen restore. Everything is labelled **NOT VALID — REHEARSAL**.
Short phases make cadence misses likely in L2/L3; that is expected and does
not stop it. The stdout ends with a `Coverage:` line (also
`rehearsal_coverage` in the JSON).

**Go / no-go: start the session only if** the final label is exactly
`NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage`
**and** `"rehearsal_pass": true`. That flag requires setup, 3 idle sub-phases, the warm-up, 6 pauses
and all six blocks without errors; for each of L1–L4 at least one **live** M2 call, one fallback M2
call and one selector call with a fallback-scene context; `max_live_boxes` reported for every block;
and no phase over the re-read cap (`read_retry_over_cap` empty). Live M2 calls 0 in a layout means the
camera saw fewer than two objects; a `max_live_boxes` near or above 32 means too much clutter: fix the
scene (3–10 objects, no shelves) and rehearse again. Any other failure: send the files and do not
start the session.

Send these exact files, only those actually created:
`owner_rehearsal_p1_fix4d.json`, `.stdout`, `.stderr`,
`owner_rehearsal_p1_fix4d_llama-server.log`, `owner_rehearsal_p1_fix4d_idle_1.json`, `_idle_2.json`,
`_idle_3.json`, `owner_rehearsal_p1_fix4d_warmup_L0.json`,
`owner_rehearsal_p1_fix4d_pause_01_L0.json` … `_pause_06_L0.json` (one per block, named like the block),
and `owner_rehearsal_p1_fix4d_block_01_L0.json`, `_block_02_L1.json`,
`_block_03_L2.json`, `_block_04_L3.json`, `_block_05_L4.json`,
`_block_06_L0.json`.

2. Full session, only after a passing rehearsal. Recharge to **≥ 80 %** if needed, unplug the charger,
run the checklist again. One command, about **95–100 minutes** (86 minutes scheduled plus setup, model
builds, camera transitions and cleanup), **plus up to 25 min of D2 cooling** (at most 300 s after each
of the five pauses that follow the first measured block):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --output ~/storage/downloads/campaign/owner_session_p1_fix4d.json > ~/storage/downloads/campaign/owner_session_p1_fix4d.stdout 2> ~/storage/downloads/campaign/owner_session_p1_fix4d.stderr)
```

Progress: `tail -n 3 ~/storage/downloads/campaign/owner_session_p1_fix4d.stdout` is safe **after** the
run; during the run do not open a second session (see "How to start"). Each phase prints START/END lines.

Send these exact files, only those actually created:

- `owner_session_p1_fix4d.json` (includes idle sub-phases and all pauses)
- `owner_session_p1_fix4d.stdout`, `owner_session_p1_fix4d.stderr`
- `owner_session_p1_fix4d_llama-server.log`
- `owner_session_p1_fix4d_idle_1.json`, `_idle_2.json`, `_idle_3.json`
- `owner_session_p1_fix4d_warmup_L0.json`
- `owner_session_p1_fix4d_pause_01_L0.json`, `_pause_02_L1.json`, `_pause_03_L2.json`,
  `_pause_04_L3.json`, `_pause_05_L4.json`, `_pause_06_L0.json`
- `owner_session_p1_fix4d_block_01_L0.json`, `_block_02_L1.json`, `_block_03_L2.json`,
  `_block_04_L3.json`, `_block_05_L4.json`, `_block_06_L0.json`

## RECOVERY (M8) — only if the runner was killed or hung and never printed its final label

A hard kill (out-of-memory kill, force-closing Termux, a battery cut) skips the runner's cleanup. From
**native Termux**, after any runner process is gone (`pgrep -fa phase1.py` prints nothing):

```sh
su -c 'settings put system screen_off_timeout SAVED_MS'   # SAVED_MS = the number printed in the .stdout as "screen timeout saved SAVED_MS ms"
su -c 'am force-stop com.pixelrobot.robotcam'
pkill -f llama-server
termux-wake-unlock
```

If the saved value printed was already 2147483647, use your normal timeout (for example 60000).
Keep the files as they are and send them; do not rerun under the same stem.

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
YOLO/M2/selector stopped, Gemma loaded, extended by D2 cooling only as above (at most 300 s).
Skin/status, caps, power and memory remain sampled through idle phases and pauses, extensions included.

First measured start skin sets `T_ref_c`. Later blocks more than 1.5 °C
away are **NOT COMPARABLE — START TEMP**; other validity rules remain.
Comparability is retained even if another failure already invalidates a block.
Completion does not establish block validity. `--blocks L0,L1,L2` (or another
subset) retains the L0 warm-up and all preceding pauses. Below 80% battery
refuses setup. Below 25% before any warm-up, block, idle phase or pause
stops cleanly and leaves remaining layouts in `unrun_blocks`.
Thermal/charger/cores/process/sensor/camera/inference/server/cleanup failures
stop later blocks (after the single H2 re-read where it applies). Only local cadence failures
(including skipped camera slots) and the re-read cap may continue after successful
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
every second 640 (every 10 s, 18 calls per 180 s; D1). L2: BIG 6–7 / two threads before each selector using latest completed
640. L3: MID 4–5 / two threads after each 640; skipped YOLO slots while M2 runs
are counted, never queued. L4: BIG 6–7 / two threads after each 640.
At most one M2 request runs at a time; overload is recorded.

All M2 layouts use live 640 input with at least two boxes. With fewer, every
slot performs real inference on committed
`benchmark/relate_anything/desk2/speed_photo.jpg` and precomputed yolo11s boxes
in `speed_input.json`. Both SHA-256 values are hard-coded and verified in
setup and fallback loading; the existing PIL loader checks image size/box count.
More than 32 live boxes: the 32 highest-confidence boxes, in detector order (H3). Per call: `scene`,
`live_boxes`, `input_boxes`, `truncated_boxes`, wall/ORT inference times and context. Per block:
`live_calls`, `fallback_calls`, `truncated_slots`, calls, slots and `max_live_boxes`. Fallback context starts with
**(fallback scene)**, including later selector appends. Threshold-passing
top-five relation formatting stays unchanged. Context accuracy is not graded.

Each block/pause records start/end battery level/temperature, every discovered
thermal-zone type/temp, cooling-device type/cur_state, policy0 scaling_max_freq,
readable boost min/max and power-hint nodes, camera state, Android status and screen state
(wakefulness/display lines and resumed activity; also recorded on every phase failure).
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
`self_check_fix3c.py` reproduces the owner failure on the base guard with the archived
monitor_errors line, passes it after the fix, refuses identical text outside our tree, fails
closed on unreadable/hidden/cyclic ancestry, checks real `/proc` ancestry with an own child and a
reparented orphan, and audits every command line the runner spawns (only diagnostics `node` and
the owned llama-server match; both are direct children). FIX3C outputs: `checks/p1_fix3c/`.
`self_check_fix3d.py` uses the real `dump_once` and parser behind a fake `cr.root`. It checks: one empty
read then success (passes, attempts 2, raw kept); partial/oversized raw (truncated); three empty reads
(fails closed, 3 raws in the error); rc ≠ 0, timeout and OSError (fail at once, no retry); a partial
status-4 read (fails at once, in the reader, the `rest()` check and the worker); and the retry window.
For the window, retries start and are accepted only within 2 s of the first read's end, for first reads
of 0.01–29 s, retries of 0.01–29.9 s, a 0/3 s launch outside the timeout and an OS oversleep of
0/0.3/3 s. A late complete retry fails closed; the round-2 code accepted it. It also checks the 4.87 s
worker grid: a 2.2 s read skips no slot, and a 5.0 s read skips exactly one. `self_check.py`'s
lag-probe lifecycle covers its checked thermal worker.
It also reproduces the shared-file race with cr.root's exact su form and a fake su/dumpsys in a forced
interleaving (shared tag: rc 0, no skin, no status; per-thread tags: both complete). Finally, it covers
the real `dump_check` in `rest()` (camera OFF and ON, last in-phase check) and at `live_block()`
skin_end (session and rehearsal). FIX3D outputs: `checks/p1_fix3d/`.
`self_check_fix4.py` covers the FIX4 changes: H1 (the audited `rest()` loses a pause with a 1.6 s
second-shell setup; fixed order and zero bracket-to-thread gap with 0.4/1.6/5 s setups), the H2 helper
(re-read success, persistent failure with both errors, other errors not re-read, `/health` 2 of 2,
per-row and per-phase records, the > 3 cap making a block or pause NOT VALID without a stop), H3 (40 boxes → top 32), H4 (single miss, repeat, 2+2, 3 in a row, bad/other_session,
polling), D1 (18 L1 calls), D2 (no T_ref, cooled within, 300 s exhausted, below the band, the band edge,
a sensor stop during the extension), M1 (atomic write, per-phase files, IN PROGRESS label), M3, M4, M5
(stale heartbeat → SIGTERM, transport bound before tokenize, timeouts), H5, L3, L4, L5, L7 and this
file's checklist/recovery text. `self_check_fix3c.py` now also audits the guard's own pgrep (L6).
Existing self-checks were adapted where FIX4 intentionally changed behaviour (D1 counts, M3, M4
messages, H2 pid re-check, L7 stubs).
`self_check_fix4c.py` (FIX4C): for every reader wrapped by the re-read helper, through its own parser with only
the transport mocked, a non-empty answer (good, partial, malformed or bad) is read exactly once, and an empty answer
is re-read once and fails closed when the re-read is empty or bad; it also re-runs every probe of the six FIX4/FIX4B
review rounds (table) and checks that the thermal reader is unchanged. `self_check_fix4d.py` (FIX4D): the same rule
through the REAL `cr.root` (subprocess stdout and stderr mocked) and the REAL `RootShell` (a fake `su`): a stderr-only su
failure is read once and fails closed; a blank stdout+stderr answer is re-read once; it re-runs every earlier probe
table. Outputs: `checks/p1_fix4d/`.
No live timed block/session/root preflight/rehearsal was run by the Coder;
agents resident invalidate timing and proot cannot reach root. The owner
rehearsal (mandatory) is the first live run of the fix4 code; all live paths (root
affinity, PSS with the camera off and on, camera restarts, real fallback
M2/YOLO overlap, selector HTTP/context/timeouts, pauses and native cleanup)
remain unvalidated until it passes.

Historical owner evidence and source discrepancies: [RUN_INDEX.md](RUN_INDEX.md).

## CAMPAIGN_P23 — L2 only, one mode per session

Phases 2/3 use M2 relsgg-vits16 fp32 on BIG cpus 6–7 / 2 threads, immediately before each
20 s selector call, on the latest completed YOLO-640 frame. YOLO follows the #128 CONFIRM setting:
1 frame/s, yolo11s on MID cpus 4–5 / 2 threads; `SizePolicy(interval_s=5)` replaces 320 with 640,
with the first 640 at +5 s after camera readiness. There is one inference per frame slot.
M2 fallback (<2 boxes), top-32 selection and explicit fallback contexts retain the Phase-1 rules.
Selector accuracy is NOT EVALUATED. Motors OFF throughout; never connect or command motors.

### Owner checklist and start rule

Use the OWNER CHECKLIST above: airplane mode; Do Not Disturb; alarms/timers off; updates paused;
Magisk toasts off; upright mount and 3–10 stationary objects, lights on, no shelves;
battery >= 80 %, charger unplugged; native Termux foreground and screen ON, hands off.
Exit all agents, model servers, robot/benchmark processes and editors. Start exactly the command below,
**no wrapper** and no second Termux session. A fresh stem is mandatory; change every filename together.
The screen timeout saved/restored and wake lock are handled as in Phase 1. Ctrl-C requests normal cleanup;
a second signal during cleanup is ignored. Hard-kill recovery uses the RECOVERY block above, after
`pgrep -fa phase23.py` shows that the runner is gone. Keep all incomplete files.

**recharge to >= 80 % and let the phone rest >= 30 min between sessions**.
This also applies between the rehearsal and the first measured session.

Before the first idle origin of each invocation, setup waits up to 60 s for 10 consecutive power samples whose receipt intervals are all <= 1.5 s, with the last sample <= 1.5 s old at that origin. `setup.power_settle` records `settle_s`, sample count and maximum interval; timeout is a setup failure (NOT VALID), with no measured phase. This wait is outside every measured window and does not change any phase duration or power-coverage rule. Preflight-only has no sampler or idle origin, so it does not settle.

### Mandatory rehearsal — one command for all three modes

From native Termux, with no agent resident:

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase23.py --dry-run --output ~/storage/downloads/campaign/owner_rehearsal_p23_02.json > ~/storage/downloads/campaign/owner_rehearsal_p23_02.stdout 2> ~/storage/downloads/campaign/owner_rehearsal_p23_02.stderr)
```

About 16–19 min, allowing setup and camera transitions: idle 30 s / gate at most 30 s before each mode;
endurance 180 s; fixed 3 wall-clock cycles of 40 s active slot / 20 s pause slot; adaptive 480 s, with recorded
T_hi = final gate skin + 1.0 C and the session's minimum active 60 s / minimum camera-OFF pause 30 s.
For each rehearsal adaptive pause, restart at the checked high-switch trigger skin minus 2.0 C, recorded as `switch_trigger_skin_c` and `low_threshold_c`; this keeps the sessions' 2 C hysteresis size. With the archived L2 overshoot, restart no longer depends on cooling below the gate reading. Measured sessions keep restart at T_hi minus 2.0 C (35.0 C at the default T_hi of 37.0 C).
The longer adaptive window allows cooling: archived Phase-1 `owner_session_p1_fix4d_*` dumps show L2 skin rising about 3 C in the first 60 s, while camera-OFF pauses fall only 1.97–2.97 C in 120 s and 2.75–4.29 C in 300 s.
This is live hardware rehearsal, **NOT VALID as results**. It uses all normal readers, inference,
monitoring, fallback and cleanup paths. Mock tests cover emergency, refusal, warm-start gate timeout
and clean battery-stop paths, plus full-length camera drift and fractional end slots;
the owner must never deliberately heat the phone to provoke an emergency.

Proceed only if `rehearsal_coverage.rehearsal_pass` is true: setup and all three modes completed;
live M2 ran in every mode (forced fallback is only the first active phase's first selector);
all 3 fixed pauses ran; adaptive crossed both hysteresis thresholds
and restarted its camera; camera restarts ran without failures; no reader-cap or other failure.
The threshold depends on the real cooling curve: if adaptive never pauses or never cools enough
to restart before its deadline, rehearsal_pass is false. Send the files and do not start a session.
The runner requires `--rehearsal` for each measured session and rejects mock/failed/other-code evidence.

Send **every file actually created** with this stem:

- `owner_rehearsal_p23_02.json`, `.stdout`, `.stderr`, `_llama-server.log`.
- For each suffix `endurance`, `fixed`, `adaptive`: `owner_rehearsal_p23_02_SUFFIX.json`,
  `owner_rehearsal_p23_02_SUFFIX_idle.json`, `owner_rehearsal_p23_02_SUFFIX_gate.json`.
- `owner_rehearsal_p23_02_endurance_active_01.json`.
- Fixed: `owner_rehearsal_p23_02_fixed_active_01.json` … `_active_03.json`,
  and `owner_rehearsal_p23_02_fixed_pause_01.json` … `_pause_03.json`.
- Adaptive: every `owner_rehearsal_p23_02_adaptive_active_NN.json` and
  `owner_rehearsal_p23_02_adaptive_pause_NN.json` (count follows the measured thresholds).
- For every mode, all `owner_rehearsal_p23_02_SUFFIX_samples_NNNN.json` checkpoint files.

Offline check only: `python benchmark/campaign/phase23.py --dry-run --mock --output /tmp/FRESH_p23_mock.json`.
It is NOT VALID, cannot set rehearsal_pass true, and cannot authorize a measured session.

### Three measured sessions — separate invocations

Optional preflight checks all owner protections without a measured run:
`python -u ~/robot/benchmark/campaign/phase23.py --mode endurance --preflight --output ~/storage/downloads/campaign/owner_preflight_p23_01.json`.
Preflight is not a rehearsal pass. Its fresh-stem JSON and `_llama-server.log` are the evidence.

1. Endurance: L2 from a cold start, maximum 30 min of active time; stop earlier at Android SEVERE
   (status >=3), or at any #127 emergency. Expected about 37–40 min including setup and 5 min idle,
   plus up to 15 min cold-gate waiting; SEVERE/emergency can end it earlier.

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase23.py --mode endurance --rehearsal ~/storage/downloads/campaign/owner_rehearsal_p23_02.json --output ~/storage/downloads/campaign/owner_endurance_p23_01.json > ~/storage/downloads/campaign/owner_endurance_p23_01.stdout 2> ~/storage/downloads/campaign/owner_endurance_p23_01.stderr)
```

2. Fixed (owner M3): 2160 s measured wall-clock grid, 12 cycles of 180 s with a 120 s active slot
   and a 60 s pause slot. Camera start/stop and bookkeeping consume the slots; actual inference
   and confirmed camera-OFF durations vary. Expected about 43–46 min
   including setup and idle, plus up to 15 min cold gate.

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase23.py --mode fixed --rehearsal ~/storage/downloads/campaign/owner_rehearsal_p23_02.json --output ~/storage/downloads/campaign/owner_fixed_p23_01.json > ~/storage/downloads/campaign/owner_fixed_p23_01.stdout 2> ~/storage/downloads/campaign/owner_fixed_p23_01.stderr)
```

3. Adaptive: 36 min; active until skin >= T_hi (default 37.0 C), pause until skin <= T_hi - 2.0 C;
   minimum active 60 s and minimum camera-OFF pause 30 s. `--t-hi` is recorded and accepts 25..43 C.
    Expected about 43–46 min plus up to 15 min cold gate. The final segment is truncated by the 36 min limit.
    If a pause reaches its low threshold with less than 60 s remaining, stay camera OFF to the deadline.

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase23.py --mode adaptive --t-hi 37.0 --rehearsal ~/storage/downloads/campaign/owner_rehearsal_p23_02.json --output ~/storage/downloads/campaign/owner_adaptive_p23_01.json > ~/storage/downloads/campaign/owner_adaptive_p23_01.stdout 2> ~/storage/downloads/campaign/owner_adaptive_p23_01.stderr)
```

For each session send all files actually created:
`STEM.json`, `STEM.stdout`, `STEM.stderr`, `STEM_llama-server.log`, `STEM_MODE.json`,
`STEM_MODE_idle.json`, `STEM_MODE_gate.json`, and every `STEM_MODE_active_NN.json` /
`STEM_MODE_pause_NN.json`. Endurance normally has only `_active_01.json`; fixed normally has
12 active and 12 pause files; adaptive counts vary. STEM is exactly `owner_endurance_p23_01`,
`owner_fixed_p23_01` or `owner_adaptive_p23_01`, and MODE is its corresponding command's mode.
Also send every `STEM_MODE_samples_NNNN.json`: atomic raw-sample checkpoints at most 60 s apart,
with 2 s overlap. They retain completed windows if the current active phase is hard-killed.

### Measurement and validity rules

After setup, camera-OFF idle lasts 300 s with llama-server loaded (YOLO/M2 sessions remain resident,
inference stopped). The camera-OFF cold gate then waits up to 900 s for skin <= idle-start skin + 1.0 C.
Its SAME final checked reading is both the gate decision and the active-start reading. If still hot,
run anyway as **WARM START — heat-timing results NOT VALID**. Camera start follows that reading and
its heat/time belongs to the active interval; detector/selector slot origins begin at camera readiness.
Every camera restart similarly belongs to active time. Fixed deadlines remain on the 180 s cycle grid;
the stop-camera transition consumes the beginning of the nominal pause and its actual OFF timestamp is
recorded. Fixed measured slots end at their grid boundaries, with the whole mode ending at +2160 s;
evidence completion and cleanup lag are recorded outside that measurement. This is not a claim of
120 s of inference or 60 s of confirmed OFF time. Adaptive's 60 s minimum active is counted from
camera-start invocation, including startup; its minimum pause counts 30 s only after stop is confirmed.
The 36 min deadline may truncate a terminal pause; no restart occurs with less than 60 s remaining.
Inference calls drain before camera stop; any call crossing a fixed active-slot boundary is a cadence
failure and its overrun is recorded in the pause slot's inference time. Gemma stays loaded. Sensor/status/caps/power/
memory sampling continues through every transition and pause. Actual phase/cycle seconds are recorded.
Every measured phase, cycle and the totals record `camera_start_s`, `inference_s`, `camera_stop_s`,
confirmed `camera_off_s` and `camera_on_without_inference_s`. Transition timestamps are retained.
The times are clipped to the measured phase/slot: `inference_s` is the interval from camera readiness
through the last completed YOLO/M2/selector call, including inter-call waits, rather than summed
parallel CPU execution times. Confirmed OFF time ends when the next camera-start invocation begins.
Camera-on-without-inference includes startup, stop and boundary idle; start/stop times are subsets
of it. Inference + confirmed OFF + camera-on-without-inference equals measured wall time.
`duty_fraction` uses `inference_s / measured wall seconds` for phases, cycles and totals;
`duty_description` explicitly describes the slots and variable actual inference/OFF durations.

P23 frame deadlines track the completed frame's capture time plus 1 s, preventing accumulation of the
camera's slightly-longer-than-1 s period. `SizePolicy(5)` still selects exactly one size per due frame.
Only complete service slots are planned: a YOLO slot needs 1.35 s H4 wait plus 2 s terminal inference budget
before the active deadline; a selector needs its complete 20 s slot. Partial terminal slots are explicitly
unscheduled in `slot_plan`, rather than reported as missing. Every planned slot remains required;
real skipped slots, missing selector/M2 calls or deadline overruns invalidate the phase.
There is no separate per-call YOLO duration limit; the budget only reserves the terminal tail.
An early stop is latched on the main thread. In-flight work drains, with its duration recorded;
later temperatures/status cannot erase the stop or conceal earlier missing calls.

60 s windows report due/done/skipped YOLO and ms, M2 calls/inference ms, selector calls/ms/misses,
time-weighted current_now watts (0.37 s sampler), skin/status, caps for each policy, memory/PSS, re-reads.
`read_re_reads` includes both the FIX4D empty-answer re-reads and FIX3D thermal retries;
`empty_answer_re_reads` and `thermal_re_reads` retain their separate counts.
Events use seconds since the first active start, including pauses: first policy4/policy6 window >=10 %
capped, first skin crossings 35/37/39/41/43 C, all sampled Android status changes, stop reason.
Caps retain power_map's later-sample ownership and no tail credit; `cap_tail_unknown_s` exposes the tail.

Cadence misses (including skipped slots and calls overrunning deadlines) invalidate results.
H4 is unchanged: wait 1.35 s for a new frame; skip a missing/repeated slot; stop at 3 consecutive misses
or any bad/other-session frame. Root/sensor/charger/process/cores/affinity/camera/inference/server,
memory, power-coverage, LMK or cleanup failure invalidates and ends the session. Battery below 25 %
stops through normal cleanup. #127 CRITICAL, battery >=45 C, CPU >=110 C in three 1 s samples retain
precedence. **All emergency stops are NOT VALID**; none is accepted here as a measured performance result.
Endurance SEVERE alone is a regular **VALID — STOPPED AT SEVERE**, subject to every other validity check.

Read re-read cap: at most 3 in EACH contiguous 180 s interval (short final interval has the same cap)
within idle, gate and the measured mode, including separate mode/final cleanup windows; setup retains
Phase 1's cap of 3. A burst cannot be hidden by
averaging it over a long run. Exceeding the cap invalidates evidence and fails rehearsal_pass.
This cap applies to the FIX4D empty-answer ledger, as in Phase 1; thermal retries retain
their existing FIX3D attempt/time bounds and are recorded separately without a new allowance.
FIX4D is unchanged: only an answer carrying nothing (stdout AND su stderr blank, a Popen launch failure,
or HTTP timeout) gets one re-read. Every non-empty answer goes once to its base parser. Thermal keeps
FIX3D, /health keeps 2-of-2, and the successful pidof still listing a camera pid keeps its owner re-check.

Only owner runs with all agents exited can produce timings. Preparation/mocks inside proot cannot reach
root and are NOT VALID. No DOC DIFF is prepared now; E4 defers documentation until after Phases 2/3.
The changed Phase-1 start gate also requires a new Phase-1 rehearsal before any future Phase-1 reuse;
the archived FIX4D results remain unchanged and do not validate the new gate wiring on hardware.


P23 review clarifications: selectors plan `floor(camera-ready remaining active seconds / 20)` complete
slots. With a 3 s camera start, a normal fixed active slot plans 5 calls (60 across 12 cycles), and a
40 s rehearsal active slot plans 1 (3 total); `slot_plan` records the actual count. Offline mock camera
transitions take 3 s START / 2 s STOP and use the same complete-slot formulas. Rehearsal still requires
live M2 and fallback M2/selector coverage in every mode; only the first active phase forces fallback.

Adaptive reserves 1 s beyond the minimum active duration before restarting, avoiding a short final
active due to bookkeeping between the restart decision and camera start. Final termination may cut
the last OFF pause; every restart requires at least 30 s confirmed OFF. A bracket emergency replaces
the stop reason and stop event, and invalidates the mode even if the active loop had reached SEVERE.

Memory/PSS reads wait for camera transitions under the camera-state lock; other sensor/power/safety
monitors continue. Main-thread bounded worker drain renews the heartbeat; worker threads cannot renew
it. BatteryStop from selector/M2 remains a clean battery stop. Camera failures include recovered empty
`am start` launch re-reads as well as camera wrapper retries; any failed attempt prevents rehearsal pass.

Atomic numbered 60 s checkpoints retain completed inference work as well as sensor samples, with
2 s overlap. Individual normal-session phase files retain IN PROGRESS labels; mode JSON and the final
main JSON are the authoritative final validity. Additional signals are ignored only during main's
final resource cleanup, not earlier per-phase cleanup. SEVERE stop, emergency stops, near-deadline
adaptive restart suppression and camera launch failures are tested offline; the live rehearsal covers
all normal active/pause paths and adaptive switching, not these forced failure scenarios.
