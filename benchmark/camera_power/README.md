# CAMPOWER1/2 — camera power archive

Camera-only research benchmark: battery watts, CPU seconds by process group, frame cadence/age, startup/stop checks, requested/applied Camera2 diagnostics, luma/sharpness and saved example frames. No YOLO, Gemma or selector workloads are started. Image metrics alone do not establish detection quality. [RESEARCH.md](RESEARCH.md) retains research notes and labelled guesses.

Requires the RobotCam test build from branch `robotcam-camera-power`, CAMPOWER2 commit `90ef439`. That branch stays unmerged; ARCHIVE3 does not modify it or include an APK. CAMPOWER1 smoke outputs use the earlier test build and runner; their run.json records runner `d30dd04c…`, while current CAMPOWER2 source and its later runs use `fdb6380e…`. This is historical version provenance, not a mismatch with the final CAMPOWER2 coder report. The earlier CAMPOWER1 runner snapshot is not present in this archive; its report and run hashes are retained.

Human runs only, native Termux, motors off, unplugged, screen on and Termux in front, fixed scene and lighting; exit agents first. The original reviewed instructions are preserved below. The full recorded run explicitly selected six blocks: baseline_first, manual_200, manual_500, mode_A_manual_500, mode_A, baseline_last, 120 s each; default eight-block instructions below describe available runner defaults, not that recorded invocation.

The full run is valid except manual_500: one gap is 0.01 s over tolerance. Smoke manual_1000 and mode_A_manual_1000 sit at the 1 fps edge and are INCOMPLETE. Dark-room CAMPOWER1 smoke (luma about 4) is not representative. Runner polling at 10 Hz adds CPU load equally across blocks, so these watts are not comparable to DUTY1. No robot camera choice is inferred by ARCHIVE3.

Archive prepared by ARCHIVE3 without staging, commit or push. Source code is unchanged; report hashes were checked before writing archive documentation. Run files and reports were copied byte for byte. `runs/thermal.log` covers multiple sessions and is not assigned to a single run. Zone9/10/11 are CPU-core readings, not a heat state. Skin, Android status and cpufreq caps carry the heat evidence. Saved JPEGs are owner-approved public benchmark photos.

See [RUN_INDEX.md](RUN_INDEX.md) for fixed validity labels, run/log matches and block lists; [ARTIFACTS.md](ARTIFACTS.md) for sizes, hashes, sources and exclusions. Independent ARCHIVE3 review is retained verbatim in the Downloads coder report; it checks archival completeness and labels, not new code behaviour. Historical coder/reviewer reports in `reports/` retain their original verdicts and limitations.

## Fixed run labels (verbatim task text)

- smoke console 20261003T003910Z: SMOKE, dark room (luma ~4), CAMPOWER1 build; not representative.
- smoke console 20261003T062323Z: SMOKE, lit room, CAMPOWER1 build.
- smoke console 20261003T073038Z: SMOKE, lit, CAMPOWER2 build; manual_1000 / mode_A_manual_1000 INCOMPLETE (1 fps at its edge).
- full console 20261003T075659Z: VALID except manual_500 INCOMPLETE (one gap 0.01 s over tolerance); CAMPOWER2 build 90ef439. Runner 10 Hz polling adds CPU load equally to all blocks; watts not comparable to DUTY1.

## Preserved CAMPOWER2 run instructions

# CAMPOWER2 — camera only, human runs

Test APK: branch `robotcam-camera-power`, GitHub Actions workflow **robotcam debug
APK**, artifact **robotcam-debug-apk**. Unzip and install `app-debug.apk`.
If signature mismatch (`INSTALL_FAILED_UPDATE_INCOMPATIBLE`), uninstall the old
RobotCam, install again, then delete `~/storage/downloads/robotcam` from Termux
once (old shared files belong to the old install). Open RobotCam once and grant
Camera/Notifications, then stop it. No robot choice has been made.

Motors off, no USB/serial. Coder has not run hardware. Quit every agent, exit
Debian/proot, stop robot-chat/chat.py, llama-server and any benchmark. Unplug the
charger, keep native Termux in front, screen on, phone fixed in the same indoor
scene/mount. Do not change lighting, framing or phone pose between blocks.

```bash
bash ~/robot/benchmark/camera_power/oneshot.sh --smoke
bash ~/robot/benchmark/camera_power/oneshot.sh
bash ~/robot/benchmark/camera_power/oneshot.sh --resume ~/camera_power/run_<UTC>
# smoke resume also needs --smoke
```

The launcher reuses DUTY1's source at runtime: root logger, screen timeout
read/set/restore, 5 min idle, Discharging checks before/after idle, resident-agent,
server and runner refusals, signals and cleanup. It adds camera_power to the
competing-runner refusal and changes only launcher paths. Helpers are imported
from DUTY1/POWERMAP/CORESIDENCY. Their detector/server modules load at import;
this runner never constructs a detector, loads weights, starts Gemma or a selector.
It never imports motors.py. Existing NumPy/Pillow only, no new dependencies.

Default block order: baseline with **no extras**, manual B 200 ms, B 500 ms,
B 1000 ms, A rate 1/manual 1000 ms, A rate 1/manual 500 ms, A rate 1/auto,
baseline with no extras again. **8 x 120 s = 16 min block time**. Printed wall
estimate: **21–85 min** including 5 min idle and up to 8 min skin gate per block,
plus startup/stop/query overhead. Smoke: 8 x 20 s = 2 min 40 s; 7 min 40 s
including idle plus overhead, no gate wait, **SMOKE: not a measurement**.

Optional switches remain available by explicit block name; `--blocks` accepts
comma-separated names in the given order. Names: baseline_first, manual_200,
manual_500, manual_1000, mode_A_manual_1000, mode_A_manual_500, mode_A,
baseline_last, preview, record, fast, off, focus_1m, dump, other_camera.
`other_camera` requires `--camera-id ID`; supplying the ID alone does not add a
block. Camera IDs must be independently enumerated BACK cameras; physical-only
IDs are not selectable. The supplied dump exposes only ID 0 back and ID 1 front.

```bash
bash ~/robot/benchmark/camera_power/oneshot.sh --smoke --blocks dump
bash ~/robot/benchmark/camera_power/oneshot.sh --blocks baseline_first,preview,record,fast,off,focus_1m,baseline_last
# Only if an independently enumerated alternative BACK ID becomes available:
bash ~/robot/benchmark/camera_power/oneshot.sh --blocks baseline_first,other_camera,baseline_last --camera-id ID
```

Characteristics are copied from Download after an explicit dump block.
No camera off-between-shots variant. Custom block lists print their own estimate;
keep full-run selections within the requested ~24 min block-time budget.

Outputs in `~/camera_power/run_<UTC>[_smoke]`: run/helper hashes, atomic block
JSON, `report.txt`, per-frame luma/sharpness/age and paired diagnostics, three
images per block (start, middle, end). Saved images are the validated upright
JPEG decoded and re-encoded at quality 95, not original JPEG bytes. Image metrics
use 320x320 bilinear grayscale (aspect stretched consistently), interior 4-neighbour
Laplacian variance. Noise can increase sharpness: inspect images, especially dark
scenes. No YOLO-quality claims follow from luma/sharpness alone.

Power uses DUTY1's 0.5 s battery sampler and interpolated trapezoidal integral;
no extrapolation or gaps >1.5 s. Fuel-gauge averaging/sign calibration remains
unverified. Initial sampler sliver excluded; startup is included. Stop latency
includes a 3 s no-advancing-frame check and force-stop, rather than pure close time.
CPU uses provider/app identity-aware proc-stat snapshots every 5 s and boundaries;
query overhead/timing skew uncorrected. Short-lived processes between snapshots
can be missed. Saved raw identities permit inspection; missing groups are n/a.

Limits/gates/end-check/force-stop are the reviewed helper rules: skin <= original
idle +1.5 C, max 8 min then WARM START; CRITICAL Android status, battery >=45 C,
CPU >=110 C three consecutive 1 s samples; stale sensors fail closed. zone9/10/11
are CPU fault readings, not heat state. Core loss discards/redoes a block. Other
failures refuse/abort, retaining completed block JSONs. Completed INCOMPLETE
blocks are retained on resume; move their block JSON aside to redo. Resume requires
same smoke/--blocks/camera ID and hashes; the interrupted block is redone. Application
options are checked against frame.json; an old APK lacking `power.schema_version=2` is refused.
`startup_wait_polls` counts rejected polls before the first good frame.
`frames_failed` counts rejected polls after that frame, **not distinct failed
captures**. `raw_status_counts` preserves every reader status. In CAMPOWER1,
`frames_failed` summed ALL non-ok/non-repeat polls, including startup absence:
the ordinary smoke blocks' 16–17 failures (mode A: 6) were all startup polls.
`missing` combines absent JPEG, age >2 s and capture predating block start;
`bad` means decode/comment/clock failure; `other_session` means restart. The
JPEG reader contract is unchanged. Repeats and unpaired diagnostic sidecars
are separate counts, not failures. A stale repeated JPEG still fails the reader's
age check before repeat classification; this remains a post-start failure.
Observed new-frame rate and maximum poll-to-new-frame gap are reported; a gap
over 1/rate + min(frame_ms/1000, 0.5) + 0.35 s marks INCOMPLETE. Poll timestamps
include decoding/metrics overhead and are not exact sensor arrival timings. Result metadata describes the latest completed request (often preview),
not necessarily the paired JPEG's sensor result. Null means unavailable; a requested
slow duration is not proof of a slow sensor: inspect result duration and AE mode.

Offline only:

```bash
python3 benchmark/camera_power/test_camera_power.py
```

Research and labelled guesses are in [RESEARCH.md](RESEARCH.md). Build/HAL/manual
cadence, other-camera geometry, image quality and all savings need human validation.


CAMPOWER2 manual B repeats its chosen capture template directly to the publish
surface; no periodic still queue behind slow preview frames. A/B manual paths
share anchored pacing. Exposure never exceeds the converged value or requested
frame duration; post-RAW boost is copied when supported/reported, with requested
and RESULT values logged. The 30 s AE re-convergence remains. Hardware cadence,
brightness, power and template behaviour require another human smoke run.
Do not resume CAMPOWER1 outputs after this code change; preserve their evidence.
