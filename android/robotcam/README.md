# RobotCam — camera service for the Termux prototype

A small helper app that replaces `termux-camera-photo` in the Termux prototype.
`termux-camera-photo` reopens the camera on every call, waits 0.5 s for a preview and
always shoots 12 MP (~3 s per call, 3 of 20 calls wrote no usable file). RobotCam
keeps the back camera open and publishes one small, fresh JPEG about once a second.

It is **not** the product Android app (`docs/APP_CLAUDE.md`, `docs/APP_STATUS.md`)
and is not built from that app's module layout. It contains no robot or motor code.

## What it does

- Foreground service (type `camera`) with an ongoing notification that shows the
  mode, chosen size, frame-rate range, frames written and write errors.
- Back camera, Camera2, lowest supported AE frame-rate range.
- Frames come from a `YUV_420_888` stream and are compressed to JPEG in the app
  (Android `YuvImage`, quality 85), only for frames that are published. The camera's
  own JPEG path is not used: on the Pixel 7 its smallest JPEG size is 1920x1080.
- Frame size rule, applied to the camera's YUV sizes: exactly 640x480 if offered;
  otherwise the smallest 4:3 size of at least 640x480; otherwise the smallest size at
  least 640 wide; otherwise the largest size. The full list of YUV sizes, the chosen
  size and the rule that picked it are logged and written into the sidecar.
- Orientation: pixels are not rotated. An EXIF Orientation tag derived from the
  sensor orientation (90° → 6) is written into the JPEG, so PIL's
  `ImageOps.exif_transpose` (used by `detect_person.py`) turns it upright, as with
  `termux-camera-photo` and the phone upright.
- Two capture modes, chosen at start with the intent extra `mode` (default `B`):
  - `A` — a repeating request streams YUV frames at the frame size; one frame per
    publish period is copied out and published.
  - `B` — a repeating preview to a small YUV surface (same aspect ratio, frames
    dropped) keeps exposure and focus converged; one still capture per publish
    period goes straight to the YUV frame surface and is published as soon as it
    arrives.
  Only the newest frame is kept in memory in both modes.
- Publish rate: intent extra `rate`, 1 (default) or 2 frames per second; any other
  value falls back to 1.
- Each frame is published as two files, each through a `*.tmp` file and `rename()`:
  - `frame.jpg` — the JPEG, with a JPEG comment
    `robotcam session=<session_id> frame=<N> capture_boot_ms=<B> capture_wall_ms=<W> clock=<sensor|arrival>`
  - `frame.json` — a diagnostics sidecar, written after the image (not part of the
    reader contract, see below):
    ```json
    {"session_id":"9f2c…","frame":N,"capture_boot_ms":B,"capture_wall_ms":W,
     "written_wall_ms":X,"mode":"B",
     "rate":1,"width":640,"height":480,"size_rule":"exact 640x480","jpeg_quality":85,
     "encode_ms":E,"exif_orientation":6,"bytes":B,"fps_min":a,"fps_max":b,
     "clock":"sensor","yuv_sizes":["4080x3072",…,"640x480",…]}
    ```
    `session_id` is random per service start; `frame` counts publications within
    that session. `capture_boot_ms` is the capture time on the boot clock
    (`SystemClock.elapsedRealtime`, Linux `CLOCK_BOOTTIME`): the sensor timestamp when
    the camera reports a realtime timestamp base (`clock=sensor`), otherwise the
    arrival time of the frame in the app (`clock=arrival`). `capture_wall_ms` is the
    same instant on the wall clock, for humans only.
- Unpublishing: both files are deleted before every start and camera restart, on
  any camera error or disconnect, and on stop. Once stop begins nothing more is
  published (publishing and deletion share one lock and a stop flag).
- If the camera is lost (another app took it, an error, any exception in a camera
  callback) it closes everything and retries every 2 s.
- Log: `Log.i` under tag `RobotCam` (size list, chosen size and rule, mode, rate,
  fps range, errors prefixed `ERROR`), plus a heartbeat line on the first and every
  30th published frame.

## Reader rule (for the robot)

The reader contract is **`frame.jpg` alone**. It is replaced by a single `rename()`, so one
read always gets one whole frame, and its JPEG comment carries everything the reader
needs: `robotcam session=<session_id> frame=<N> capture_boot_ms=<B> capture_wall_ms=<W> clock=<sensor|arrival>`. `frame.json` is
diagnostics only and not part of the contract: it is renamed after the JPEG, so a
reader that reads both can see a new JPEG with the previous sidecar (the Pixel 7 retest
of e21c970 at 2 frames/s hit this in 19 of 30 reads).

A frame is usable only if all of these hold; otherwise treat it as **no frame**:

1. `frame.jpg` exists and reads completely;
2. it decodes, and its comment matches
   `robotcam session=<hex> frame=<digits> capture_boot_ms=<digits> capture_wall_ms=<digits> clock=<sensor|arrival>`;
3. the session is the one the robot expects (pin it on the first good frame; a new
   session means the service restarted);
4. its age on the boot clock, `CLOCK_BOOTTIME now - capture_boot_ms` (Python:
   `time.clock_gettime(time.CLOCK_BOOTTIME)`), is between -0.2 s and 2 s. The wall
   clock is not used: it can step (NTP, manual change). A negative age beyond -0.2 s
   means the clocks disagree and is also "stop".

Frames from builds before this format (no `capture_boot_ms`) fail rule 2.

`read_frame()` in `benchmark/robotcam/robotcam_test.py` implements this rule.

**The robot must treat a missing, old, unreadable or other-session frame as "stop",
never as "clear".** The files can disappear at any moment (stop, camera error), and
the start activity and stop receiver are exported without a permission (next
section), so another app on the phone can switch the camera off.

## Exported components (accepted risk)

`StartActivity` and `ControlReceiver` are exported with no permission, so Termux
can reach them. Any app on the phone can therefore start the camera service
(through the visible activity) or stop it. The owner accepted this risk for a
prototype helper. It cannot make the robot move: stopping only removes frames,
which the reader rule above turns into "stop".

Permissions: `CAMERA`, `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_CAMERA`,
`POST_NOTIFICATIONS`. No `INTERNET`, no storage permission, no wake lock.
No dependencies: Android platform APIs and the Kotlin standard library only.

## Output location: `/sdcard/Download/robotcam/`

**Device-verified 2026-09-28 on the Pixel 7:** the app writes here and Termux
reads the files without root.
Retest of 8817a47 (camera JPEG path, 1920x1440 frames): 40/40 frames ok in modes A
and B, stop cleared the files.
Retest of e21c970 (in-app 640x480): rate 1, modes A and B: 40/40 ok, exact 640x480,
encode 2-7 ms, decode ~9 ms. Rate 2 exposed the JSON/JPEG read race that led to the
JPEG-only reader rule below.

Termux sees it as `~/storage/downloads/robotcam/` (after `termux-setup-storage`).

Why this directory, without root:

- The app's private directories (`/data/data/<pkg>`, and `Android/data/<pkg>` on
  Android 11+) are not readable by Termux.
- On Android 11+ an app may create files of any type in the shared `Download/`
  directory by plain file path, with no storage permission, and may rename its
  own files there. `Pictures/` and `DCIM/` would not accept the `.json` sidecar or
  the `.tmp` names.
- Termux reads shared storage after `termux-setup-storage`. On Android 11+ that
  asks for "All files access" for Termux, which it needs to see files another app
  created in `Download/`. This is an ordinary permission grant, not root.

Limits:

- Android 10 (API 29, the minSdk) does not allow the plain-path write without a
  storage permission, so on Android 10 the notification shows write errors. The
  target phone runs Android 14+.
- Files in `Download/` belong to the app that created them. After an **uninstall**
  and reinstall the new install cannot replace the old files (write errors,
  `EACCES`). Delete the folder once from Termux: `rm -r ~/storage/downloads/robotcam`.
  Updating the app in place (`adb install -r` / installing a newer APK over it)
  keeps ownership.

## Install on the phone

1. Download the `robotcam-debug-apk` artifact from the GitHub Actions run
   (workflow "robotcam debug APK") and unzip it to get `app-debug.apk`.
2. Copy it to the phone and open it (allow "install unknown apps" for the file
   manager or browser when asked), or `adb install -r app-debug.apk`.
3. Open **RobotCam** once from the launcher and grant Camera and Notifications.
   The service starts and the notification appears.
4. In Termux, once: `termux-setup-storage` and allow access (All files access on
   Android 11+). Termux's Python needs Pillow for the test script
   (already used by `detect_person.py`).

## Termux commands

```bash
# start, mode B (default: preview + one still per second)
am start -n com.pixelrobot.robotcam/.StartActivity
# start, mode A (repeating YUV stream); a running service switches mode/rate in place;
# a plain start resets it to mode B, rate 1
am start -n com.pixelrobot.robotcam/.StartActivity --es mode A
# publish twice per second (combine with --es mode A as needed)
am start -n com.pixelrobot.robotcam/.StartActivity --ei rate 2

# stop (camera off, files removed)
am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP

# is it running / what did it choose (the file is absent while stopped)
cat ~/storage/downloads/robotcam/frame.json

# test (timed: run with no agent resident, see docs/WORKFLOW.md)
python benchmark/robotcam/robotcam_test.py -n 60 --out ~/robotcam_test.json
# at rate 2
python benchmark/robotcam/robotcam_test.py -n 60 --interval 0.5 --out ~/robotcam_test_r2.json
```

Start uses `am start` on an activity, not `am startservice`/`am broadcast`: a
camera-type foreground service may only use the camera if it was started while the
app was in the foreground (Android 11+ while-in-use rule, enforced at
`startForeground()` on Android 14). Launching the activity puts the app in the
foreground for that moment; the service keeps camera access after the activity
closes. Starting an activity from Termux requires Termux to be in the foreground
(screen on, Termux visible) or to have "Display over other apps".
Diagnostics: the notification and `frame.json`. `logcat` is not a reliable
diagnostic here: on the Pixel 7, `logcat -s RobotCam` showed nothing, from Termux
and also via `su`, even after the heartbeat line was added in e21c970. The cause is
not known (not investigated further, by decision). The service logs with `Log.i`
under tag `RobotCam`; `adb logcat -s RobotCam` from a computer is untested.

## Known limitations

- **Overlapping opens on restart.** A mode/rate change or camera error closes the
  camera and opens it again at once (a restart does not wait for a pending open), so
  two opens can briefly overlap. Callbacks from the superseded open are rejected by
  the attempt-generation checks and close their camera; if the overlap makes the new
  open fail, it goes through the normal error path (files deleted, retry after 2 s).
  Expect a short gap in frames, not uninterrupted capture: in a storm of 6 starts
  0.3 s apart on the Pixel 7 the service recovered and then delivered 20/20 good
  frames. Not changed by owner decision (FINDINGS.md, round 3, finding 2).
- **No wake lock** (not a permitted permission): capture with the screen off is
  untested.
- **RAM use** is not measured.
- **logcat** returns nothing for `RobotCam` on the Pixel 7, also via `su` (see
  Diagnostics).

## Build

CI: `.github/workflows/robotcam.yml` builds the debug APK on pushes to `main` or
the PR branch that touch `android/robotcam/`, and uploads it as the artifact
`robotcam-debug-apk`. No signing secrets: the debug
build is signed with the runner's throwaway debug key, so each CI build has a
different signature and an update over a previous CI build fails with
`INSTALL_FAILED_UPDATE_INCOMPATIBLE` — uninstall first (and then delete the output
folder, see above).

Local: `gradle -p android/robotcam assembleDebug` with Gradle 8.14, JDK 17 and an
Android SDK with platform 35.

Toolchain: AGP 8.7.3, Kotlin 2.1.0, compileSdk/targetSdk 35, minSdk 29.


## CAMPOWER1 test branch only

These runtime extras are experiments, not decided for the robot. Omitting all
new extras keeps the same requests, targets, sizes and publish scheduling. The
JPEG reader contract, clocks and unpublish/generation checks remain unchanged.
Result callbacks add diagnostic observation overhead, not request changes.

| Extra | Type / values | Default |
| --- | --- | --- |
| capture_template | `--es`: still, preview, record (B shots only) | still |
| processing | `--es`: default, fast, off | default (no overrides) |
| focus_diopters | `--ef`: finite 0 to camera minimum-focus-distance limit | omitted (template AF) |
| frame_ms | `--ei`: 0 or 200–500 | 0 (template AE and FPS) |
| camera_id | `--es`: enumerated BACK camera ID | omitted (same first BACK ID) |
| dump_characteristics | `--ez`: true/false | false |

`processing` requests supported OFF/FAST NR, edge, hot pixel, aberration and
shading (off falls back to FAST if OFF unavailable); tone-map FAST, face detection
OFF, lens OIS and video stabilization OFF when advertised. Unavailable controls
are left at template defaults, visible in requested/result fields.

`frame_ms` requires MANUAL_SENSOR and sufficient maximum duration. AE warms up
for at least 1 s and until CONVERGED or FLASH_REQUIRED, then copies result exposure
and ISO and requests AE OFF with a longer frame duration. Every 30 s it restores
AE until convergence. Exposure is preserved, not increased to fill the period;
AWB/AF retain template behaviour unless fixed focus was explicitly supplied.
The HAL may clamp/ignore duration: inspect RESULT, never infer fps from the extra.
Non-convergence stays automatic and is labelled `ae_converging`, not manual.
Unsupported options fail through unpublish/retry; invalid extra syntax stops.

Example (human only, native Termux):

```bash
am start -n com.pixelrobot.robotcam/.StartActivity --ei frame_ms 500
am start -n com.pixelrobot.robotcam/.StartActivity --es processing off
am start -n com.pixelrobot.robotcam/.StartActivity --ez dump_characteristics true
```

A plain start resets all options. A changed option restarts the camera through
the existing generation checks. `frame.json` now always contains `variant` and
`power.options`, `power.requested`, `power.result`, camera ID, result frame number,
sensor timestamp and manual phase. These are the **latest completed request**
observation (possibly preview/previous frame), not a promise of a JPEG-matched
result. Null means unreported/unrequested. The sidecar itself must be paired with
JPEG session/frame for benchmark diagnostics; robot readers still read JPEG alone.
The optional all-ID dump is `Download/robotcam/characteristics.json`, refreshed
on camera open/retry when asked, never a frame or reader dependency.

CI builds `robotcam-camera-power`, artifact `robotcam-debug-apk`; a new CI debug
signature usually needs uninstall/reinstall and deletion of the old output folder
from Termux to restore shared-storage ownership. Hardware and savings unverified.
