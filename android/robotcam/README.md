# RobotCam — camera service for the Termux prototype

A small helper app that replaces `termux-camera-photo` in the Termux prototype.
`termux-camera-photo` reopens the camera on every call, waits 0.5 s for a preview and
always shoots 12 MP (~3 s per call, 3 of 20 calls wrote no usable file). RobotCam
keeps the back camera open and publishes one small, fresh JPEG about once a second.

It is **not** the product Android app (`docs/APP_CLAUDE.md`, `docs/APP_STATUS.md`)
and is not built from that app's module layout. It contains no robot or motor code.

## What it does

- Foreground service (type `camera`) with an ongoing notification that shows the
  chosen size, frame-rate range, frames written and write errors.
- Back camera, Camera2, JPEG at the supported size nearest to 640x480, lowest
  supported AE frame-rate range, `JPEG_ORIENTATION` = sensor orientation (the same
  as `termux-camera-photo` with the phone upright). The chosen values are in the
  notification, in the sidecar and in `logcat -s RobotCam`.
- A repeating request streams JPEGs; only the newest frame is kept in memory.
- About once a second (only if a newer frame arrived) it writes, each through a
  `*.tmp` file and `rename()`:
  - `frame.jpg` — the JPEG, with a JPEG comment `robotcam frame=N capture_wall_ms=T`
  - `frame.json` — the sidecar, written after the image:
    ```json
    {"frame":N,"capture_wall_ms":T,"written_wall_ms":W,"session_start_ms":S,
     "width":640,"height":480,"bytes":B,"fps_min":a,"fps_max":b,"timestamp_source":"sensor"}
    ```
    `frame` counts writes since the service started (`session_start_ms`).
    `capture_wall_ms` is the sensor timestamp converted to wall-clock ms when the
    camera reports a realtime timestamp base (`"sensor"`), otherwise the arrival
    time of the frame (`"arrival"`).
  The two renames are not one atomic step. A reader that must be sure image and
  sidecar match compares the JPEG comment's `frame=` with the sidecar.
- On stop the camera is closed and both files are deleted, so a reader never
  mistakes the last frame for a live one.
- If the camera is lost (another app took it, an error) it retries every 2 s.

Permissions: `CAMERA`, `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_CAMERA`,
`POST_NOTIFICATIONS`. No `INTERNET`, no storage permission, no wake lock.
No dependencies: Android platform APIs and the Kotlin standard library only.

## Output location: `/sdcard/Download/robotcam/`

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
# start (brings up an invisible activity for a moment, then returns to Termux)
am start -n com.pixelrobot.robotcam/.StartActivity

# stop (camera off, files removed)
am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP

# is it running / what did it choose (the file is absent while stopped)
cat ~/storage/downloads/robotcam/frame.json

# test (timed: run with no agent resident, see docs/WORKFLOW.md)
python benchmark/robotcam/robotcam_test.py -n 60 --out ~/robotcam_test.json
```

Start uses `am start` on an activity, not `am startservice`/`am broadcast`: a
camera-type foreground service may only use the camera if it was started while the
app was in the foreground (Android 11+ while-in-use rule, enforced at
`startForeground()` on Android 14). Launching the activity puts the app in the
foreground for that moment; the service keeps camera access after the activity
closes. Starting an activity from Termux requires Termux to be in the foreground
(screen on, Termux visible) or to have "Display over other apps".
Diagnostics: the notification, or `adb logcat -s RobotCam` from a computer
(`logcat` inside Termux without root shows only Termux's own log).

## Build

CI: `.github/workflows/robotcam.yml` builds the debug APK on pushes that touch
`android/robotcam/` and uploads it as an artifact. No signing secrets: the debug
build is signed with the runner's throwaway debug key, so each CI build has a
different signature and an update over a previous CI build fails with
`INSTALL_FAILED_UPDATE_INCOMPATIBLE` — uninstall first (and then delete the output
folder, see above).

Local: `gradle -p android/robotcam assembleDebug` with Gradle 8.14, JDK 17 and an
Android SDK with platform 35.

Toolchain: AGP 8.7.3, Kotlin 2.1.0, compileSdk/targetSdk 35, minSdk 29.
