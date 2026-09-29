# RobotCam review findings

Independent review findings on PR #1, by round and number, with how each was resolved,
so the next reviewer can check them one by one. "Fix round" names the commit that
contains the fix.

## Round 1 — review of 621b874

| # | Finding | Resolution | Fix round |
|---|---|---|---|
| 1 | Output location in `Download/`: unclear whether the app can write there and Termux can read it without root | Location kept. Device-verified on the Pixel 7 (2026-09-28): the app writes `/sdcard/Download/robotcam/`, Termux reads it without root. Recorded in README "Output location". | 8817a47 (README) |
| 2 | Stale frames: old files could survive a stop or camera error, and the writer could publish after stop (join-timeout race) | Published files are deleted before every start and camera restart, on any camera error or disconnect, and on stop. Publishing and deletion share one lock and a `publishing` flag, and a frame is only written if it is still the current one, so nothing is published once stop begins. Random `session_id` per start in the sidecar and the JPEG comment. | 8817a47 |
| 3 | Retry bypass: exceptions in session/request calls inside callbacks escaped the retry path | Every camera callback path catches failures and goes through `failed()`: unpublish, close all resources, reopen after 2 s. Extended in round 2 (findings 1 and 2). | 8817a47 |
| 4 | Pair mismatch: `frame.jpg` and `frame.json` are renamed separately, so a reader can pair an image with another frame's sidecar | First a session+frame pair check (8817a47). The Pixel 7 retest of e21c970 at 2 frames/s showed 19/30 reads caught by it, so the contract became JPEG-only: the reader reads `frame.jpg` alone (one rename, one whole frame) and takes session, frame and capture time from the JPEG comment; `frame.json` is diagnostics only. Retest at 2 frames/s: 30/30 ok, age median 0.38 s. | 8817a47, then 3f8b527 |
| 5 | Exported components: `StartActivity` and `ControlReceiver` are exported without a permission | Kept; the owner accepts the risk. Documented in README "Exported components (accepted risk)": another app can start or stop the camera; stopping only removes frames, and the reader rule turns a missing, old, unreadable or other-session frame into "stop", never "clear". | 8817a47 (README) |
| 6 | Test script aborted on a malformed or partial sidecar | Malformed input counts as a bad sample and never aborts the run. Since 3f8b527 the sidecar is only read by the optional `--check-sidecar` diagnostic, which also catches its errors; an undecodable JPEG or malformed comment is a `bad` sample. | 8817a47, 3f8b527 |

## Round 2 — review of 3f8b527

| # | Finding | Resolution | Fix round |
|---|---|---|---|
| 1 | Obsolete camera-open attempts: a mode/rate change while `openCamera()` is pending starts a second open; stale callbacks can configure against the new reader or close and unpublish the active camera | Each open attempt has a generation number (`gen`); `closeCamera()` starts a new one, and an `opening` flag blocks a second open while one is pending (except on a restart, which clears it: see round 3, finding 2). Device, session and capture callbacks capture their generation; a callback from an obsolete attempt closes what it was handed (camera, session) and returns without touching the current camera or unpublishing. | daa1c25 |
| 2 | Image callbacks: `acquireLatestImage()` and the YUV copy were uncaught on the camera thread, including callbacks queued for a closed reader | `onFrame` and `onPreview` ignore callbacks from a reader that is no longer current, and catch any exception from acquiring or copying the image; it goes through `failed()` (unpublish, close, retry). No path lets it crash the process. | daa1c25 |
| 3 | Age check used the wall clock | The JPEG comment carries `capture_boot_ms` (sensor timestamp on the `elapsedRealtime` / `CLOCK_BOOTTIME` base, or the arrival time on the same clock if the camera's timestamps are not realtime-based, marked `clock=sensor` or `clock=arrival`); `capture_wall_ms` stays for humans. The reader computes age with `time.clock_gettime(time.CLOCK_BOOTTIME)` and treats age < -0.2 s or > 2 s as "stop". README reader rule and `robotcam_test.py` updated. | daa1c25 |

### Round 2 device evidence (daa1c25, Pixel 7)

- Mode B: 15/15 ok. Mode A at rate 2: 20/20 ok.
- Switch storm, 6 starts 0.3 s apart: the process stayed alive; afterwards 20/20 ok.
  This shows recovery, not uninterrupted capture (see round 3, finding 2).
- The output folder was empty after every stop.
- `clock=sensor`: the Pixel 7's camera timestamps are on the boot-clock base.
- Not measured: RAM use. `logcat -s RobotCam` returns nothing on this phone (also
  via `su`), so the log lines could not be checked on the device.

## Round 3 — final review of daa1c25

| # | Severity | Finding | Resolution | Fix round |
|---|---|---|---|---|
| 1 | medium | `robotcam_test.py`: numeric fields from the JPEG comment were converted outside the error handler (an overlong digit field raised `ValueError`); the sidecar check could raise `OverflowError` on `{"frame":1e999}` | Digit fields are length-bounded in the comment regex (frame ≤ 12, times ≤ 15 digits, session ≤ 32 hex); the conversions sit in a handler that turns `ValueError`/`OverflowError` into a `bad` sample; the sidecar check also catches `OverflowError`. Synthetic checks added for 5000-digit frame and boot fields, overlong wall and session fields, and sidecars with `1e999`, `NaN` and a 5000-digit integer: all end as `bad` / "sidecar unreadable", none aborts. | the commit that adds this section |
| 2 | low | A restart clears `opening` and calls `openCamera()` at once, so two opens can overlap | **Not changed (owner decision); known limitation**, documented here and in README "Known limitations". Callbacks of the superseded open are rejected by the generation checks (round 2, finding 1) and close their camera; if the overlap makes the new open fail, it ends in the normal error path (unpublish, close, retry after 2 s). The storm test above showed recovery, not uninterrupted capture. | — |
| 3 | low | Round-2 device evidence missing from this file | Recorded above under "Round 2 device evidence", including that RAM is unmeasured and logcat returns nothing on this phone. | the commit that adds this section |
