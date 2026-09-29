# robotcam — Termux-side test of the RobotCam service

`robotcam_test.py` reads the frames that the RobotCam app (`android/robotcam/`)
writes to `~/storage/downloads/robotcam/` and reports read and decode time, frame
age, and missing (including older than 2 s), unreadable, other-session, repeated or
skipped frames. It reads `frame.jpg` alone and takes session, frame and capture time
from the JPEG comment; `--check-sidecar` adds an optional `frame.json` cross-check
that never changes a read's status. Its `read_frame()` is the reader rule the robot
should use.
Start/stop commands and install steps are in `android/robotcam/README.md`.

```bash
python benchmark/robotcam/robotcam_test.py -n 60 --out ~/robotcam_test.json
```

Runs: none yet. Timed runs follow `docs/WORKFLOW.md` ("Timed benchmarks on the
phone") and are archived here per `benchmark/INDEX.md`.
