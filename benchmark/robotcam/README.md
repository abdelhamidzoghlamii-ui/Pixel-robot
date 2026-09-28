# robotcam — Termux-side test of the RobotCam service

`robotcam_test.py` reads the frames that the RobotCam app (`android/robotcam/`)
writes to `~/storage/downloads/robotcam/` and reports read and decode time, frame
age from the sidecar, and missing, repeated, stale, skipped or mismatched frames.
Start/stop commands and install steps are in `android/robotcam/README.md`.

```bash
python benchmark/robotcam/robotcam_test.py -n 60 --out ~/robotcam_test.json
```

Runs: none yet. Timed runs follow `docs/WORKFLOW.md` ("Timed benchmarks on the
phone") and are archived here per `benchmark/INDEX.md`.
