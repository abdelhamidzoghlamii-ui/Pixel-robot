# Operations history — dated phone evidence

These notes were moved from COMMANDS.md without changing their historical
claims. Verify phone files again before using an old recovery procedure.

## Historical recovery provenance

These notes preserve the 2026-09-05 HANDOFF and DECISIONS #73 evidence. The phone
archives have not been freshly inspected or verified in this documentation cleanup.

| Recovery item | Historical provenance and limitation |
|---|---|
| `~/build-b1609.tar.gz` | DECISIONS #73 records a 49 MB archive of llama.cpp commit `e1a1abb7`, version 1609, with all four critical files present and gzip integrity checked at that time. Restore the whole build tree; present archive integrity is unverified here. |
| `~/robot-preclaudecode.tar.gz` | The old handoff describes this as `~/robot` before any agent touched it. Exact contents, corresponding commit, and present integrity are UNKNOWN; no new verification is claimed. |
| Git commit `fe14be2` | Historical CP2102/nav/calibration snapshot, not the current working state and not an asserted identity for either archive. Read `git rev-parse HEAD` for the current checkout. |

DECISIONS #73 supersedes the old single-launcher rollback instructions:
`llama-server.swafix` was a 5.9 KB fragment that depended on shared libraries.
`llama-server.working` was a genuine earlier 12 MB static fallback, but predates
the SWA fix (#6); using it loses that fix. Another documented recovery route is
rebuilding llama.cpp from commit `e1a1abb7`.

Historical whole-tree restoration command from #73, for native Termux only after
verifying the archive and obtaining human authorization for replacing the build:

```bash
cd ~/llama.cpp && rm -rf build && tar xzf ~/build-b1609.tar.gz
```

The old handoff's `git checkout -- .` discarded unstaged tracked-file edits; it
did not restore the historical snapshot or recover ignored models. It is not a
routine recovery step. Identify and preserve current work and agree exact recovery
targets before any destructive Git operation.

The earlier handoff recorded about 420 MB of unused ONNX files, including
`yolo11x.onnx` (218 MB) and `yolov8m.onnx` (100 MB). Those are historical size
observations, not a fresh inventory. See [STATUS.md](STATUS.md) and `.gitignore`
for model/capture exclusions; adding ignore rules does not remove files already
in Git history.

## Historical agent environment (DECISIONS #82)

The prior Claude Code 2.1.261 Debian session could read/edit files and run non-root
Python, but could not use `su` or `/dev/bus/usb` (DECISIONS #82). The
`motors.py`, `teleop.py`, `run_mission.py`, `log_run.py`, `dist_raw.py`, and
`cp2102_test.py` hardware operations remain human-controlled in native Termux.
That session incorrectly called `Robot.run_mission()` dead after reading only
`main.py`; `run_mission.py` calls it. Scope cross-file investigations explicitly.
#82 also records why AVF was rejected: no USB host controller and only
`/mnt/shared` shared, rather than access to the robot checkout.
