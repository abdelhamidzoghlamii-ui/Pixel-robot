# Coder report — camera heat benchmark

Base: `main` `af5793938f238a17d2cd3bb9a4edccd232dda2ed`. No commit, push, or 180-second timed run.

## Candidate

- `benchmark/camera_heat/camera_heat.py` and `run_camera_heat.sh` implement four blocks in order, with a native Termux helper for root sensors, Android camera commands, and PSS, while the oneshot runner schedules 180-second windows and the idle + 4 °C gate.
- `run_camera_heat.sh` was copied to `~/ladder/` and is executable. Repository and installed copies have SHA-256 `4058b71094f408b2fd901dff6865f7d7e87f9ab7309ab8ce8c86e361881e4dae`. `camera_heat.py` SHA-256 is `63275fa0f967e0905c662d1a2daad21bc1b0fed9e39270d85c6ce6906eb95059`.
- Installed `~/ladder/oneshot.sh` SHA-256 is `db1f315681892f0c1f89de92b86d7495a47cc171a3c03a170fc51d2f8dbdc8e8`.
- The two new files are intent-to-add only because `.gitignore` ignores `benchmark/*`. Their contents are unstaged. No robot code was changed.

## Checks and limits

`python3 -m py_compile benchmark/camera_heat/camera_heat.py` and `bash -n benchmark/camera_heat/run_camera_heat.sh` passed. An actual `--toy` run was attempted but stopped before the first block: in this agent's proot environment `/sys/class/thermal/thermal_zone9/temp` is absent and `battery/current_now` is permission denied even via `su`. Therefore the four requested 20-second device checks remain unverified. No temperature, power, PSS, or frame result is claimed.

For a native toy after closing the agent, use the same command below with `--toy` appended to `oneshot.sh`. Review the toy artifacts before starting the full run. The full run command from native Termux is:

```bash
( bash ~/ladder/run_camera_heat.sh --helper </dev/null > ~/ladder/camera_heat_helper.log 2>&1 &
  helper=$!; trap 'kill "$helper" 2>/dev/null || true' EXIT
  RUN_SCRIPT=run_camera_heat.sh bash ~/ladder/oneshot.sh )
```

Minimum full-run duration is 17 minutes: 5 minutes idle in oneshot plus four 3-minute blocks. Cooling gates, startup and PSS calls add time; budget about 20–30 minutes or longer if z9 does not return to idle + 4 °C. The toy takes at least 6 minutes 20 seconds: oneshot's 5-minute idle plus four 20-second blocks, excluding gates and startup.

RobotCam warm-up reads are included in failed reads. Android `dumpsys meminfo` adds load only to blocks 2–4, as required. In `old_path`, RobotCam is stopped, so its package PSS is expected to be null; camera-provider PSS is still requested. The power figure assumes negative `current_now` means discharge; the phone's sign has not been verified and raw µA/µV are retained in `sensors.jsonl`. Each `dumpsys` raw file remains in `/data/local/tmp`, consistent with the no-deletion instruction. Old-path temporary frames are the only files this code deletes.

## Reviewer invocation

Fresh Claude Code CLI session each round, model `claude-opus-5-5`, effort `medium`, Ponytail off. Read-only flags confirmed from `claude --help`: `--tools Read,Glob,Grep --disallowedTools Bash,Edit,Write,NotebookEdit,Task --permission-mode plan --permission-prompts none --strict-mcp-config --no-session-persistence`; headless `-p --output-format json`. No edit/write tools were available. Final reviewer response follows verbatim.

## Final independent review (verbatim)

# Independent review, round 4: `benchmark/camera_heat`, base af57939

I used read-only tools only. I made no edits and no git changes, ran nothing, and didn't touch the camera, the motors or any other reviewer. I didn't write a plan file, because you asked for no writes.

**Verdict: INCOMPLETE.** The fix resolves F1, F2 is gone, and the change adds no new blocker. It isn't a PASS yet because every hardware path is still unverified.

## What I read
- The repo's `camera_heat.py` (lines 110–219) and `run_camera_heat.sh`. They match the diff.
- The installed `~/ladder/oneshot.sh`.
- A search of `robotcam_test.py` for signal handling.
- The contracts cited in round 3. I didn't reread them, because nothing outside the helper's `finally` and the launch line changed.
- You reported the hashes, the executable bit and the passing `py_compile` and `bash -n` checks. I didn't rerun them.

## Round-3 findings

| Prior | Status | Evidence |
|---|---|---|
| **F1** second signal aborts cleanup | **Resolved** | `camera_heat.py:199-200` set SIGTERM and SIGHUP to `SIG_IGN` as the first statements of `finally`, before any blocking call. |
| F2 duplicate `--helper` | Resolved | `run_camera_heat.sh:11` now passes only `"$@"`. |
| F3 leftover `/data/local/tmp` files | Open (Low) | Unchanged. Disclose it in the report. |
| M2 / N1 hardware toy | **Open** | The 20 s toy was blocked in proot: `thermal_zone9` is absent and `current_now` is denied. |
| L1 sign of `current_now` | Open (Low) | Unchanged. The raw µA and µV values are in `sensors.jsonl`. |
| L3, N2, block-4 `null` app PSS | Accepted | Disclose them in the report. |

## Why the fix holds

**Closing the Termux session**
- SIGHUP reaches the whole foreground process group. The helper is backgrounded without job control, so it's in that group.
- The helper raises `KeyboardInterrupt`, enters `finally`, and ignores both signals within a few bytecodes.
- The subshell's EXIT trap `kill` arrives milliseconds later and is ignored.
- `$helper` is the Python process itself, because `run_camera_heat.sh` uses `exec`.

**Runner error during `old_path`**
- The trap's SIGTERM now arrives during the 17 s capture join and is ignored.
- `rmtree` at `:211` runs whenever the thread has finished.

**Children launched during cleanup**
- The final `am broadcast` inherits `SIG_IGN` for HUP and TERM, because `subprocess` only restores SIGPIPE and SIGXFSZ. That helps: a hangup can't kill the stop.
- A stuck `am` is still bounded, because the 20 s timeout uses SIGKILL.

**Cleanup is still bounded, since the helper can't be killed by plain signals while it runs**
- Sensor join: 10 s.
- `reader.wait()` at `:205` has no timeout. That's safe because `robotcam_test.py` installs no signal handlers, so the SIGTERM from `terminate()` (and the session SIGHUP) end it by default.
- Capture join: 17 s.
- `am`: 20 s.
- Worst case is about 47 s, after which the process exits normally.
- The old-path thread spawns no new `termux-camera-photo` after the deadline is set to now. The only exception is a sub-millisecond check-then-spawn race, and that capture is itself bounded by its 15 s timeout.

## New findings

**G1. Note: a residual race of a few bytecodes.** A second signal that lands after the first `KeyboardInterrupt` but before `:199` finishes would still raise inside `finally`. The window is microseconds, against a millisecond-scale gap before the bash trap. Closing it completely would take blocking the signals with `signal.pthread_sigmask` before the `try`. That isn't worth it here, so don't fix it.

**G2. Note: session close is still unverified on hardware.** This review assumes Termux delivers SIGHUP to the foreground process group when a session closes, rather than SIGKILL to the group. If it sends SIGKILL, no in-process cleanup can run and "always stop RobotCam" can't be guaranteed in that case. It doesn't block the normal path, where the runner stops RobotCam itself.

## Requirements
Unchanged from round 3 and still met statically:
- four blocks with the same rates, reader intervals and old-path loop
- 180 s blocks (`--toy` gives 20 s) and the z9 idle + 4 °C gate
- sensors every 5 s through native root, with no pipes inside `su`
- the mid-block meminfo in blocks 2–4 written with `</dev/null` to `/data/local/tmp`
- all report fields
- only the temporary frames deleted
- RobotCam stopped on every path

No installs, robot code, timed run, commit or push.

## Verdict: **INCOMPLETE**

There are no code blockers. To reach PASS, run the native toy with the documented subshell and `--toy` appended to `oneshot.sh`. All four blocks must finish and produce:
- non-null z9 and watts
- a RobotCam app PSS in blocks 2–3 and a provider PSS in blocks 2–4
- non-zero reader attempts
- a deleted `camera_heat_frames_*` folder
- RobotCam stopped at the end (`frame.json` absent)

The report must disclose:
- the warm-up failures (L3)
- the unverified current sign (L1)
- the extra load from the memory reading in blocks 2–4 (N2)
- the expected `null` app PSS in block 4
- the leftover `/data/local/tmp` files (F3)

The timed 180 s run, the commit and the push are your call.
