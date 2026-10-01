# Co-residency benchmark — smoke run (human, native Termux)

Motors off: nothing here imports `motors.py` or opens USB/serial. The runner and `oneshot.sh` refuse to run
inside Debian/proot. `su` works only in native Termux.

## Before

1. Quit every agent (Claude, Codex, AGY) and exit Debian. Stop `robot-chat`/`chat.py` and any llama-server.
2. Unplug the charger (`oneshot.sh` refuses unless the battery status is `Discharging`).
3. Keep Termux in front and the screen on for the whole run.

## Smoke run (about 5 min idle + about 3 min)

From native Termux (prompt `~ $`):

```bash
bash ~/robot/benchmark/coresidency/oneshot.sh --smoke
```

This runs 20 s blocks with no cooldown gate. It still takes one real cache drop (handshake), one cold and one warm Gemma load, and the MTP
snapshot. Output: `~/coresidency/run_<UTC>_smoke/` (`block_*.json`, `loads.json`, `run.json`, `report.txt`,
llama-server logs), plus `~/storage/downloads/coresidency_run_<UTC>_smoke_report.txt`. The console log is
`~/coresidency/oneshot_console_<UTC>.log`.

Check in the report: every block has frames processed and no failed reads. B3/B4 have selector calls with no
errors. Gemma cold/warm load values are present, and the cold mode says `full-cold`. PSS columns are not n/a. LMK lines,
zone9 and mean W are filled. The console ends with `screen timeout restored to <old> ms; read back: <old>`.

## Full run (after a good smoke run)

```bash
bash ~/robot/benchmark/coresidency/oneshot.sh
```

This runs four 180 s blocks, each gated on VIRTUAL-SKIN ≤ idle skin + 1.5 °C and zone9 ≤ idle + 4 °C (idle = the
reading at runner start, after the 5 min idle; if the gate is not reached in 15 min the block starts anyway and the
report shows `warm start`). The Gemma loads are gated the same way; the MTP snapshot (B5, RAM only) is not gated.

Continue an interrupted run (a completed block is kept; the interrupted block is redone):

```bash
bash ~/robot/benchmark/coresidency/oneshot.sh --resume ~/coresidency/run_<UTC>
```

Add `--smoke` as well when resuming a smoke run.

If the report starts with `INCOMPLETE:` (the RobotCam end check failed after a block or B5: capture not shown stopped
within 15 s, or the force-stop failed; a block not run because a limit was already met at its start; a block stopped
fail-closed: no VIRTUAL-SKIN + status reading for 60 s, or no CPU-zone or battery reading for 5 s; a Gemma block
without a successful selector call, or with a due 20 s slot missed, started > 5 s late or finished after the block; a
1 s sample without every `scaling_max_freq`; a 5 s sample with a root/PSS/thermal error in any block or in B5, or no
usable sample; any failed RobotCam read, no frames, or no new frame or no RobotCam process at a block's end; no
detections or no drift for a required detector size; llama-server not surviving a Gemma block; a battery status other
than `Discharging` during a block; a failed LMK logcat query), the runner exits
non-zero and the block JSON is kept as evidence. To redo that block, move its `block_<name>.json` out of the run
folder, then resume.

## If something is left running

```bash
am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP
su -c "ps -A | grep -E 'python|llama'"     # then: su -c "kill <PID>"  (never pkill -f, COMMANDS §8)
su -c "settings get system screen_off_timeout </dev/null >/data/local/tmp/st.txt 2>&1"; su -c "cat /data/local/tmp/st.txt"
rm -r ~/coresidency/.lock    # only if no launcher is running (a killed launcher leaves it; the next start refuses)
```
