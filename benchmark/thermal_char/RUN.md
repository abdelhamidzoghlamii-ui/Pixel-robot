# Thermal characterization run (human, native Termux)

Motors off: nothing here imports `motors.py` or opens USB/serial. The runner and `oneshot.sh` refuse to run
inside Debian/proot; `su` works only in native Termux. Measurements only: the report recommends no thresholds.

Load: RobotCam mode B rate 2, 640 detection on every new frame, llama-server (server_manager `setup_q4` flags)
generating back-to-back (fixed prompt, n_predict 256, temperature 0). The load stops at the first of:
VIRTUAL-SKIN >= 48.0 degC, battery >= 45.0 degC, Android thermal status >= EMERGENCY (5), a BIG/MID/LITTLE zone
>= 110 degC in 3 consecutive 1 s samples, or 20 min of load. It also stops (fail closed, run marked INCOMPLETE)
when there is no VIRTUAL-SKIN + status reading for 60 s from load start, no CPU zone or battery temperature
reading for 5 s, no new RobotCam frame for 10 s, a llama-server or detector failure, or cores 4-7 are lost. The
same stops apply while llama-server and RobotCam are starting (the report then says `during startup:`). After
the load, 5 min of cooldown logging.

## Before

1. Quit every agent (Claude, Codex, AGY) and exit Debian. Stop `robot-chat`/`chat.py` and any llama-server.
   No co-residency run may be active (the launcher refuses while `~/coresidency/.lock` exists).
2. Unplug the charger (`oneshot.sh` refuses unless the battery status is `Discharging`).
3. Keep Termux in front and the screen on for the whole run. Note the room temperature and how the phone is
   mounted (case, on the robot, on a table); it goes into `--note`.

## Smoke run (about 5 min idle + about 2 min)

From native Termux (prompt `~ $`):

```bash
bash ~/robot/benchmark/thermal_char/oneshot.sh --smoke --note "room 23 degC, phone on the robot, no case"
```

60 s of load, 30 s of cooldown, same stops. Output: `~/thermal_char/run_<UTC>_smoke/` (`samples_1s.jsonl`,
`thermalservice_5s.jsonl`, `thermalservice_raw.jsonl`, `frames.jsonl`, `gen.jsonl`, `llama-server.log`,
`run.json`, `report.txt`), plus `~/storage/downloads/thermal_char_run_<UTC>_smoke_report.txt`. The console
log is `~/thermal_char/oneshot_console_<UTC>.log`.

Check in the report: no `INCOMPLETE:` line; `VIRTUAL-SKIN in HAL temperatures: yes` (the list of HAL names is
printed either way); dumps with skin and status are close to the number of dumps; incomplete 1 s samples 0 and
largest gap about 1 s; the work-rate windows show frames and tok/s; `run.json` `after_stop` shows no RobotCam
pids and `llama_server_running: false`. The console ends with `screen timeout restored to <old> ms; read back:
<old>`.

## Full run (after a good smoke run; about 5 min idle + up to 20 min load + 5 min cooldown)

```bash
bash ~/robot/benchmark/thermal_char/oneshot.sh --note "room 23 degC, phone on the robot, no case"
```

There is no resume: an interrupted run keeps its logs and a report whose stop reason starts with `aborted:`.
Start a new run after the phone has cooled.

## Offline checks (Debian, no phone hardware)

```bash
cd /termux-home/robot/benchmark/thermal_char
/data/data/com.termux/files/usr/bin/python test_thermal_char.py   # fakes: sysfs, su, dumpsys, frames, server
bash test_oneshot.sh                                              # launcher against fake su/settings/pgrep
```

## If something is left running

```bash
am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP
su -c "ps -A | grep -E 'python|llama'"     # then: su -c "kill <PID>"  (never pkill -f, COMMANDS §8)
su -c "settings get system screen_off_timeout </dev/null >/data/local/tmp/st.txt 2>&1"; su -c "cat /data/local/tmp/st.txt"
rm -r ~/thermal_char/.lock    # only if no launcher is running (a killed launcher leaves it; the next start refuses)
```
