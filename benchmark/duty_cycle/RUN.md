# DUTY1 — human runs only, native Termux

Motors off: no motor import, USB or serial. The runner refuses Debian/proot.
Before **each** run: quit every agent and exit Debian, stop robot-chat/chat.py
and any llama-server. In native Termux:

```bash
pkill -f security_reminder_hook
```

Unplug the charger. Keep the screen on, Termux in front, and the phone in its
mount. The launcher refuses resident agents, other benchmark runners and any
battery state other than Discharging, including a second check after idle.
It preserves POWERMAP1's #123 timeout read/set/restore, root zone logger, wake
lock and 5 min idle. It runs Python natively.

Run each command separately. Estimates include 5 min idle, 60 s **unrecorded**
after run-start Gemma load and one warm-up, and zero to 8 min per block/start
gate. Add server load/warm-up, camera checks, call drain and transition time.
The runner prints the block list and estimate at start. Smoke skips gate waits
and is labelled **SMOKE: not a measurement**.

| Run | Blocks | Estimated wall time, plus overhead |
| --- | --- | --- |
| Smoke PARTS | P0U 20 s, LOAD 60 s (2 loads 30 s apart), P0/CAM/YOLO/YOLO_NOSPIN/SEL 20 s each | 9 min |
| Smoke CYCLE | CONT/CYC50/CYC25 20 s each; HEATCOOL 30 s HEAT + 20 s COOL | 7.8 min |
| Full PARTS | P0U, LOAD, P0, CAM, YOLO, YOLO_NOSPIN, SEL: 120 s each | 20–84 min |
| Full CYCLE | CONT/CYC50/CYC25 240 s each; HEATCOOL 360 s HEAT + 180 s COOL | 27–67 min |

```bash
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set PARTS --smoke
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set CYCLE --smoke
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set PARTS
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set CYCLE
```

Active uses RobotCam rate 1, SizePolicy drive 320 with one 640 **instead of**
320 every 5 s, MID intra-op 2 pinned to CPUs 4–5 plus its dedicated caller,
inter-op 1, verified per-second worker observations. Selector letter scoring
uses POWERMAP1 cases every 20 s of accumulated **planned active** time, slot
zero included. Pause stops/checks/force-stops the camera, performs no detection
and no selector, and keeps Gemma resident in CYCLE. No robot motion is performed.

Camera startup belongs to active time; pending selector calls and a fresh-frame
survival check finish before the STOP request. STOP/check overhead belongs to
pause and consumes its planned off window. An overlong transition or call drain
is reported explicitly; actual spans can exceed planned spans. Smoke CYC blocks
use 10 s on / 5 s off, with a last partial cycle excluded from whole-cycle mean.
Full CYC50 uses 50/10; CYC25 uses 25/5. HEATCOOL is one atomic block, no COOL gate.

PARTS P0U/LOAD have Gemma unloaded; other parts have Gemma resident. CAM reads
and decodes frames with no detection; YOLO has no selector. YOLO_NOSPIN uses both
ORT spinning options at zero, rejecting rather than substituting if unsupported.
SEL has camera off. LOAD checks GGUF residency by mincore before each spawn;
zero resident pages = cold, any resident pages = warm (including partial cache).
No cache drop. One case call per load; connection/tokenization setup is separate.

Battery sampling is 0.5 s. Sysfs fuel-gauge averaging is unverified. Energy above
base and per-call energy are derived integrals, n/a for uncovered edges or gaps
>1.5 s. P0U is LOAD's baseline; P0 is the resident parts' baseline. Invalid
baselines cannot be used. CYCLE has no measured P0, so above-P0 energy is n/a;
its required phase and whole-cycle power are reported directly.

Resume with the **same set and smoke flag**; substitute the actual run folder:

```bash
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set PARTS --resume ~/duty_cycle/run_<UTC>_PARTS
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set CYCLE --resume ~/duty_cycle/run_<UTC>_CYCLE
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set PARTS --smoke --resume ~/duty_cycle/run_<UTC>_PARTS_smoke
bash ~/robot/benchmark/duty_cycle/oneshot.sh --set CYCLE --smoke --resume ~/duty_cycle/run_<UTC>_CYCLE_smoke
```

Finished block JSONs remain byte-for-byte; the interrupted block is redone.
An interruption during COOL redoes HEATCOOL entirely. Idle baseline is retained;
resume reloads/warms Gemma and waits 60 s. A completed INCOMPLETE block is retained
as evidence; move its `block_<name>.json` aside before resume to redo it.

Outputs: `~/duty_cycle/run_<UTC>_<set>[_smoke]/run.json`, atomic block JSONs,
server logs, `report.txt`; report copy in `~/storage/downloads/`. Console and
thermal logger live in `~/duty_cycle/`. zone9/10/11 are **CPU-core reading, not a
heat state**. Heat limits are Android CRITICAL, battery >=45 C, CPU fault >=110 C
in 3 consecutive samples. Stale sensors fail closed. Gate: VIRTUAL-SKIN <=
original idle +1.5 C, max 8 min, then marked warm start. No hardware run has been
performed by the Coder.

## If something is left running

Human only, from native Termux:

```bash
am broadcast -n com.pixelrobot.robotcam/.ControlReceiver -a com.pixelrobot.robotcam.STOP
su -c "am force-stop com.pixelrobot.robotcam </dev/null >/data/local/tmp/duty_stop.txt 2>&1"
su -c "ps -A | grep -E 'python|llama'"
# Inspect the listed PIDs, then: su -c "kill <PID>" (no blanket pkill).
su -c "settings get system screen_off_timeout </dev/null >/data/local/tmp/st.txt 2>&1"
su -c "cat /data/local/tmp/st.txt"
# If timeout restoration failed, use the saved OLD_TIMEOUT in the console log:
# su -c "settings put system screen_off_timeout <OLD_TIMEOUT> </dev/null >/data/local/tmp/st.txt 2>&1"
# Only after confirming no launcher/logger/runner remains:
rm -r ~/duty_cycle/.lock
```

Offline checks (mock root/Android/model calls only):

```bash
python -B ~/robot/benchmark/duty_cycle/test_duty_cycle.py
bash ~/robot/benchmark/duty_cycle/test_oneshot.sh
python -B ~/robot/benchmark/power_map/test_power_map.py
python -B ~/robot/benchmark/coresidency/test_coresidency.py
bash ~/robot/benchmark/coresidency/test_oneshot.sh
python -B ~/robot/benchmark/duty_cycle/confirm_view.py
```

`confirm_view.py` searches for the unique recorded `20261002T090821Z_CONFIRM`
folder in the repository archive and native-home `power_map` output root. It
prints the chosen path. The existing run is currently in the latter; expected
answers absent from its JSON print **not recorded**, never filled from today's
case file. An explicit `--run PATH` is also supported for read-only fixtures.
