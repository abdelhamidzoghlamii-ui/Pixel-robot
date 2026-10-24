#!/data/data/com.termux/files/usr/bin/bash
# VIS-3D: b11716 text (A, B, C) and image (D) tests, ONE test per launch; no motors, no camera. Start from NATIVE
# Termux (prompt "~ $"), no agent running, charger unplugged, battery >= 80 %, Termux in front:
#   bash ~/robot/benchmark/gemma_vision/vis3d/oneshot.sh --test A    (or B, C, D)
# chat_mtp_repro/oneshot.sh (VIS-3C) with this folder's runner: same 5 s root thermal logger, checks and screen handling;
# no 5 min idle (as VIS-3C). The runner runs in native Termux, not in Debian: the robot
# runs ONNX Runtime and llama-server natively. Root commands that call Android services use the DECISIONS #123
# form: su -c "<cmd> </dev/null >/data/local/tmp/<file> 2>&1", then the file is read.
H=/data/data/com.termux/files/home
L=$H/robot_runs/vis3d
R=$H/robot/benchmark/gemma_vision/vis3d
S=/data/local/tmp/vis3d_screen.txt
mkdir -p "$L"
CONSOLE=$L/oneshot_console_$(date -u +%Y%m%dT%H%M%SZ).log
say() { echo "[oneshot] $*" | tee -a "$CONSOLE"; }

if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo "Run this from native Termux (~ \$), not from Debian."; exit 1
fi
su -c id >/dev/null 2>&1 || { echo "No root: su failed. Check Magisk."; exit 1; }
[ -f "$R/vis3d.py" ] || { echo "Missing $R/vis3d.py"; exit 1; }
case "$*" in "--test A"|"--test B"|"--test C"|"--test D") ;; *) echo "Usage: oneshot.sh --test A|B|C|D"; exit 1;; esac
if pgrep -fa 'claude|agy|node|codex' >/dev/null; then
  echo "An agent is still running:"; pgrep -fa 'claude|agy|node|codex'; echo "Quit it, then rerun."; exit 1
fi
if pgrep -fa 'llama-server|chat\.py' >/dev/null; then
  echo "llama-server or chat.py is already running:"; pgrep -fa 'llama-server|chat\.py'; echo "Stop it, then rerun."; exit 1
fi
if pgrep -fa 'chat_mtp\.py|repro\.py|vis3d\.py|power_map\.py|coresidency\.py|thermal_char\.py|vis[0-9]_run\.py' >/dev/null; then
  echo "A benchmark runner is already running:"; pgrep -fa 'chat_mtp\.py|repro\.py|vis3d\.py|power_map\.py|coresidency\.py|thermal_char\.py|vis[0-9]_run\.py'; echo "Stop it, then rerun."; exit 1
fi
check_battery() {  # checked at start and again right before the runner
  BAT=$(su -c 'cat /sys/class/power_supply/battery/status' 2>/dev/null)
  if [ "$BAT" != Discharging ]; then
    echo "[oneshot] Battery status is '$BAT', not Discharging: unplug the charger (power numbers would be invalid)." | tee -a "$CONSOLE"
    exit 1
  fi
}
cleanup() {
  [ -n "$LOCKED" ] || return 0  # refused before taking the lock: another run's files are not ours to touch
  touch "$L/.stop_thermal"
  if [ "$SCREEN_SET" = 1 ] && [[ $OLD_TIMEOUT =~ ^[0-9]+$ ]]; then
    su -c "settings put system screen_off_timeout $OLD_TIMEOUT </dev/null >$S 2>&1"
    su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
    BACK=$(su -c "cat $S" 2>/dev/null | tr -d '[:space:]')
    if [ "$BACK" = "$OLD_TIMEOUT" ]; then
      say "screen timeout restored to $OLD_TIMEOUT ms; read back: $BACK"
    else
      say "WARNING: screen timeout NOT restored (read back '$BACK', expected $OLD_TIMEOUT). Run in native Termux:"
      say "  su -c \"settings put system screen_off_timeout $OLD_TIMEOUT </dev/null >$S 2>&1\""
      RESTORE_FAILED=1
    fi
  fi
  [ -z "${PAUSE:-}" ] || kill "$PAUSE" 2>/dev/null
  termux-wake-unlock
  # the lock goes only once the logger has exited (it polls .stop_thermal every 5 s):
  # otherwise a new run could remove .stop_thermal under it and share its files with it
  JOBS="${LOGGER:-}"
  for i in $(seq 40); do
    ALIVE=; for p in $JOBS; do kill -0 "$p" 2>/dev/null && ALIVE=1; done
    [ -z "$ALIVE" ] && break
    sleep 0.5
  done
  if [ -n "$ALIVE" ]; then
    say "WARNING: logger still running after 20 s; $L/.lock kept. Remove it when they have stopped."
    RESTORE_FAILED=1
  else
    say "stopped logger"
    rm -rf "$L/.lock"
  fi
  [ -z "${RESTORE_FAILED:-}" ] || exit 1
}
LOCKED=
SCREEN_SET=0
trap cleanup EXIT
RUNNER=
LAUNCHING=
PENDING=
on_signal() {  # before the runner starts: exit (the EXIT trap cleans up); after: forward it and keep waiting
  if [ -n "$RUNNER" ]; then kill -TERM "$RUNNER" 2>/dev/null
  elif [ -n "$LAUNCHING" ]; then PENDING=1  # between `&` and RUNNER=$!: forwarded right after
  else exit "$1"; fi
}
trap 'on_signal 143' TERM; trap 'on_signal 129' HUP; trap 'on_signal 130' INT
# one run at a time: runs share thermal.log, .stop_thermal and port 8080
if ! mkdir "$L/.lock" 2>/dev/null; then
  echo "Another VIS-3D launcher holds $L/.lock (pid $(cat "$L/.lock/pid" 2>/dev/null))."
  echo "If no run is active (for example after a crash), remove it: rm -r $L/.lock"; exit 1
fi
LOCKED=1
echo $$ > "$L/.lock/pid"
check_battery

termux-wake-lock
pause() { sleep "$1" & PAUSE=$!; wait "$PAUSE"; PAUSE=; }  # a trapped signal interrupts wait, not a foreground sleep
# keep the screen on (Android moves Termux off the big cores when it is not in front)
OLD_TIMEOUT=
if su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"; then
  OLD_TIMEOUT=$(su -c "cat $S" 2>/dev/null | tr -d '[:space:]')
fi
if ! [[ $OLD_TIMEOUT =~ ^[0-9]+$ ]]; then
  say "Could not read the screen timeout (read '$OLD_TIMEOUT': $(su -c "cat $S" 2>/dev/null | head -c 200)); not started."; exit 1
fi
SCREEN_SET=1  # from here on, cleanup restores the validated old value
su -c "settings put system screen_off_timeout 2147483647 </dev/null >$S 2>&1"
su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
NOW=$(su -c "cat $S" 2>/dev/null | tr -d '[:space:]')
if [ "$NOW" != 2147483647 ]; then
  say "Could not set the screen timeout (read back '$NOW'); not started."; exit 1
fi
say "screen timeout was $OLD_TIMEOUT ms; set to 2147483647 for the run (read back)"
[ "$OLD_TIMEOUT" = 2147483647 ] && say "WARNING: the saved timeout is already 2147483647 (left by an earlier run?); set it yourself afterwards"
rm -f "$L/.stop_thermal"

su -c "while [ ! -f $L/.stop_thermal ]; do echo \"\$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=\$(cat /sys/class/thermal/thermal_zone9/temp) z10=\$(cat /sys/class/thermal/thermal_zone10/temp) z11=\$(cat /sys/class/thermal/thermal_zone11/temp) CPU-core_reading,_not_a_heat_state=1\" >> $L/thermal.log; sleep 5; done" &
LOGGER=$!

pause 6
say "CPU-core reading, not a heat state: $(tail -1 "$L/thermal.log")"
check_battery

# The runner runs as a job so TERM/HUP/INT sent to this launcher reach it: it stops llama-server
# itself, and we wait for that before the EXIT trap restores the screen timeout.
LAUNCHING=1
env PYTHONUNBUFFERED=1 python -u "$R/vis3d.py" "$@" > >(tee -a "$CONSOLE") 2>&1 &
RUNNER=$!
[ -z "$PENDING" ] || kill -TERM "$RUNNER" 2>/dev/null
while kill -0 "$RUNNER" 2>/dev/null; do wait "$RUNNER"; done
wait "$RUNNER"; RC=$?
if [ "$RC" = 0 ]; then say "done; console log: $CONSOLE"; else say "RUNNER FAILED (exit $RC); console log: $CONSOLE"; fi
exit "$RC"
