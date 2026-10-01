#!/data/data/com.termux/files/usr/bin/bash
# Thermal characterization run, motors off. Copy of ../coresidency/oneshot.sh with only the paths, the runner name
# and the refusals of a concurrent co-residency/thermal run changed. Start from NATIVE Termux (prompt "~ $"), no agent
# running, charger unplugged, Termux in front:
#   bash ~/robot/benchmark/thermal_char/oneshot.sh --note "<room degC, mounting>"           (full run)
#   bash ~/robot/benchmark/thermal_char/oneshot.sh --smoke --note "<room degC, mounting>"   (60 s load)
# From ~/ladder/oneshot.sh (db1f3156...): thermal logger, checks, screen kept on, 5 min idle, root cache drops on
# request (file handshake with the runner). Unlike it, the runner runs in native Termux, not in Debian: the robot
# runs ONNX Runtime and llama-server natively. Root commands that call Android services use the DECISIONS #123
# form: su -c "<cmd> </dev/null >/data/local/tmp/<file> 2>&1", then the file is read.
H=/data/data/com.termux/files/home
L=$H/thermal_char
R=$H/robot/benchmark/thermal_char
S=/data/local/tmp/thermal_char_screen.txt
mkdir -p "$L"
CONSOLE=$L/oneshot_console_$(date -u +%Y%m%dT%H%M%SZ).log
say() { echo "[oneshot] $*" | tee -a "$CONSOLE"; }

if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo "Run this from native Termux (~ \$), not from Debian."; exit 1
fi
su -c id >/dev/null 2>&1 || { echo "No root: su failed. Check Magisk."; exit 1; }
[ -f "$R/run_thermal_char.sh" ] || { echo "Missing $R/run_thermal_char.sh"; exit 1; }
if pgrep -fa 'claude|agy|node|codex' >/dev/null; then
  echo "An agent is still running:"; pgrep -fa 'claude|agy|node|codex'; echo "Quit it, then rerun."; exit 1
fi
if pgrep -fa 'llama-server|chat\.py' >/dev/null; then
  echo "llama-server or chat.py is already running:"; pgrep -fa 'llama-server|chat\.py'; echo "Stop it, then rerun."; exit 1
fi
if pgrep -fa 'coresidency\.py|thermal_char\.py' >/dev/null; then
  echo "A co-residency or thermal runner is already running:"; pgrep -fa 'coresidency\.py|thermal_char\.py'; echo "Stop it, then rerun."; exit 1
fi
if [ -d "$H/coresidency/.lock" ]; then  # the co-residency launcher uses the same RobotCam and port 8080
  echo "The co-residency launcher holds $H/coresidency/.lock; wait for it or remove a stale lock."; exit 1
fi
check_battery() {  # checked at start and again after the idle, right before the runner
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
  # the lock goes only once the logger and watcher have exited (they poll .stop_thermal every 1-5 s):
  # otherwise a new run could remove .stop_thermal under them and share its files with them
  JOBS="${LOGGER:-} ${WATCHER:-}"
  for i in $(seq 40); do
    ALIVE=; for p in $JOBS; do kill -0 "$p" 2>/dev/null && ALIVE=1; done
    [ -z "$ALIVE" ] && break
    sleep 0.5
  done
  if [ -n "$ALIVE" ]; then
    say "WARNING: logger/watcher still running after 20 s; $L/.lock kept. Remove it when they have stopped."
    RESTORE_FAILED=1
  else
    say "stopped logger and watcher"
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
# one run at a time: runs share thermal.log, .stop_thermal, the handshake files, RobotCam and port 8080
if ! mkdir "$L/.lock" 2>/dev/null; then
  echo "Another thermal launcher holds $L/.lock (pid $(cat "$L/.lock/pid" 2>/dev/null))."
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
rm -f "$L/.stop_thermal" "$L/.drop_request" "$L/.drop_done" "$L/.drop_done.tmp"

su -c "while [ ! -f $L/.stop_thermal ]; do echo \"\$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=\$(cat /sys/class/thermal/thermal_zone9/temp) z10=\$(cat /sys/class/thermal/thermal_zone10/temp) z11=\$(cat /sys/class/thermal/thermal_zone11/temp)\" >> $L/thermal.log; sleep 5; done" &
LOGGER=$!

# cache-drop watcher: runner creates .drop_request, we drop and write .drop_done
( while [ ! -f "$L/.stop_thermal" ]; do
    if [ -f "$L/.drop_request" ] && [ ! -f "$L/.drop_done" ]; then
      if su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; then r=ok; else r=failed; fi
      echo $r > "$L/.drop_done.tmp" && mv "$L/.drop_done.tmp" "$L/.drop_done"; say "cache drop: $r"
    fi
    sleep 1
  done ) &
WATCHER=$!


pause 6
say "thermal: $(tail -1 "$L/thermal.log")"
say "idling 5 minutes; keep Termux in front"
pause 300
say "thermal after idle: $(tail -1 "$L/thermal.log")"
check_battery

# The runner runs as a job so TERM/HUP/INT sent to this launcher reach it: it stops RobotCam and llama-server
# itself, and we wait for that before the EXIT trap restores the screen timeout.
LAUNCHING=1
env PYTHONUNBUFFERED=1 bash "$R/run_thermal_char.sh" "$@" > >(tee -a "$CONSOLE") 2>&1 &
RUNNER=$!
[ -z "$PENDING" ] || kill -TERM "$RUNNER" 2>/dev/null
while kill -0 "$RUNNER" 2>/dev/null; do wait "$RUNNER"; done
wait "$RUNNER"; RC=$?
if [ "$RC" = 0 ]; then say "done; console log: $CONSOLE"; else say "RUNNER FAILED (exit $RC); console log: $CONSOLE"; fi
exit "$RC"
