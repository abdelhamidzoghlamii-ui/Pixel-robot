#!/data/data/com.termux/files/usr/bin/bash
# Marked adaptation of vis1/oneshot.sh: vis2 paths, one run folder (bridge = output), --preflight exits early.
set -u
H=/data/data/com.termux/files/home
R=$H/robot/benchmark/gemma_vision/vis2
PY=/data/data/com.termux/files/usr/bin/python3
if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo 'Run from native Termux, not Debian.'; exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
SMOKE=0
PREFLIGHT=0
for ARG in "$@"; do
  [ "$ARG" = --smoke ] && SMOKE=1
  [ "$ARG" = --preflight ] && PREFLIGHT=1
done
for L in "$H/robot/benchmark/gemma_vision/vis1/runs/.lock" "$R/runs/.lock"; do
  [ -e "$L" ] && { echo "Existing benchmark lock: $L; inspect before removing it."; exit 1; }
done
"$PY" -B "$R/vis2_run.py" --preflight || exit 1
proot-distro login debian --bind "$H:/termux-home" -- /bin/bash -c \
  'cd /termux-home/robot && PYTHONDONTWRITEBYTECODE=1 python3 -B benchmark/gemma_vision/vis2/vis2_run.py --preflight' \
  || exit 1
su -c id >/dev/null 2>&1 || { echo 'No Magisk root'; exit 1; }
if [ "$PREFLIGHT" = 1 ]; then echo 'VIS-2 READY (native + Debian preflight passed; nothing started).'; exit 0; fi
mkdir -p "$R/runs"
LOCK=$R/runs/.lock
mkdir "$LOCK" 2>/dev/null || { echo "Existing launch lock: $LOCK; inspect before removing it."; exit 1; }
NAME=$(date -u +%Y%m%dT%H%M%SZ)
[ "$SMOKE" = 1 ] && NAME=smoke_$NAME
B=$R/runs/$NAME
mkdir "$B"
echo $$ > "$LOCK/pid"
touch "$B/owner_launch"
S=/data/local/tmp/vis2_screen_$$.txt
CONSOLE=$B/console.log
OLD_TIMEOUT=
RUNNER=
MONITOR=
PAUSE=
CHILD=
cleanup() {
  touch "$B/stop"
  [ -z "$MONITOR" ] || wait "$MONITOR" 2>/dev/null
  [ -z "$PAUSE" ] || { kill -TERM "$PAUSE" 2>/dev/null; wait "$PAUSE" 2>/dev/null; }
  if [[ $OLD_TIMEOUT =~ ^[0-9]+$ ]]; then
    su -c "settings put system screen_off_timeout $OLD_TIMEOUT </dev/null >$S 2>&1"
    su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
    BACK=$(su -c "cat $S" | tr -d '[:space:]')
    [ "$BACK" = "$OLD_TIMEOUT" ] || echo "WARNING: restore screen timeout manually to $OLD_TIMEOUT"
  fi
  termux-wake-unlock
  rmdir "$LOCK" 2>/dev/null || { rm -f "$LOCK/pid"; rmdir "$LOCK"; }
}
trap cleanup EXIT
on_signal() {
  if [ -n "$RUNNER" ]; then
    # proot may not forward signals: the runner also watches bridge/abort every telemetry check.
    touch "$B/abort"
  else
    exit 130
  fi
}
trap on_signal INT TERM HUP
pause() { sleep "$1" & PAUSE=$!; wait "$PAUSE"; PAUSE=; }
check_battery() {
  readarray -t BATTERY < <(su -c 'cat /sys/class/power_supply/battery/capacity /sys/class/power_supply/battery/status')
  [[ ${BATTERY[0]:-} =~ ^[0-9]+$ ]] && [ "${BATTERY[0]}" -ge 50 ] &&
    [ "${BATTERY[1]:-}" = Discharging ] || { echo 'Need >=50% battery, charger unplugged.'; exit 1; }
}
check_battery
termux-wake-lock
su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
OLD_TIMEOUT=$(su -c "cat $S" | tr -d '[:space:]')
[[ $OLD_TIMEOUT =~ ^[0-9]+$ ]] || { echo 'Screen timeout unreadable'; exit 1; }
su -c "settings put system screen_off_timeout 2147483647 </dev/null >$S 2>&1"
su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
[ "$(su -c "cat $S" | tr -d '[:space:]')" = 2147483647 ] || exit 1
"$PY" -B "$R/vis2_run.py" --monitor "$B" >>"$B/monitor.log" 2>&1 &
MONITOR=$!
if [ "$SMOKE" = 1 ]; then
  echo "VIS-2 --smoke: no idle, waiting for the monitor. Folder: $B" | tee -a "$CONSOLE"
  for I in $(seq 1 40); do [ -f "$B/latest.json" ] && break; sleep 1; done
else
  echo "VIS-2 idling 5 minutes. Keep Termux foreground, screen on. Folder: $B" | tee -a "$CONSOLE"
  pause 300
fi
check_battery
"$PY" -B "$R/vis2_run.py" --preflight || exit 1
[ -f "$B/latest.json" ] || { echo 'Monitor produced no samples'; exit 1; }
cp "$B/latest.json" "$B/idle.json"
( while [ ! -f "$B/stop" ]; do
    if [ -f "$B/drop_request" ] && [ ! -f "$B/drop_done" ]; then
      if su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; then value=ok; else value=failed; fi
      echo "$value" > "$B/drop_done.tmp"
      mv "$B/drop_done.tmp" "$B/drop_done"
    fi
    sleep .2
  done ) &
CHILD=$!
proot-distro login debian --bind "$H:/termux-home" -- /bin/bash -c \
  'cd /termux-home/robot && PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python3 -B benchmark/gemma_vision/vis2/vis2_run.py "$@"' \
  vis2 --bridge "/termux-home/robot/benchmark/gemma_vision/vis2/runs/$NAME" "$@" >>"$CONSOLE" 2>&1 &
RUNNER=$!
while kill -0 "$RUNNER" 2>/dev/null; do wait "$RUNNER"; done
wait "$RUNNER"; RC=$?
touch "$B/stop"
wait "$CHILD"
echo "VIS-2 exited $RC. Folder: $B" | tee -a "$CONSOLE"
exit "$RC"
