#!/data/data/com.termux/files/usr/bin/bash
# Marked adaptation of vis2/oneshot.sh (itself adapted from vis1): vis3 paths and runner; also refuses while a VIS-2 lock exists.
# VIS-3B: one test per launch (args go to both preflights), folder named after the test, no 5-minute idle.
set -u
H=/data/data/com.termux/files/home
R=$H/robot/benchmark/gemma_vision/vis3
PY=/data/data/com.termux/files/usr/bin/python3
if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo 'Run from native Termux, not Debian.'; exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
PREFLIGHT=0
PARGS=()
for ARG in "$@"; do
  if [ "$ARG" = --preflight ]; then PREFLIGHT=1; else PARGS+=("$ARG"); fi
done
# Folder label from the runner's validated configuration (refuses invalid arguments).
SUFFIX=$("$PY" -B "$R/vis3_run.py" --name "${PARGS[@]}") || exit 1
for L in "$H/robot/benchmark/gemma_vision/vis1/runs/.lock" "$H/robot/benchmark/gemma_vision/vis2/runs/.lock" "$R/runs/.lock"; do
  [ -e "$L" ] && { echo "Existing benchmark lock: $L; inspect before removing it."; exit 1; }
done
"$PY" -B "$R/vis3_run.py" --preflight "${PARGS[@]}" || exit 1
proot-distro login debian --bind "$H:/termux-home" -- /bin/bash -c \
  'cd /termux-home/robot && PYTHONDONTWRITEBYTECODE=1 python3 -B benchmark/gemma_vision/vis3/vis3_run.py --preflight "$@"' \
  vis3 "${PARGS[@]}" || exit 1
su -c id >/dev/null 2>&1 || { echo 'No Magisk root'; exit 1; }
if [ "$PREFLIGHT" = 1 ]; then echo 'VIS-3 READY (native + Debian preflight passed; nothing started).'; exit 0; fi
mkdir -p "$R/runs"
LOCK=$R/runs/.lock
mkdir "$LOCK" 2>/dev/null || { echo "Existing launch lock: $LOCK; inspect before removing it."; exit 1; }
NAME=$(date -u +%Y%m%dT%H%M%SZ)$SUFFIX
B=$R/runs/$NAME
mkdir "$B"
echo $$ > "$LOCK/pid"
touch "$B/owner_launch"
S=/data/local/tmp/vis3_screen_$$.txt
CONSOLE=$B/console.log
OLD_TIMEOUT=
RUNNER=
MONITOR=
CHILD=
cleanup() {
  touch "$B/stop"
  [ -z "$MONITOR" ] || wait "$MONITOR" 2>/dev/null
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
"$PY" -B "$R/vis3_run.py" --monitor "$B" >>"$B/monitor.log" 2>&1 &
MONITOR=$!
echo "VIS-3 $* — waiting for the monitor. Keep Termux foreground, screen on. Folder: $B" | tee -a "$CONSOLE"
for I in $(seq 1 40); do [ -f "$B/latest.json" ] && break; sleep 1; done
check_battery
[ -f "$B/latest.json" ] || { echo 'Monitor produced no samples'; exit 1; }
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
  'cd /termux-home/robot && PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python3 -B benchmark/gemma_vision/vis3/vis3_run.py "$@"' \
  vis3 --bridge "/termux-home/robot/benchmark/gemma_vision/vis3/runs/$NAME" "$@" >>"$CONSOLE" 2>&1 &
RUNNER=$!
while kill -0 "$RUNNER" 2>/dev/null; do wait "$RUNNER"; done
wait "$RUNNER"; RC=$?
touch "$B/stop"
wait "$CHILD"
echo "VIS-3 exited $RC. Folder: $B" | tee -a "$CONSOLE"
exit "$RC"
