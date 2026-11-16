#!/data/data/com.termux/files/usr/bin/bash
# Marked adaptation of coresidency/oneshot.sh (#123 file redirects); owner/native only.
set -u
H=/data/data/com.termux/files/home
R=$H/robot/benchmark/gemma_vision/vis1
PY=/data/data/com.termux/files/usr/bin/python3
if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo 'Run from native Termux, not Debian.'; exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
SMOKE=0
for ARG in "$@"; do [ "$ARG" = --smoke ] && SMOKE=1; done
"$PY" -B "$R/vis1_run.py" --preflight || exit 1
# Debian has its own python3/site-packages (VIS1-FIX: requests/PIL missing there stalled three
# owner starts after the 5-minute idle); run the same --preflight in that interpreter first.
proot-distro login debian --bind "$H:/termux-home" -- /bin/bash -c \
  'cd /termux-home/robot && PYTHONDONTWRITEBYTECODE=1 python3 -B benchmark/gemma_vision/vis1/vis1_run.py --preflight' \
  || exit 1
su -c id >/dev/null 2>&1 || { echo 'No Magisk root'; exit 1; }
# All repo outputs stay inside vis1. Lock cannot overlap another owner launch.
mkdir -p "$R/runs"
LOCK=$R/runs/.lock
mkdir "$LOCK" 2>/dev/null || { echo "Existing launch lock: $LOCK; inspect before removing it."; exit 1; }
B=$R/runs/bridge_$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$B"
echo $$ > "$LOCK/pid"
touch "$B/owner_launch"
S=/data/local/tmp/vis1_screen_$$.txt
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
  [[ ${BATTERY[0]:-} =~ ^[0-9]+$ ]] && [ "${BATTERY[0]}" -ge 50 ] && [ "${BATTERY[0]}" -le 80 ] &&
    [ "${BATTERY[1]:-}" = Discharging ] || { echo 'Need 50-80% battery, charger unplugged.'; exit 1; }
}
check_battery
termux-wake-lock
su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
OLD_TIMEOUT=$(su -c "cat $S" | tr -d '[:space:]')
[[ $OLD_TIMEOUT =~ ^[0-9]+$ ]] || { echo 'Screen timeout unreadable'; exit 1; }
su -c "settings put system screen_off_timeout 2147483647 </dev/null >$S 2>&1"
su -c "settings get system screen_off_timeout </dev/null >$S 2>&1"
[ "$(su -c "cat $S" | tr -d '[:space:]')" = 2147483647 ] || exit 1
"$PY" -B "$R/vis1_run.py" --monitor "$B" >>"$B/monitor.log" 2>&1 &
MONITOR=$!
# CPU thermals (zone9/10/11), skin, status, battery and caps are in samples.jsonl from root.
if [ "$SMOKE" = 1 ]; then
  echo "VIS-1 --smoke: skipping the 5-minute idle, waiting for the monitor instead. Bridge: $B" | tee -a "$CONSOLE"
  for I in $(seq 1 40); do [ -f "$B/latest.json" ] && break; sleep 1; done
else
  echo "VIS-1 idling 5 minutes. Keep Termux foreground, screen on. Bridge: $B" | tee -a "$CONSOLE"
  pause 300
fi
check_battery
"$PY" -B "$R/vis1_run.py" --preflight || exit 1
[ -f "$B/latest.json" ] || { echo 'Monitor produced no samples'; exit 1; }
cp "$B/latest.json" "$B/idle.json"
# Native cache-drop handshake, same protocol as archived oneshot.sh.
( while [ ! -f "$B/stop" ]; do
    if [ -f "$B/drop_request" ] && [ ! -f "$B/drop_done" ]; then
      if su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; then value=ok; else value=failed; fi
      echo "$value" > "$B/drop_done.tmp"
      mv "$B/drop_done.tmp" "$B/drop_done"
    fi
    sleep .2
  done ) &
CHILD=$!
# Pass the same bind paths the repository documents; Debian is only the HTTP/server runner.
proot-distro login debian --bind "$H:/termux-home" -- /bin/bash -c \
  'cd /termux-home/robot && PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python3 -B benchmark/gemma_vision/vis1/vis1_run.py "$@"' \
  vis1 --bridge "/termux-home/robot/benchmark/gemma_vision/vis1/runs/$(basename "$B")" "$@" >>"$CONSOLE" 2>&1 &
RUNNER=$!
while kill -0 "$RUNNER" 2>/dev/null; do wait "$RUNNER"; done
wait "$RUNNER"; RC=$?
touch "$B/stop"
wait "$CHILD"
echo "VIS-1 exited $RC. Send this console log: $CONSOLE" | tee -a "$CONSOLE"
exit "$RC"
