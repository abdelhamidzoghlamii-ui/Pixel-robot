#!/data/data/com.termux/files/usr/bin/bash
# One-shot s1o speed run, v2. Start from NATIVE Termux (prompt "~ $"):
#   bash ~/ladder/oneshot.sh                  (fresh run)
#   bash ~/ladder/oneshot.sh --resume <dir>   (continue a run)
# Thermal logger, checks, screen kept on, 5 min idle, the run inside Debian,
# and root cache drops on request (file handshake with the runner).
H=/data/data/com.termux/files/home
L=$H/ladder
CONSOLE=$L/oneshot_console_$(date -u +%Y%m%dT%H%M%SZ).log
say() { echo "[oneshot] $*" | tee -a "$CONSOLE"; }

if [ -d /termux-home ]; then echo "Run this from native Termux (~ \$), not from Debian."; exit 1; fi
su -c id >/dev/null 2>&1 || { echo "No root: su failed. Check Magisk."; exit 1; }
[ -f "$L/run_s1o_speed.sh" ] || { echo "Missing $L/run_s1o_speed.sh"; exit 1; }
if pgrep -fa 'claude|agy|node' >/dev/null; then
  echo "An agent is still running:"; pgrep -fa 'claude|agy|node'; echo "Quit it, then rerun."; exit 1
fi

termux-wake-lock
# keep the screen on (Android moves Termux off the big cores when it is not in front)
OLD_TIMEOUT=$(su -c 'settings get system screen_off_timeout')
su -c 'settings put system screen_off_timeout 2147483647'
rm -f "$L/.stop_thermal" "$L/.drop_request" "$L/.drop_done" "$L/.drop_done.tmp"

su -c "while [ ! -f $L/.stop_thermal ]; do echo \"\$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=\$(cat /sys/class/thermal/thermal_zone9/temp) z10=\$(cat /sys/class/thermal/thermal_zone10/temp) z11=\$(cat /sys/class/thermal/thermal_zone11/temp)\" >> $L/thermal.log; sleep 5; done" &

# cache-drop watcher: runner creates .drop_request, we drop and write .drop_done
( while [ ! -f "$L/.stop_thermal" ]; do
    if [ -f "$L/.drop_request" ] && [ ! -f "$L/.drop_done" ]; then
      if su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; then r=ok; else r=failed; fi
      echo $r > "$L/.drop_done.tmp" && mv "$L/.drop_done.tmp" "$L/.drop_done"; say "cache drop: $r"
    fi
    sleep 1
  done ) &

cleanup() {
  touch "$L/.stop_thermal"
  su -c "settings put system screen_off_timeout ${OLD_TIMEOUT:-60000}"
  termux-wake-unlock
  say "stopped logger and watcher; screen timeout restored to ${OLD_TIMEOUT}"
}
trap cleanup EXIT

sleep 6
say "thermal: $(tail -1 "$L/thermal.log")"
say "idling 5 minutes; keep Termux in front"
sleep 300
say "thermal after idle: $(tail -1 "$L/thermal.log")"

proot-distro login debian --bind "$H:/termux-home" -- \
  env PYTHONUNBUFFERED=1 LADDER_COLD_HANDSHAKE=1 /termux-home/ladder/run_s1o_speed.sh "$@" 2>&1 | tee -a "$CONSOLE"
say "done; console log: $CONSOLE"
