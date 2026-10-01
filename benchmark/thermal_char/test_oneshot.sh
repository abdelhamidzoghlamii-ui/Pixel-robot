#!/bin/bash
# Offline test of oneshot.sh (copy of ../coresidency/test_oneshot.sh, paths changed, 3 checks added) (runs in Debian/proot): a copy with the native-Termux check disabled and paths moved
# to a temp dir runs against fake su/settings/pgrep/sleep. Checks the screen-timeout read/set/restore (DECISIONS
# #123), the refusals (agent, llama-server, charging at start and after the idle) and that the runner gets the arguments.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
H=$T/home
mkdir -p "$T/bin" "$T/dlt" "$T/sys/class/power_supply/battery" "$H/robot/benchmark/thermal_char"
for z in 9 10 11; do mkdir -p "$T/sys/class/thermal/thermal_zone$z"; echo 36000 > "$T/sys/class/thermal/thermal_zone$z/temp"; done
sed -e 's#^H=/data/data/com.termux/files/home#H='"$H"'#' -e 's#\[ -d /termux-home \] || ##' \
    -e 's#/data/local/tmp#'"$T/dlt"'#g' -e 's#/sys/#'"$T/sys/"'#g' "$HERE/oneshot.sh" > "$T/oneshot.sh"
printf '#!/bin/bash\necho "RUNNER ARGS: $*"\n' > "$H/robot/benchmark/thermal_char/run_thermal_char.sh"
cat > "$T/bin/su" <<EOF
#!/bin/bash
[ "\$1" = -c ] || exit 1
PATH=$T/bin:\$PATH exec bash -c "\$2"
EOF
cat > "$T/bin/settings" <<EOF
#!/bin/bash
case "\$FAKE_SETTINGS:\$1" in
  broken:get) echo "Failure calling service settings: Failed transaction (2147483646)"; exit 0;;
  *:get) cat $T/screen;;
  ignore_put:put) exit 0;;
  fail_restore:put) [ "\$4" = 2147483647 ] && echo "\$4" > $T/screen; exit 0;;
  *:put) echo "\$4" > $T/screen;;
esac
EOF
printf '#!/bin/bash\n[ -n "${FAKE_PGREP:-}" ] && [[ $* == *"$FAKE_PGREP"* ]] && { echo "123 $FAKE_PGREP"; exit 0; }; exit 1\n' > "$T/bin/pgrep"
cat > "$T/bin/sleep" <<EOF
#!/bin/bash
[ "\$1" = 300 ] && [ -n "\${FAKE_CHARGE_DURING_IDLE:-}" ] && echo Charging > $T/sys/class/power_supply/battery/status
[ "\$1" = 300 ] && [ -n "\${FAKE_LONG_IDLE:-}" ] && exec /bin/sleep 30
/bin/sleep 0.1
EOF
printf '#!/bin/bash\n' > "$T/bin/termux-wake-lock"; cp "$T/bin/termux-wake-lock" "$T/bin/termux-wake-unlock"
chmod +x "$T/bin/"* "$H/robot/benchmark/thermal_char/run_thermal_char.sh"
fail=0
run() { PATH=$T/bin:$PATH PREFIX=/data/data/com.termux/files/usr bash "$T/oneshot.sh" "$@" 2>&1; }
check() { if grep -q -- "$2" <<<"$1"; then echo "ok: $3"; else echo "FAIL: $3"; echo "$1" | tail -8; fail=1; fi; }

echo Discharging > "$T/sys/class/power_supply/battery/status"
echo 60000 > "$T/screen"
out=$(run --smoke)
check "$out" 'screen timeout was 60000 ms' 'old timeout read through the file'
check "$out" 'RUNNER ARGS: --smoke' 'runner gets the arguments'
check "$out" 'restored to 60000 ms; read back: 60000' 'timeout restored and read back'
[ "$(cat "$T/screen")" = 60000 ] && echo 'ok: setting is 60000 afterwards' || { echo 'FAIL: setting not restored'; fail=1; }

out=$(FAKE_SETTINGS=broken run)
check "$out" 'Could not read the screen timeout' 'error text is not taken as a timeout'
if grep -q 'RUNNER ARGS' <<<"$out"; then echo 'FAIL: ran without a screen timeout'; fail=1; else echo 'ok: not started'; fi
if grep -q 'restored' <<<"$out"; then echo 'FAIL: restore attempted after a failed read'; fail=1; else echo 'ok: no restore after a failed read'; fi
[ "$(cat "$T/screen")" = 60000 ] && echo 'ok: setting untouched' || { echo 'FAIL: setting changed'; fail=1; }

out=$(FAKE_SETTINGS=ignore_put run)
check "$out" "Could not set the screen timeout (read back '60000')" 'unchanged setting after put is caught'
if grep -q 'RUNNER ARGS' <<<"$out"; then echo 'FAIL: ran with the screen timeout unset'; fail=1; else echo 'ok: not started'; fi
echo 60000 > "$T/screen"
out=$(FAKE_SETTINGS=fail_restore run --smoke); rc=$?
check "$out" "screen timeout NOT restored (read back '2147483647', expected 60000)" 'failed restore is reported'
[ "$rc" != 0 ] && echo "ok: failed restore exits non-zero ($rc)" || { echo 'FAIL: failed restore exited 0'; fail=1; }
echo 60000 > "$T/screen"
out=$(FAKE_PGREP=coresidency run); check "$out" 'A co-residency or thermal runner is already running' 'refuses with a runner running'
mkdir "$H/thermal_char/.lock"; echo 4242 > "$H/thermal_char/.lock/pid"; rm -f "$H/thermal_char/.stop_thermal"
out=$(run); check "$out" 'Another thermal launcher holds .*(pid 4242)' 'refuses while another launcher holds the lock'
[ -d "$H/thermal_char/.lock" ] && [ ! -f "$H/thermal_char/.stop_thermal" ] && echo "ok: the other run's lock and logger untouched" \
  || { echo "FAIL: refused launcher touched the other run's lock or logger"; fail=1; }
if grep -q 'restored' <<<"$out"; then echo 'FAIL: refused launcher ran cleanup'; fail=1; else echo 'ok: refused launcher ran no cleanup'; fi
rm -r "$H/thermal_char/.lock"
out=$(FAKE_PGREP=codex run); check "$out" 'An agent is still running' 'refuses with codex running'
out=$(FAKE_PGREP=llama-server run); check "$out" 'llama-server or chat.py is already running' 'refuses with llama-server running'
printf '#!/bin/bash\necho boom; exit 3\n' > "$H/robot/benchmark/thermal_char/run_thermal_char.sh"
out=$(run); rc=$?
check "$out" 'RUNNER FAILED (exit 3)' 'runner failure reported'
[ "$rc" = 3 ] && echo 'ok: launcher exits with the runner status' || { echo "FAIL: launcher exit $rc"; fail=1; }
check "$out" 'restored to 60000 ms' 'timeout restored after a runner failure'
printf '#!/bin/bash\ntrap "echo RUNNER GOT TERM; exit 143" TERM\necho RUNNER STARTED\nfor i in $(seq 300); do /bin/sleep 0.1; done\n' \
  > "$H/robot/benchmark/thermal_char/run_thermal_char.sh"
for sig in TERM HUP INT; do
  echo 60000 > "$T/screen"; rm -f "$H/thermal_char/.stop_thermal"
  # a background job of this script starts with SIGINT ignored (bash cannot trap that); reset it as a terminal would
  FAKE_LONG_IDLE=1 PATH=$T/bin:$PATH PREFIX=/data/data/com.termux/files/usr python3 -c \
    'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp("bash", ["bash", sys.argv[1]])' \
    "$T/oneshot.sh" > "$T/idle.out" 2>&1 & launcher=$!
  for i in $(seq 100); do grep -q 'idling 5 minutes' "$T/idle.out" && break; /bin/sleep 0.1; done
  start=$SECONDS; kill -$sig $launcher; wait $launcher; rc=$?
  out=$(cat "$T/idle.out")
  [ $((SECONDS - start)) -lt 10 ] && echo "ok: SIG$sig during the idle stops the launcher at once (exit $rc)" || { echo "FAIL: SIG$sig during idle took $((SECONDS - start)) s"; fail=1; }
  check "$out" 'restored to 60000 ms; read back: 60000' "timeout restored after SIG$sig during the idle"
  [ -f "$H/thermal_char/.stop_thermal" ] && echo "ok: logger and watcher told to stop after SIG$sig during the idle" || { echo 'FAIL: no .stop_thermal'; fail=1; }
  if grep -q 'RUNNER' <<<"$out"; then echo "FAIL: runner started after SIG$sig during the idle"; fail=1; fi
done
for sig in TERM HUP; do
  echo 60000 > "$T/screen"
  PATH=$T/bin:$PATH PREFIX=/data/data/com.termux/files/usr bash "$T/oneshot.sh" > "$T/sig.out" 2>&1 & launcher=$!
  for i in $(seq 100); do grep -q 'RUNNER STARTED' "$T/sig.out" && break; /bin/sleep 0.1; done
  kill -$sig $launcher; wait $launcher; rc=$?
  out=$(cat "$T/sig.out")
  check "$out" 'RUNNER GOT TERM' "SIG$sig to the launcher reaches the runner"
  check "$out" 'RUNNER FAILED (exit 143)' "launcher waits for the runner after SIG$sig (exit $rc)"
  check "$out" 'restored to 60000 ms; read back: 60000' "timeout restored after SIG$sig"
done
printf '#!/bin/bash\necho "RUNNER ARGS: $*"\n' > "$H/robot/benchmark/thermal_char/run_thermal_char.sh"
out=$(FAKE_CHARGE_DURING_IDLE=1 run); rc=$?
check "$out" "Battery status is 'Charging'" 'charger plugged in during the idle is caught'
if grep -q 'RUNNER ARGS' <<<"$out"; then echo 'FAIL: ran while charging'; fail=1; else echo "ok: runner not started (exit $rc)"; fi
check "$out" 'restored to 60000 ms' 'timeout restored after the late refusal'
echo Charging > "$T/sys/class/power_supply/battery/status"
out=$(run); check "$out" "Battery status is 'Charging'" 'refuses while charging'
n=$(pgrep -fc "$H/thermal_char/\.stop_thermal" || true)
[ "${n:-0}" = 0 ] && echo 'ok: no logger or watcher left running' || { echo "FAIL: $n logger/watcher processes left"; fail=1; }
[ ! -e "$H/thermal_char/.lock" ] && echo 'ok: lock released after every run' || { echo 'FAIL: lock left behind'; fail=1; }

echo 60000 > "$T/screen"
out=$(FAKE_PGREP=thermal_char run); check "$out" 'A co-residency or thermal runner is already running' 'refuses with a thermal runner running'
mkdir -p "$H/coresidency/.lock"
out=$(run); check "$out" 'The co-residency launcher holds' 'refuses while the co-residency launcher holds its lock'
if grep -q 'RUNNER ARGS' <<<"$out"; then echo 'FAIL: ran beside a co-residency launcher'; fail=1; fi
rm -r "$H/coresidency/.lock"
echo Discharging > "$T/sys/class/power_supply/battery/status"
out=$(run --smoke --note "22 degC, on the robot"); check "$out" 'RUNNER ARGS: --smoke --note 22 degC, on the robot' 'note with spaces reaches the runner'
exit $fail
