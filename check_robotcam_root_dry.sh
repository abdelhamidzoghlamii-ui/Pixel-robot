#!/data/data/com.termux/files/usr/bin/bash
# Native Termux only. Root-form mission check; --dry never opens motor USB.
set -u
if [[ ${PREFIX:-} != /data/data/com.termux/files/usr ]]; then
    echo 'Run from native Termux, not Debian/proot.' >&2
    exit 2
fi

log=/storage/emulated/0/Download/robotcam_root_dry.log
start=$SECONDS
su -c 'LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib /data/data/com.termux/files/usr/bin/timeout -s TERM 30s /data/data/com.termux/files/usr/bin/python -u /data/data/com.termux/files/home/robot/run_mission.py --dry "explore and map the rooms" </dev/null >/storage/emulated/0/Download/robotcam_root_dry.log 2>&1'
status=$?
printf 'su exit=%s elapsed=%ss log=%s\n' "$status" "$((SECONDS-start))" "$log"
tail -n 20 "$log"
if [[ $status != 0 && $status != 124 ]]; then
    exit "$status"
fi
