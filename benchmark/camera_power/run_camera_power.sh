#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo 'Run from native Termux, not Debian/proot.' >&2
  exit 2
fi
exec python -u /data/data/com.termux/files/home/robot/benchmark/camera_power/camera_power.py "$@"
