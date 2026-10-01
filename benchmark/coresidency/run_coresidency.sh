#!/data/data/com.termux/files/usr/bin/bash
# Native Termux only; started by oneshot.sh in this folder.
# Usage: run_coresidency.sh [--smoke] [--resume RUN_DIR]
set -euo pipefail
if [ -d /termux-home ] || [ "${PREFIX:-}" != /data/data/com.termux/files/usr ]; then
  echo 'Run from native Termux, not Debian/proot.' >&2
  exit 2
fi
exec python -u /data/data/com.termux/files/home/robot/benchmark/coresidency/coresidency.py "$@"
