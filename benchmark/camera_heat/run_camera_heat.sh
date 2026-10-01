#!/bin/bash
# Native Termux after quitting the agent:
# ( bash ~/ladder/run_camera_heat.sh --helper </dev/null > ~/ladder/camera_heat_helper.log 2>&1 &
#   helper=$!; trap 'kill "$helper" 2>/dev/null || true' EXIT
#   RUN_SCRIPT=run_camera_heat.sh bash ~/ladder/oneshot.sh )
# Append --toy to oneshot.sh for four 20-second blocks.
set -euo pipefail
if [ -d /termux-home ]; then
  exec python3 /termux-home/robot/benchmark/camera_heat/camera_heat.py --run "$@"
fi
exec python /data/data/com.termux/files/home/robot/benchmark/camera_heat/camera_heat.py "$@"
