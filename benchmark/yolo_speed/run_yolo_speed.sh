#!/bin/bash
# YOLO speed ladder (yolo_speed.py): 14 configs, stock yolo11/yolo26 n/s/m at 320 and 640, the deployed
# yolo11m.onnx at 640 and its int8 dynamic-quantized copy. Research only: offline, no motors, no main.py.
# One cold config at a time: thermal gate z9 <= idle + 4 degC, page-cache drop through the handshake
# (LADDER_COLD_HANDSHAKE=1), fresh native-Termux python worker, 5 warm-up + 60 timed frames; losing cores 4-7
# redoes the config.
#
# Usage: run_yolo_speed.sh [OUT_DIR]          new run
#        run_yolo_speed.sh --resume OUT_DIR   keep OUT_DIR's completed configs, run the rest
# Unattended, from native Termux:  RUN_SCRIPT=run_yolo_speed.sh bash ~/ladder/oneshot.sh [--resume OUT_DIR]
set -euo pipefail
cd /termux-home/ladder
RESUME=()
if [ "${1:-}" = --resume ]; then
  OUT=${2:?usage: run_yolo_speed.sh --resume OUT_DIR}
  RESUME=(--resume)
else
  OUT=${1:-/termux-home/ladder/yolo_speed_$(date -u +%Y%m%dT%H%M%SZ)}
fi
python3 /termux-home/robot/benchmark/yolo_speed/yolo_speed.py --out "$OUT" "${RESUME[@]}" \
  --thermal-log /termux-home/ladder/thermal.log \
  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
