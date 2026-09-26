#!/bin/bash
# s1o speed variants on the full ladder. Research only: offline, no motors, no main.py.
# Every variant: cold main block (60 cases x 2 orders, 4 threads), then cached t3 and t4 blocks;
# s1o_b1609 (the baseline) runs its main block only.
# Every block waits until z9 <= the idle reading taken at start + 4 degC. If cores 4-7 are lost the run
# pauses, and redoes the interrupted block once they are back.
#
# Usage: run_s1o_speed.sh [OUT_DIR]          new run
#        run_s1o_speed.sh --resume OUT_DIR   keep OUT_DIR's completed valid blocks, run the rest
# Cold loads: with LADDER_COLD_HANDSHAKE=1 the runner creates /termux-home/ladder/.drop_request and waits up to
# 60 s for .drop_done ("ok" or "failed") from a native-Termux root watcher; without it, it prompts here.
#
# Before running, from native Termux (not proot), start the root thermal logger and leave it running:
#   su -c 'while true; do echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=$(cat /sys/class/thermal/thermal_zone9/temp) z10=$(cat /sys/class/thermal/thermal_zone10/temp) z11=$(cat /sys/class/thermal/thermal_zone11/temp)"; sleep 5; done >> /data/data/com.termux/files/home/ladder/thermal.log'
# Let the phone idle ~5 min with the logger running, keep Termux in the foreground (cores 4-7), then run
# this script inside Debian.
set -euo pipefail
cd /termux-home/ladder
RESUME=()
if [ "${1:-}" = --resume ]; then
  OUT=${2:?usage: run_s1o_speed.sh --resume OUT_DIR}
  RESUME=(--resume)
else
  OUT=${1:-/termux-home/ladder/s1o_speed_$(date -u +%Y%m%dT%H%M%SZ)}
fi
python3 ladder.py cases/ladder_cases_v1.jsonl --out "$OUT" "${RESUME[@]}" \
  --models s1o_b1609,s1o_b2351,s1o_b2351_q40,s1o_b2351_qwen2b,s1o_b2351_qwen08b --main-only s1o_b1609 \
  --threads 3,4 --thermal-log /termux-home/ladder/thermal.log \
  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
