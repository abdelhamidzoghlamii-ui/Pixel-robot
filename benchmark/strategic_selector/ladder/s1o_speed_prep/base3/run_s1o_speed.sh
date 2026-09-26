#!/bin/bash
# s1o speed variants on the full ladder. Research only: offline, no motors, no main.py.
# Every variant: cold main block (60 cases x 2 orders, 4 threads), then cached t3 and t4 blocks.
# Every block waits until z9 <= the idle reading taken at start + 2 degC.
#
# Before running, from native Termux (not proot), start the root thermal logger and leave it running:
#   su -c 'while true; do echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=$(cat /sys/class/thermal/thermal_zone9/temp) z10=$(cat /sys/class/thermal/thermal_zone10/temp) z11=$(cat /sys/class/thermal/thermal_zone11/temp)"; sleep 5; done >> /data/data/com.termux/files/home/ladder/thermal.log'
# Let the phone idle ~5 min with the logger running, keep Termux in the foreground (cores 4-7), then run
# this script inside Debian. At each cold load it asks you to drop the page cache from native Termux.
set -euo pipefail
cd /termux-home/ladder
OUT=${1:-/termux-home/ladder/s1o_speed_$(date -u +%Y%m%dT%H%M%SZ)}
python3 ladder.py cases/ladder_cases_v1.jsonl --out "$OUT" \
  --models s1o_b1609,s1o_b2351,s1o_b2351_q40,s1o_b2351_qwen2b,s1o_b2351_qwen08b \
  --threads 3,4 --thermal-log /termux-home/ladder/thermal.log \
  > >(tee "$OUT.stdout.txt") 2> >(tee "$OUT.stderr.txt" >&2)
