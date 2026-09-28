#!/bin/bash
# s1o letter scoring on the robot's dotprod b1609 build with Gemma E2B Q4_0 (s1o_b1609dp_q40), to compare with
# s1o_b2351_q40 of run_s1o_speed.sh. Research only: offline, no motors, no main.py.
# Main block only: cold load, 60 cases x 2 orders, 4 threads. Same runner, cases, thermal gate, cpuset guard,
# cold handshake and --resume as run_s1o_speed.sh.
#
# Usage: run_s1o_dp_q40.sh [OUT_DIR]          new run
#        run_s1o_dp_q40.sh --resume OUT_DIR   keep OUT_DIR's completed valid blocks, run the rest
# Start it from native Termux with oneshot.sh: RUN_SCRIPT=run_s1o_dp_q40.sh bash ~/ladder/oneshot.sh
set -euo pipefail
cd /termux-home/ladder
RESUME=()
if [ "${1:-}" = --resume ]; then
  OUT=${2:?usage: run_s1o_dp_q40.sh --resume OUT_DIR}
  RESUME=(--resume)
else
  OUT=${1:-/termux-home/ladder/s1o_dp_q40_$(date -u +%Y%m%dT%H%M%SZ)}
fi
python3 ladder.py cases/ladder_cases_v1.jsonl --out "$OUT" "${RESUME[@]}" \
  --models s1o_b1609dp_q40 --main-only s1o_b1609dp_q40 \
  --threads 4 --thermal-log /termux-home/ladder/thermal.log \
  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
