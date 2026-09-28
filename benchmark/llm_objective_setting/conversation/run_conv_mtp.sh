#!/bin/bash
# MTP speculative decoding and Qwen3.5-4B Q4_0 speed on the robot's llama.cpp (server_manager.py's binary and flags
# plus --cache-ram 0). Research only: offline, no motors, no main.py. Five configs, one cold block each over 13 of the
# DECISIONS #110 prompts at temperature 0: gemma_e2b_q40, gemma_e2b_q40_mtp (drafter), qwen35_4b_q4km,
# qwen35_4b_q40mtp (MTP off), qwen35_4b_q40mtp_on. Every block waits until z9 <= idle + 4 degC; losing cores 4-7
# pauses the run and redoes the block; cold loads drop the page cache through the handshake (LADDER_COLD_HANDSHAKE=1).
# Afterwards score_c.py grades the C replies (report only) into OUT_DIR/c_scores.txt.
#
# Usage: run_conv_mtp.sh [OUT_DIR]          new run
#        run_conv_mtp.sh --resume OUT_DIR   keep OUT_DIR's completed blocks, run the rest
# Unattended, from native Termux:  RUN_SCRIPT=run_conv_mtp.sh bash ~/ladder/oneshot.sh [--resume OUT_DIR]
set -euo pipefail
C=/termux-home/robot/benchmark/llm_objective_setting/conversation
cd /termux-home/ladder
RESUME=()
if [ "${1:-}" = --resume ]; then
  OUT=${2:?usage: run_conv_mtp.sh --resume OUT_DIR}
  RESUME=(--resume)
else
  OUT=${1:-/termux-home/ladder/conv_mtp_$(date -u +%Y%m%dT%H%M%SZ)}
fi
python3 "$C/conv_speed_mtp.py" --out "$OUT" "${RESUME[@]}" \
  --thermal-log /termux-home/ladder/thermal.log \
  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
# report only: the archived graders can raise on odd reply shapes; report.txt and c_results.json are already written
python3 "$C/score_c.py" "$OUT/c_results.json" > "$OUT/c_scores.txt" 2>&1 || echo "score_c.py FAILED (exit $?)" >> "$OUT/c_scores.txt"
cat "$OUT/c_scores.txt"
