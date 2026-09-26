#!/bin/bash
# Conversation speed of five models on the robot's llama.cpp (server_manager.py's binary and flags).
# Server: server_manager.py binary and flags plus --cache-ram 0 (see conv_speed.server_cmd).
# Research only: offline, no motors, no main.py. Per model: a cold block and a cached block, each one pass over
# the 27 DECISIONS #110 prompts (conversation, Q&A, objective-setting): load time, time to first token, prompt and
# generation tok/s, peak RSS, thermal start/end. Every block waits until z9 <= idle + 4 degC; losing cores 4-7
# pauses the run and redoes the block; cold loads drop the page cache through the handshake (LADDER_COLD_HANDSHAKE=1).
#
# Usage: run_conversation.sh [OUT_DIR]          new run
#        run_conversation.sh --resume OUT_DIR   keep OUT_DIR's completed blocks, run the rest
# Unattended, from native Termux:  RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh [--resume OUT_DIR]
set -euo pipefail
cd /termux-home/ladder
RESUME=()
if [ "${1:-}" = --resume ]; then
  OUT=${2:?usage: run_conversation.sh --resume OUT_DIR}
  RESUME=(--resume)
else
  OUT=${1:-/termux-home/ladder/conversation_$(date -u +%Y%m%dT%H%M%SZ)}
fi
python3 conv_speed.py --out "$OUT" "${RESUME[@]}" \
  --models gemma_e2b_q4km,gemma_e2b_q40,gemma_e4b_q4km,qwen35_4b,qwen35_2b \
  --thermal-log /termux-home/ladder/thermal.log \
  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
