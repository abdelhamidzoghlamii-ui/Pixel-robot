#!/bin/bash
# $1 = extra server args; tries prompts of N tokens (token id 1000 repeated, after BOS 2)
U=/termux-home/llama.cpp-upstream/build/bin
LD_LIBRARY_PATH=$U:/data/data/com.termux/files/usr/lib $U/llama-server -m /termux-home/models/gemma-4-e2b-it-q4_k_m.gguf --threads 4 --threads-batch 4 --parallel 1 --ctx-size 2048 --host 127.0.0.1 --port 8093 $1 > $2 2>&1 &
PID=$!
until curl -sf 127.0.0.1:8093/health >/dev/null; do sleep 0.5; kill -0 $PID 2>/dev/null || { echo "server died at load"; exit; }; done
for n in 40 63 64 65 70 77; do
  ids=$(python3 -c "print([2]+[1000]*($n-1))")
  r=$(curl -s -m 120 127.0.0.1:8093/completion -d "{\"prompt\": $ids, \"n_predict\": 1, \"n_probs\": 10, \"cache_prompt\": false}" | head -c 60)
  if kill -0 $PID 2>/dev/null; then echo "$n ok"; else wait $PID; echo "$n CRASH exit $?"; exit; fi
done
kill $PID; wait $PID 2>/dev/null
