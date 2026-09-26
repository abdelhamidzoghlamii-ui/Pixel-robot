#!/bin/bash
S=$1; L=/termux-home/ladder; OUT=$S/e2e_run; FLAG=$S/lose_cores
rm -rf $OUT $FLAG; echo "zone9=BIG" > $S/fake_thermal.log
$S/fake_thermal.sh $S/fake_thermal.log >/dev/null 2>&1 & TP=$!
( while true; do if [ -f $L/.drop_request ] && [ ! -f $L/.drop_done ]; then echo failed > $L/.drop_done.tmp; mv $L/.drop_done.tmp $L/.drop_done; echo "[watcher] answered failed"; fi; sleep 0.3; done ) & WP=$!
sleep 1
ARGS="$S/quick5.jsonl --out $OUT --models s1o_b2351_qwen08b,s1o_b2351_qwen2b --main-only s1o_b2351_qwen08b --threads 3,4 --thermal-log $S/fake_thermal.log"
cd $L
LADDER_COLD_HANDSHAKE=1 python3 $S/quick2.py $FLAG $ARGS </dev/null > $S/e2e_1.out 2> $S/e2e_1.err & RP=$!
# lose cores 6 s into qwen2b main, for 15 s
until grep -q "s1o_b2351_qwen2b main\] thermal start" $S/e2e_1.out 2>/dev/null; do sleep 0.5; done
sleep 6; touch $FLAG; echo "[e2e] cores lost"; sleep 15; rm $FLAG; echo "[e2e] cores back"
# kill during qwen2b t3
until grep -q "s1o_b2351_qwen2b t3\] thermal start" $S/e2e_1.out 2>/dev/null; do sleep 0.5; done
sleep 4; kill $RP; wait $RP; echo "[e2e] run killed during qwen2b t3 (exit $?)"
sleep 2; pkill -P $RP 2>/dev/null; pgrep -af llama-server | grep -v pgrep
LADDER_COLD_HANDSHAKE=1 python3 $S/quick2.py $FLAG $ARGS --resume </dev/null > $S/e2e_2.out 2> $S/e2e_2.err; echo "[e2e] resume exit $?"
kill $TP $WP
