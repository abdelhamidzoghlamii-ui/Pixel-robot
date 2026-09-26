#!/bin/bash
# conv_quality for qwen35_4b only, --cache-ram 0 now in conv_speed.server_cmd (first attempts were killed by memory growth)
cd /termux-home/ladder && python3 -c "
import sys; sys.path.insert(0, '/termux-home/ladder')
import conv_speed as cs; cs.MODELS = {'qwen35_4b': cs.MODELS['qwen35_4b']}
import conv_quality as q; sys.argv = ['conv_quality.py', '/termux-home/ladder/conversation_quality_qwen35_4b', '--runs', '3']; q.main()
" > /termux-home/ladder/conversation_quality_qwen35_4b.stdout.txt 2> /termux-home/ladder/conversation_quality_qwen35_4b.stderr.txt
echo "exit $?"
