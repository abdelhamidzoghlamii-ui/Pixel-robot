#!/bin/bash
cd /termux-home/ladder && python3 conv_quality.py /termux-home/ladder/conversation_quality --runs 3 > /termux-home/ladder/conversation_quality.stdout.txt 2> /termux-home/ladder/conversation_quality.stderr.txt
echo "quality exit $?"
