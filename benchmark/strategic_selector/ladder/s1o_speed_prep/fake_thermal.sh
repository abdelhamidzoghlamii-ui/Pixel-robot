#!/bin/bash
# synthetic stand-in for the root logger during the quick check only
while true; do echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=30000 z10=31000 z11=32000" >> "$1"; sleep 5; done
