#!/bin/bash
# Four memcheck runs in sequence; before each, wait for zone9 < 50 C (max 5 min) from ladder/thermal.log.
cd /termux-home/memcheck
cool() {
  for i in $(seq 60); do
    t=$(tail -1 /termux-home/ladder/thermal.log | sed -n 's/.*z9=\([0-9]*\).*/\1/p')
    [ -n "$t" ] && [ "$t" -lt 50000 ] && break
    sleep 5
  done
  echo "$(date -u +%FT%TZ) cool-wait end: $(tail -1 /termux-home/ladder/thermal.log)"
}
for spec in "robot_default:--shape robot" "robot_cram0:--shape robot --cache-ram-0" \
            "conv_default:--shape conv" "conv_cram0:--shape conv --cache-ram-0"; do
  name=${spec%%:*}; args=${spec#*:}
  cool
  python3 memcheck.py "$name" $args > "$name.stdout.txt" 2> "$name.stderr.txt"
  echo "$(date -u +%FT%TZ) $name exit=$?"
done
