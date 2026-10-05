"""Read-only LITTLE-cap evidence. Missing optional nodes remain explicit."""
import math
import time


class BatteryStop(RuntimeError):
    pass


def parse(text):
    values = {}
    for line in text.splitlines():
        if '\t' not in line:
            continue
        path, value = line.split('\t', 1)
        values[path] = value
    return values


def battery(cr, minimum):
    rc, text = cr.root(f'cat {cr.BATTERY}/capacity', 'campaign_capacity')
    try:
        level = int(text.strip())
    except ValueError:
        raise RuntimeError('battery capacity unreadable')
    if rc or not 0 <= level <= 100:
        raise RuntimeError('battery capacity unreadable')
    if level < minimum:
        error = BatteryStop if minimum==25 else RuntimeError
        raise error(f'battery {level}% below {minimum}%')
    return level


def snapshot(cr, camera_on):
    # Read sysfs only; never write caps, boosts or power hints.
    began=time.monotonic()
    script = f'''
read_node() {{
  if [ -r "$1" ]; then
    value=$(cat "$1" 2>/dev/null) || value=UNREADABLE
    printf '%s\\t%s\\n' "$1" "$value"
  else printf '%s\\tUNREADABLE\\n' "$1"; fi
}}
for f in {cr.BATTERY}/capacity {cr.BATTERY}/temp \
 /sys/class/thermal/thermal_zone*/type /sys/class/thermal/thermal_zone*/temp \
 /sys/class/thermal/cooling_device*/type /sys/class/thermal/cooling_device*/cur_state \
 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq \
 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; do read_node "$f"; done
paths=$(find /sys/devices/platform /sys/devices/system/cpu /sys/kernel /sys/power /dev/cpuctl -maxdepth 5 -type f 2>/dev/null | \
 grep -Ei '(boost.*(min|max)|(min|max).*boost|power.?hint|/(cpu|gpu|mif|int).*_(min|max)_freq)')
printf 'optional_discovery\\tbest effort: readable boost min/max and power-hint nodes\\n'
for f in $paths; do read_node "$f"; done
'''
    # cr.root appends '; }': a trailing newline before that semicolon cannot parse.
    rc, text = cr.root(script.strip(), 'campaign_diagnostics')
    if rc:
        raise RuntimeError('diagnostics root read failed')
    nodes = parse(text)
    for path in (cr.BATTERY+'/capacity', cr.BATTERY+'/temp',
                 '/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq'):
        try:
            if not math.isfinite(float(nodes[path])):
                raise ValueError('nonfinite')
        except (KeyError, ValueError):
            raise RuntimeError('required diagnostics unreadable: '+path)
    dump = cr.read_dump()
    ended=time.monotonic()
    return dict(t=ended, read_elapsed_s=ended-began, camera_on=camera_on,
                camera_state_source='requested state; startup/stop checked by runner, no snapshot state readback',
                battery_level=int(nodes[cr.BATTERY+'/capacity']),
                battery_temperature_c=float(nodes[cr.BATTERY+'/temp'])/10,
                android_thermal_status=dump['status'], skin=dump['skin'], nodes=nodes,
                raw_node_output=text)
