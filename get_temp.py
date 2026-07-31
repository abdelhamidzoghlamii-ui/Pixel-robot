#!/usr/bin/env python3
"""Real SoC temperature reader for Pixel 7 (needs root)."""
import subprocess

# Zones that matter for inference thermal management
ZONES = {9: 'BIG', 10: 'MID', 11: 'LITTLE', 12: 'GPU', 14: 'TPU', 22: 'battery'}

def read_zone(z):
    try:
        out = subprocess.check_output(
            ['su', '-c', f'cat /sys/class/thermal/thermal_zone{z}/temp'],
            timeout=5, stderr=subprocess.DEVNULL)
        return int(out.strip()) // 1000
    except:
        return None

def get_temps():
    return {name: read_zone(z) for z, name in ZONES.items()}

def get_cpu_max():
    """The number that matters — hottest CPU cluster."""
    temps = [read_zone(z) for z in (9, 10, 11)]
    valid = [t for t in temps if t is not None]
    return max(valid) if valid else None

if __name__ == '__main__':
    t = get_temps()
    for name, val in t.items():
        bar = '█' * (val // 3) if val else ''
        print(f'  {name:8}: {val}°C {bar}')
    print(f'\n  CPU hotspot: {get_cpu_max()}°C')
