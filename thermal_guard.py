"""Thermal safety for the robot. Reads real BIG-core temp, throttles Gemma."""
import subprocess, time

WARN = 82   # start backing off
CRIT = 86   # force cooldown

def big_temp():
    hot = 0
    for z in (9, 10, 11):
        try:
            v = int(subprocess.check_output(
                ['su','-c',f'cat /sys/class/thermal/thermal_zone{z}/temp'],
                timeout=3, stderr=subprocess.DEVNULL).strip()) // 1000
            hot = max(hot, v)
        except: pass
    return hot

def check_and_wait(verbose=True):
    """Call before each Gemma inference. Pauses if too hot."""
    t = big_temp()
    if t >= CRIT:
        if verbose: print(f'  [THERMAL] {t}°C CRITICAL — cooling 10s')
        time.sleep(10)
        return big_temp()
    elif t >= WARN:
        if verbose: print(f'  [THERMAL] {t}°C warm — 2s pause')
        time.sleep(2)
    return t

if __name__ == '__main__':
    print(f'Current BIG core: {big_temp()}°C')
    print(f'WARN at {WARN}°C, CRITICAL at {CRIT}°C')
