#!/usr/bin/env python3
"""Log travel vs time from the ultrasonic during a forward run. Needs root.
Point the robot square at a flat wall 2-3m away.
  python3 log_run.py 130 run130.csv
"""
import sys, time
sys.path.insert(0, "/data/data/com.termux/files/home/robot")
from motors import Motors

DUTY      = int(sys.argv[1]) if len(sys.argv) > 1 else 130
OUT       = sys.argv[2] if len(sys.argv) > 2 else f"run{DUTY}.csv"
STOP_CM   = 30.0
MAX_S     = 8.0
KEEPALIVE = 0.2

m = Motors(); m.connect(); time.sleep(1)
if not m.ping():
    raise SystemExit("no ALIVE - check the link")

t_wait = time.time()
while time.time() - t_wait < 3.0:
    if time.time() - m.state["dist_at"] < 0.5 and m.state["distance"] > 0:
        break
    time.sleep(0.05)
else:
    m.disconnect(); raise SystemExit("no DIST: arriving - is HAS_ULTRASONIC on?")

d0 = m.state["distance"]
print(f"start {d0} cm, duty {DUTY}, stopping at {STOP_CM:.0f} cm")
if d0 < 100:
    m.disconnect(); raise SystemExit("need at least 1m of run-up")

samples, last_stamp, t0, last_ka = [], 0.0, None, 0.0
m.send(f"FORWARD:{DUTY}")
t_start = time.time()

try:
    while True:
        now = time.time()
        if now - last_ka >= KEEPALIVE:
            m.send(f"FORWARD:{DUTY}")
            last_ka = now
        stamp = m.state["dist_at"]
        if stamp != last_stamp:
            last_stamp = stamp
            d = m.state["distance"]
            if 0 < d < 400:
                if t0 is None:
                    t0 = stamp
                samples.append((stamp - t0, d0 - d))
                if d <= STOP_CM:
                    print(f"  reached {d} cm - stopping")
                    break
        if now - t_start > MAX_S:
            print("  timeout - stopping")
            break
        time.sleep(0.005)
finally:
    m.stop()
    time.sleep(1.0)
    print(f"  settled at {m.state['distance']} cm")
    m.disconnect()

if len(samples) < 8:
    raise SystemExit(f"only {len(samples)} samples - ping rate too low")

with open(OUT, "w") as f:
    for t, x in samples:
        f.write(f"{t:.4f},{x:.1f}\n")

dt = [samples[i+1][0] - samples[i][0] for i in range(len(samples)-1)]
print(f"\n{len(samples)} samples -> {OUT}")
print(f"  travel {samples[-1][1]:.0f} cm in {samples[-1][0]:.2f} s")
print(f"  sample interval {min(dt)*1000:.0f}-{max(dt)*1000:.0f} ms")
if min(dt) > 0.15:
    print("  NOTE: ~5Hz fits K_I but not T1/Tt.")
    print("        Set DIST_EVERY_MS = 50 in firmware for the transient.")
print(f"\nthen:  python3 identify_it1.py {OUT} {DUTY}")
