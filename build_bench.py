#!/usr/bin/env python3
"""A/B benchmark against whatever llama-server is on 8080.
  python3 build_bench.py <label> [cycles]
Realistic robot shape: fixed system prefix (cached) + varying scene (fresh).
"""
import json, subprocess, sys, time, urllib.request

LABEL  = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
CYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
REST   = 2.0          # DECISIONS #30: rest raises sustained throughput
COOL_C = 60.0         # don't start hot

SYS = ("You are the navigation brain of a four-wheel mecanum robot exploring "
       "indoors. You receive a scene description from an object detector and a "
       "forward distance in centimetres. Choose exactly one move from: FORWARD, "
       "BACK, LEFT, RIGHT, STRAFE_LEFT, STRAFE_RIGHT, STOP. Answer with the move "
       "and one short sentence of reasoning. Never drive forward into an "
       "obstacle closer than 25cm.")

SCENES = [
    "chair right, couch center, table ahead. Distance ahead: 120cm.",
    "person center, tv left, bottle right. Distance ahead: 85cm.",
    "refrigerator center, sink left. Distance ahead: 200cm.",
    "bed center, lamp right. Distance ahead: 45cm.",
    "doorway ahead, plant left. Distance ahead: 300cm.",
]

def temp():
    try:
        out = subprocess.run(["su", "-c", "cat /sys/class/thermal/thermal_zone9/temp"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) / 1000.0
    except Exception:
        return float("nan")

def mem_free_mb():
    try:
        for line in subprocess.run(["free", "-m"], capture_output=True,
                                   text=True).stdout.splitlines():
            if line.lower().startswith("mem"):
                return int(line.split()[6])   # available
    except Exception:
        pass
    return -1

def ask(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 40,
                       "cache_prompt": True, "temperature": 0.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["timings"]

print(f"=== {LABEL} — {CYCLES} cycles, {REST}s rest ===")
t = temp()
while t > COOL_C:
    print(f"  cooling: zone9 {t:.0f}C > {COOL_C:.0f}C, waiting 30s")
    time.sleep(30)
    t = temp()
print(f"  start temp {t:.0f}C, free RAM {mem_free_mb()} MB\n")

rows = []
for i in range(CYCLES):
    scene = SCENES[i % len(SCENES)]
    tm = ask(f"{SYS}\n\nScene: {scene}\nYour move:")
    c = temp()
    rows.append((tm["prompt_ms"], tm["prompt_n"],
                 tm["predicted_per_second"], c))
    print(f"  {i+1:2}  prompt {tm['prompt_ms']:7.1f}ms ({tm['prompt_n']:3} tok)"
          f"  gen {tm['predicted_per_second']:5.1f} tok/s   {c:.0f}C")
    time.sleep(REST)

def med(v):
    s = sorted(v); n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

gen  = [r[2] for r in rows]
pm   = [r[0] for r in rows]
temps= [r[3] for r in rows]
print(f"\n--- {LABEL} ---")
print(f"  gen tok/s     median {med(gen):.1f}   min {min(gen):.1f}   max {max(gen):.1f}")
print(f"  prompt ms     median {med(pm):.1f}")
print(f"  zone9 C       start {temps[0]:.0f}   peak {max(temps):.0f}   end {temps[-1]:.0f}")
print(f"  free RAM      {mem_free_mb()} MB")
print(f"  cached tok    {[r[1] for r in rows]}")
