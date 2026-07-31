import requests, time, subprocess

URL = 'http://127.0.0.1:8080/completion'

# Tensor G2 BIG cores throttle around 80-85°C. Keep safe margin.
SAFE = 75      # green
WARN = 82      # getting hot
CRIT = 88      # throttle imminent

SYS = "You are a robot navigation AI. Reply ONE word: FORWARD, LEFT, RIGHT, BACK, STOP. Then a short reason."
SCENE = "Mission: find person. I see: hallway ahead. Distance: 200cm. What do you do?"

def cpu_temp():
    hot = 0
    for z in (9, 10, 11):
        try:
            v = int(subprocess.check_output(['su','-c',f'cat /sys/class/thermal/thermal_zone{z}/temp'],
                    timeout=3, stderr=subprocess.DEVNULL).strip()) // 1000
            hot = max(hot, v)
        except: pass
    return hot

def infer():
    p = f"<start_of_turn>user\n{SYS}\n\n{SCENE}<end_of_turn>\n<start_of_turn>model\n"
    try:
        r = requests.post(URL, json={'prompt':p,'n_predict':30,'temperature':0.1,
            'cache_prompt':True,'stop':['<end_of_turn>']}, timeout=30)
        return r.json()['timings']['predicted_per_second']
    except: return 0

def test(rest, cycles=15):
    print(f'\n  REST={rest}s — {cycles} sustained cycles:')
    speeds, temps = [], []
    for i in range(cycles):
        s = infer()
        t = cpu_temp()
        speeds.append(s); temps.append(t)
        flag = ' 🔴CRIT' if t>=CRIT else ' 🟠WARN' if t>=WARN else ' 🟡' if t>=SAFE else ''
        print(f'    {i+1:2d}: {s:4.1f} tok/s | BIG {t}°C{flag}')
        if t >= CRIT:
            print('    ⛔ Critical temp — stopping')
            break
        time.sleep(rest)
    peak = sum(speeds[:4])/4
    tail = sum(speeds[-4:])/4
    throttle = (peak-tail)/peak*100 if peak else 0
    return max(temps), tail, throttle

print('='*56)
print('  REAL THERMAL BENCHMARK — BIG core temp (zone9)')
print(f'  Safe<{SAFE}° | Warn<{WARN}° | Crit<{CRIT}°')
print('='*56)
print(f'  Idle CPU temp: {cpu_temp()}°C')

results = []
for rest in [0, 1, 2]:
    mx, tail, thr = test(rest, 15)
    stable = mx < WARN and thr < 12
    results.append((rest, mx, tail, thr, stable))
    print(f'    → max {mx}°C | sustained {tail:.1f} tok/s | throttle {thr:.0f}% | {"✅" if stable else "🔴"}')
    print(f'    Cooling 40s...')
    time.sleep(40)

print('\n' + '='*56)
print('  SUMMARY')
print(f'  {"Rest":>5} | {"MaxTemp":>7} | {"Speed":>6} | {"Throttle":>8} | Safe?')
best = None
for rest, mx, tail, thr, stable in results:
    print(f'  {rest:>4}s | {mx:>6}° | {tail:>4.1f}t | {thr:>6.0f}% | {"✅ yes" if stable else "🔴 no"}')
    if stable and best is None: best = (rest, tail, mx)
print('='*56)
if best:
    rest, spd, mx = best
    print(f'  ✅ SAFE CYCLE: {rest}s rest → {spd:.1f} tok/s, peaks {mx}°C')
    print(f'     Gemma call: ~{30/spd:.1f}s gen + {rest}s rest')
    print(f'     Robot note: Python handles most cycles at 0 heat,')
    print(f'     Gemma fires only when stuck → real thermal load is low')
else:
    print('  ⚠️  All configs ran hot — needs heatsink or longer rest')
print('='*56)
