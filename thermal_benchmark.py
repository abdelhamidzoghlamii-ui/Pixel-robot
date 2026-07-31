import requests, time, subprocess, json

URL = 'http://127.0.0.1:8080/completion'

# Battery temp thresholds (proxy for SoC — battery runs ~15-20°C cooler than CPU)
# Battery 40°C ≈ SoC ~60°C | Battery 45°C ≈ SoC ~70°C | Battery 48°C ≈ SoC ~78°C
SAFE_BATTERY = 43      # keep below this
WARN_BATTERY = 46      # throttle warning
CRITICAL     = 48      # stop

SYS = "You are a robot navigation AI. Reply ONE word: FORWARD, LEFT, RIGHT, BACK, STOP. Then one short reason."
SCENE = "Mission: find person. I see: hallway ahead. Distance: 200cm. What do you do?"

def get_battery_temp():
    try:
        out = subprocess.check_output(['termux-battery-status'], timeout=5)
        return json.loads(out)['temperature']
    except:
        return 0.0

def infer():
    prompt = f"<start_of_turn>user\n{SYS}\n\n{SCENE}<end_of_turn>\n<start_of_turn>model\n"
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            'prompt': prompt, 'n_predict': 30, 'temperature': 0.1,
            'cache_prompt': True, 'stop': ['<end_of_turn>']
        }, timeout=30)
        d = r.json()
        return d['timings']['predicted_per_second'], time.time()-t0
    except:
        return 0, 0

def test_rest(rest_time, cycles=15):
    """Run N cycles with given rest time, track temp trajectory."""
    print(f'\n  Testing REST={rest_time}s ({cycles} cycles)...')
    temps = []
    speeds = []
    start_temp = get_battery_temp()

    for i in range(cycles):
        tok_s, elapsed = infer()
        temp = get_battery_temp()
        temps.append(temp)
        speeds.append(tok_s)
        flag = ''
        if temp >= CRITICAL: flag = ' 🔴 CRITICAL'
        elif temp >= WARN_BATTERY: flag = ' 🟠 WARN'
        elif temp >= SAFE_BATTERY: flag = ' 🟡'
        print(f'    Cycle {i+1:2d}: {temp:.1f}°C | {tok_s:.1f} tok/s{flag}')
        if temp >= CRITICAL:
            print(f'    ⛔ Hit critical temp — stopping this test')
            break
        time.sleep(rest_time)

    end_temp = get_battery_temp()
    rise = end_temp - start_temp
    return {
        'rest': rest_time,
        'start': start_temp,
        'end': end_temp,
        'max': max(temps),
        'rise': rise,
        'avg_speed': sum(speeds)/len(speeds),
        'stable': max(temps) < WARN_BATTERY
    }

print('='*58)
print('  THERMAL BENCHMARK — Finding safe sustainable cycle time')
print(f'  Safe<{SAFE_BATTERY}°C | Warn<{WARN_BATTERY}°C | Critical<{CRITICAL}°C (battery)')
print('='*58)

start = get_battery_temp()
print(f'\n  Starting battery temp: {start}°C')
if start > 42:
    print('  ⚠️  Phone already warm — let it cool a few min for accurate results')

results = []
# Test from fast (no rest) to slow (more rest between inferences)
for rest in [0, 1, 2, 3]:
    r = test_rest(rest, cycles=15)
    results.append(r)
    # Cool down between tests
    print(f'    → max {r["max"]:.1f}°C, rise +{r["rise"]:.1f}°C, {r["avg_speed"]:.1f} tok/s')
    print(f'    Cooling 30s before next test...')
    time.sleep(30)

print('\n' + '='*58)
print('  SUMMARY — cycle rest time vs thermal')
print('='*58)
print(f'  {"Rest":>5} | {"Max°C":>6} | {"Rise":>6} | {"Speed":>6} | Verdict')
print(f'  {"-"*5} | {"-"*6} | {"-"*6} | {"-"*6} | -------')
best = None
for r in results:
    verdict = '✅ safe' if r['stable'] else '🔴 hot'
    print(f'  {r["rest"]:>4}s | {r["max"]:>5.1f} | +{r["rise"]:>4.1f} | {r["avg_speed"]:>4.1f}t | {verdict}')
    if r['stable'] and best is None:
        best = r

print('='*58)
if best:
    print(f'  ✅ RECOMMENDED: {best["rest"]}s rest between Gemma calls')
    print(f'     Keeps battery under {WARN_BATTERY}°C, {best["avg_speed"]:.1f} tok/s sustained')
    cycle_total = 30/best['avg_speed'] + best['rest']
    print(f'     Full nav cycle: ~{cycle_total:.1f}s ({60/cycle_total:.1f} cycles/min)')
else:
    print(f'  ⚠️  All tests ran hot — need more aggressive rest or cooling')
print('='*58)
