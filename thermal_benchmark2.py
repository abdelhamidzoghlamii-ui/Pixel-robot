import os, time, requests, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

def get_temp():
    try:
        return int(os.popen('su -c "cat /sys/class/thermal/thermal_zone9/temp"').read().strip()) // 1000
    except:
        return 0

def get_batt():
    try:
        return int(os.popen('su -c "cat /sys/class/thermal/thermal_zone25/temp"').read().strip()) // 1000
    except:
        return 0

def llm_call():
    t0 = time.time()
    try:
        resp = requests.post('http://127.0.0.1:8080/completion', json={
            'prompt': '<start_of_turn>user\nHallway clear. Person center. Mission find Chiara. Last: FWD.<end_of_turn>\n<start_of_turn>model\n',
            'n_predict': 20,
            'temperature': 0.1,
            'stop': ['<end_of_turn>']
        }, timeout=30)
        tok_s = resp.json()['timings']['predicted_per_second']
        return round(time.time()-t0, 2), round(tok_s, 1)
    except:
        return 0, 0

def warmup():
    """Run 3 warmup calls to stabilize temperature."""
    print('  Warming up model (3 calls)...')
    for i in range(3):
        llm_call()
        print(f'  Warmup {i+1}/3 | temp: {get_temp()}°C')
    # Now wait for temp to stabilize
    print('  Waiting for temp to stabilize after warmup...')
    prev = 0
    for _ in range(30):
        t = get_temp()
        if abs(t - prev) <= 2 and t < 80:
            break
        prev = t
        time.sleep(3)
    print(f'  Stable at: {get_temp()}°C')

def run_sustained(rest_seconds, cycles=20, label=''):
    """
    Run sustained test AFTER warmup.
    Measures real sustained performance.
    """
    print(f'\n{"="*55}')
    print(f'  SUSTAINED TEST: {label}')
    print(f'  Rest: {rest_seconds}s | Cycles: {cycles}')
    print(f'{"="*55}')

    temps = []
    tok_speeds = []
    times = []
    throttle_events = 0
    emergency = False

    for i in range(1, cycles+1):
        elapsed = sum(times)
        temp = get_temp()
        batt = get_batt()
        temps.append(temp)

        llm_time, tok_s = llm_call()
        if tok_s > 0:
            tok_speeds.append(tok_s)
            times.append(llm_time)

        # Detect throttling
        if len(tok_speeds) > 5:
            recent_avg = sum(tok_speeds[-3:]) / 3
            early_avg  = sum(tok_speeds[:3]) / 3
            if recent_avg < early_avg * 0.75:
                throttle_events += 1

        status = '🔴' if temp > 85 else '🟡' if temp > 70 else '🟢'
        print(f'  [{i:2d}] {status}{temp}°C batt:{batt}°C | {tok_s}tok/s | {llm_time}s')

        # Battery is the real safety limit
        if batt > 40:
            print(f'  🔴 BATTERY TOO HOT: {batt}°C — stopping!')
            emergency = True
            break

        # CPU emergency
        if temp > 95:
            print(f'  🔴 CPU TOO HOT: {temp}°C — stopping!')
            emergency = True
            break

        time.sleep(rest_seconds)

    avg_temp  = round(sum(temps)/len(temps)) if temps else 0
    max_temp  = max(temps) if temps else 0
    avg_tok   = round(sum(tok_speeds)/len(tok_speeds), 1) if tok_speeds else 0
    min_tok   = round(min(tok_speeds), 1) if tok_speeds else 0
    total_t   = round(sum(times) + rest_seconds * len(times), 1)
    cyc_min   = round(len(temps) / (total_t / 60), 1) if total_t > 0 else 0

    print(f'\n  SUMMARY:')
    print(f'  Completed: {len(temps)}/{cycles} cycles')
    print(f'  Avg temp: {avg_temp}°C | Max: {max_temp}°C')
    print(f'  Avg tok/s: {avg_tok} | Min tok/s: {min_tok}')
    print(f'  Throttle events: {throttle_events}')
    print(f'  Cycles/min: {cyc_min}')
    print(f'  Emergency stop: {emergency}')

    return {
        'label': label, 'rest': rest_seconds,
        'cycles': len(temps), 'avg_temp': avg_temp,
        'max_temp': max_temp, 'avg_tok': avg_tok,
        'min_tok': min_tok, 'throttles': throttle_events,
        'cycles_min': cyc_min, 'emergency': emergency
    }

# ── Main ──────────────────────────────────────────────
print('='*55)
print('  GEMMA E2B SUSTAINED THERMAL BENCHMARK')
print('  Tests run AFTER warmup — real performance')
print('='*55)

input('\nPress ENTER when Gemma server is ready...')

# Single warmup for all tests
warmup()

# Best candidates from first test
# 1.0s had best avg temp (79°C)
# 4.0s had best tok/s (7.1) and lowest max (89°C)
# Also test 2.0s and 3.0s as middle ground
configs = [
    (1.0,  20, '1.0s rest'),
    (2.0,  20, '2.0s rest'),
    (3.0,  20, '3.0s rest'),
    (4.0,  20, '4.0s rest'),
    (5.0,  15, '5.0s rest'),
]

results = []
for rest, cycles, label in configs:
    r = run_sustained(rest, cycles, label)
    results.append(r)
    if r['emergency']:
        print('  Skipping remaining tests — too hot')
        break
    print(f'\n  Cooling 90s...')
    time.sleep(90)

# Final table
print(f'\n{"="*65}')
print(f'  FINAL COMPARISON — POST WARMUP')
print(f'{"="*65}')
print(f'  {"Rest":>6} {"Done":>6} {"Avg°C":>7} {"Max°C":>7} {"tok/s":>7} {"min tok":>8} {"Thrott":>7} {"cyc/min":>8}')
print(f'  {"-"*63}')

best = None
best_score = 0
for r in results:
    if r['emergency']:
        marker = ' STOPPED'
    else:
        # Score: cycles/min × speed, penalize high temp and throttling
        temp_penalty = max(1, (r['avg_temp'] - 60) / 5)
        throttle_penalty = max(1, r['throttles'] * 2)
        score = (r['cycles_min'] * r['avg_tok']) / (temp_penalty * throttle_penalty)
        if score > best_score:
            best_score = score
            best = r
        marker = ''
    print(f'  {r["rest"]:>5}s {r["cycles"]:>5} {r["avg_temp"]:>6}°C {r["max_temp"]:>6}°C {r["avg_tok"]:>6} {r["min_tok"]:>7} {r["throttles"]:>6} {r["cycles_min"]:>8}{marker}')

if best:
    print(f'\n  ✅ SWEET SPOT: {best["rest"]}s rest')
    print(f'  Sustained temp: {best["avg_temp"]}°C')
    print(f'  Speed: {best["avg_tok"]} tok/s')
    print(f'  Throughput: {best["cycles_min"]} cycles/min')
    print(f'\n  Add to main.py:')
    print(f'  CYCLE_MOVE_TIME = {best["rest"]}')
