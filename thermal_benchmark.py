import os, time, requests, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

def get_temp(zone=9):
    try:
        return int(os.popen(f'su -c "cat /sys/class/thermal/thermal_zone{zone}/temp"').read().strip()) // 1000
    except:
        return 0

def get_all_temps():
    zones = {'BIG':9, 'MID':10, 'LITTLE':11, 'GPU':12, 'BATTERY':25}
    return {name: get_temp(zone) for name, zone in zones.items()}

def llm_call():
    t0 = time.time()
    try:
        resp = requests.post('http://127.0.0.1:8080/completion', json={
            'prompt': '<start_of_turn>user\nHallway ahead clear. Person center. Mission find Chiara. Last: FWD,FWD.<end_of_turn>\n<start_of_turn>model\n',
            'n_predict': 20,
            'temperature': 0.1,
            'stop': ['<end_of_turn>']
        }, timeout=30)
        tok_s = resp.json()['timings']['predicted_per_second']
        return round(time.time()-t0, 2), round(tok_s, 1)
    except:
        return 0, 0

def simulate_yolo():
    """Simulate YOLO inference heat without actual model."""
    time.sleep(0.7)  # realistic YOLO time

def cool_down(seconds):
    """Wait for cooling."""
    time.sleep(seconds)

def run_benchmark(cycle_rest, duration=120, label=""):
    """
    Run cycles for duration seconds with cycle_rest between each.
    Returns performance and thermal stats.
    """
    print(f'\n{"="*55}')
    print(f'  TEST: {label}')
    print(f'  Cycle rest: {cycle_rest}s | Duration: {duration}s')
    print(f'{"="*55}')

    # Wait for stable start temp
    print('  Waiting for stable temperature...')
    while get_temp() > 45:
        time.sleep(5)
        print(f'  Cooling... {get_temp()}°C')

    start_temp = get_temp()
    print(f'  Start temp: {start_temp}°C')

    start_time = time.time()
    cycle = 0
    temps = []
    tok_speeds = []
    throttle_events = 0
    max_temp = 0

    while time.time() - start_time < duration:
        cycle += 1
        elapsed = round(time.time() - start_time, 1)

        # Simulate YOLO
        simulate_yolo()

        # LLM call
        llm_time, tok_s = llm_call()
        if tok_s == 0:
            print(f'  [{cycle:3d}] {elapsed}s — LLM failed')
            continue

        # Check temp
        temp = get_temp()
        temps.append(temp)
        tok_speeds.append(tok_s)
        max_temp = max(max_temp, temp)

        # Detect throttling (tok/s drops significantly)
        if len(tok_speeds) > 3 and tok_s < (sum(tok_speeds[:3])/3 * 0.7):
            throttle_events += 1

        status = '🔴' if temp > 80 else '🟡' if temp > 65 else '🟢'
        print(f'  [{cycle:3d}] {elapsed}s {status}{temp}°C | {tok_s}tok/s | {llm_time}s')

        # Emergency stop
        if temp > 88:
            print(f'  🔴 EMERGENCY STOP: {temp}°C')
            break

        # Cooling window
        cool_down(cycle_rest)

    duration_actual = round(time.time() - start_time, 1)
    avg_temp = round(sum(temps)/len(temps)) if temps else 0
    avg_tok  = round(sum(tok_speeds)/len(tok_speeds), 1) if tok_speeds else 0

    result = {
        'label':      label,
        'rest':       cycle_rest,
        'cycles':     cycle,
        'duration':   duration_actual,
        'avg_temp':   avg_temp,
        'max_temp':   max_temp,
        'avg_tok_s':  avg_tok,
        'throttles':  throttle_events,
        'cycles_min': round(cycle / (duration_actual/60), 1)
    }

    print(f'\n  RESULT:')
    print(f'  Cycles: {cycle} ({result["cycles_min"]}/min)')
    print(f'  Avg temp: {avg_temp}°C | Max: {max_temp}°C')
    print(f'  Avg speed: {avg_tok} tok/s')
    print(f'  Throttle events: {throttle_events}')

    return result

# ── Main ──────────────────────────────────────────────
print('='*55)
print('  THERMAL SWEET SPOT BENCHMARK')
print('  Finding optimal cycle rest time')
print('  Each test runs 2 minutes')
print('='*55)
print('\nMake sure llama-server is running first!')
print('Run: python3 server_manager.py setup_c')
input('\nPress ENTER when server is ready...')

results = []

# Test different rest times
test_configs = [
    (0.5,  '0.5s rest — aggressive'),
    (1.0,  '1.0s rest — fast'),
    (1.5,  '1.5s rest — balanced'),
    (2.5,  '2.5s rest — motor simulation'),
    (4.0,  '4.0s rest — conservative'),
]

for rest, label in test_configs:
    result = run_benchmark(rest, duration=120, label=label)
    results.append(result)
    print(f'\n  Cooling 60s before next test...')
    time.sleep(60)

# Final comparison
print(f'\n{"="*65}')
print(f'  SWEET SPOT ANALYSIS')
print(f'{"="*65}')
print(f'  {"Rest":>6} {"Cycles/min":>11} {"Avg°C":>7} {"Max°C":>7} {"tok/s":>7} {"Throttles":>10}')
print(f'  {"-"*63}')

best_score = 0
best_config = None

for r in results:
    # Score = cycles per minute × speed / temperature penalty
    temp_penalty = max(1, (r['avg_temp'] - 50) / 10)
    score = (r['cycles_min'] * r['avg_tok_s']) / temp_penalty
    
    marker = ''
    if score > best_score:
        best_score = score
        best_config = r
        marker = ' ← SWEET SPOT'
    
    print(f'  {r["rest"]:>5}s {r["cycles_min"]:>10}/min {r["avg_temp"]:>6}°C {r["max_temp"]:>6}°C {r["avg_tok_s"]:>6} {r["throttles"]:>9}{marker}')

if best_config:
    print(f'\n  RECOMMENDATION:')
    print(f'  Cycle rest: {best_config["rest"]}s')
    print(f'  Expected: {best_config["cycles_min"]} cycles/min')
    print(f'  Sustained temp: {best_config["avg_temp"]}°C')
    print(f'  Update CYCLE_MOVE_TIME = {best_config["rest"]} in main.py')
