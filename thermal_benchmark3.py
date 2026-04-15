import os, time, requests, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# ── Sensors ───────────────────────────────────────────
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

# ── Real YOLO inference ───────────────────────────────
MODEL_PATH  = '/data/data/com.termux/files/home/robot/yolo11m.onnx'
TEST_PHOTO  = '/data/data/com.termux/files/home/robot/test_photos/scene_test.jpg'
_yolo_sess  = None

def get_yolo():
    global _yolo_sess
    if _yolo_sess is None:
        print('  Loading YOLO11m...')
        _yolo_sess = ort.InferenceSession(MODEL_PATH)
        print('  YOLO ready')
    return _yolo_sess

def run_yolo():
    """Real YOLO inference on test photo."""
    t0 = time.time()
    session = get_yolo()
    img = Image.open(TEST_PHOTO).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    session.run(None, {session.get_inputs()[0].name: arr})
    return round(time.time() - t0, 3)

# ── LLM call ──────────────────────────────────────────
def llm_call():
    t0 = time.time()
    try:
        resp = requests.post('http://127.0.0.1:8080/completion', json={
            'prompt': '<start_of_turn>user\nHallway clear. Person center close. Mission find Chiara. Last: FWD,FWD.<end_of_turn>\n<start_of_turn>model\n',
            'n_predict': 20,
            'temperature': 0.1,
            'stop': ['<end_of_turn>']
        }, timeout=30)
        tok_s = resp.json()['timings']['predicted_per_second']
        return round(time.time()-t0, 2), round(tok_s, 1)
    except:
        return 0, 0

# ── Warmup ────────────────────────────────────────────
def warmup():
    print('\n  Loading YOLO and warming up LLM...')
    get_yolo()
    run_yolo()  # warm YOLO cache
    for i in range(3):
        llm_call()
        print(f'  Warmup {i+1}/3 | temp: {get_temp()}°C')
    print('  Waiting for temp to stabilize...')
    for _ in range(40):
        if get_temp() < 55:
            break
        time.sleep(3)
    print(f'  Ready at: {get_temp()}°C')

# ── Realistic cycle ───────────────────────────────────
def run_sustained(rest_s, cycles=25, label=''):
    """
    Realistic robot cycle:
      1. YOLO inference (~0.7s) ← real heat
      2. Gemma LLM call (~4s)   ← real heat  
      3. Rest (motor time)       ← cooling
    """
    print(f'\n{"="*55}')
    print(f'  REALISTIC TEST: {label}')
    print(f'  YOLO + Gemma + {rest_s}s motor rest | {cycles} cycles')
    print(f'{"="*55}')

    temps, tok_speeds, yolo_times, llm_times = [], [], [], []
    throttle_events = 0
    emergency = False

    for i in range(1, cycles+1):
        temp = get_temp()
        batt = get_batt()
        temps.append(temp)

        # Step 1: YOLO (real inference)
        yolo_t = run_yolo()
        yolo_times.append(yolo_t)

        # Step 2: Gemma LLM
        llm_t, tok_s = llm_call()
        if tok_s > 0:
            tok_speeds.append(tok_s)
            llm_times.append(llm_t)

        # Throttle detection
        if len(tok_speeds) > 5:
            recent = sum(tok_speeds[-3:]) / 3
            early  = sum(tok_speeds[:3]) / 3
            if recent < early * 0.75:
                throttle_events += 1

        cycle_t = round(yolo_t + llm_t, 2)
        status = '🔴' if temp > 85 else '🟡' if temp > 70 else '🟢'
        print(f'  [{i:2d}] {status}{temp}°C batt:{batt}°C | {tok_s}tok/s | yolo:{yolo_t}s llm:{llm_t}s total:{cycle_t}s')

        # Safety
        if batt > 40:
            print(f'  🔴 BATTERY: {batt}°C')
            emergency = True
            break
        if temp > 93:
            print(f'  🔴 CPU: {temp}°C')
            emergency = True
            break

        # Step 3: Motor execution / cooling
        time.sleep(rest_s)

    # Stats
    avg_temp   = round(sum(temps)/len(temps)) if temps else 0
    max_temp   = max(temps) if temps else 0
    avg_tok    = round(sum(tok_speeds)/len(tok_speeds), 1) if tok_speeds else 0
    min_tok    = round(min(tok_speeds), 1) if tok_speeds else 0
    avg_yolo   = round(sum(yolo_times)/len(yolo_times), 3) if yolo_times else 0
    avg_llm    = round(sum(llm_times)/len(llm_times), 2) if llm_times else 0
    avg_cycle  = round(avg_yolo + avg_llm + rest_s, 2)
    cyc_min    = round(60 / avg_cycle, 1) if avg_cycle > 0 else 0

    print(f'\n  SUMMARY:')
    print(f'  Completed: {len(temps)}/{cycles}')
    print(f'  Avg temp: {avg_temp}°C | Max: {max_temp}°C')
    print(f'  Avg tok/s: {avg_tok} | Min: {min_tok}')
    print(f'  Avg YOLO: {avg_yolo}s | Avg LLM: {avg_llm}s')
    print(f'  Avg full cycle: {avg_cycle}s → {cyc_min} cycles/min')
    print(f'  Throttles: {throttle_events} | Emergency: {emergency}')

    return {
        'label': label, 'rest': rest_s,
        'cycles': len(temps),
        'avg_temp': avg_temp, 'max_temp': max_temp,
        'avg_tok': avg_tok, 'min_tok': min_tok,
        'avg_yolo': avg_yolo, 'avg_llm': avg_llm,
        'avg_cycle': avg_cycle, 'cyc_min': cyc_min,
        'throttles': throttle_events, 'emergency': emergency
    }

# ── Main ──────────────────────────────────────────────
print('='*55)
print('  REALISTIC ROBOT THERMAL BENCHMARK')
print('  YOLO11m + Gemma E2B + motor rest')
print('  Fine-tuning: 1.0s to 2.0s range')
print('='*55)

input('\nPress ENTER when Gemma server is ready...')

warmup()

# Fine-grained tests between 1.0 and 2.0
configs = [
    (1.0, 25, '1.0s rest'),
    (1.3, 25, '1.3s rest'),
    (1.5, 25, '1.5s rest'),
    (1.7, 25, '1.7s rest'),
    (2.0, 25, '2.0s rest'),
]

results = []
for rest, cycles, label in configs:
    r = run_sustained(rest, cycles, label)
    results.append(r)
    if r['emergency']:
        print('  Too hot — stopping remaining tests')
        break
    print(f'\n  Cooling 90s...')
    time.sleep(90)

# Final table
print(f'\n{"="*75}')
print(f'  REALISTIC BENCHMARK RESULTS — YOLO + GEMMA')
print(f'{"="*75}')
print(f'  {"Rest":>6} {"Done":>5} {"Avg°C":>6} {"Max°C":>6} {"tok/s":>6} {"YOLO":>6} {"LLM":>6} {"Cycle":>7} {"cyc/min":>8}')
print(f'  {"-"*73}')

best = None
best_score = 0
for r in results:
    if not r['emergency']:
        temp_penalty = max(1, (r['avg_temp'] - 55) / 5)
        score = (r['cyc_min'] * r['avg_tok']) / temp_penalty
        if score > best_score:
            best_score = score
            best = r
    marker = ' STOPPED' if r['emergency'] else ''
    print(f'  {r["rest"]:>5}s {r["cycles"]:>5} {r["avg_temp"]:>5}°C {r["max_temp"]:>5}°C {r["avg_tok"]:>5} {r["avg_yolo"]:>5}s {r["avg_llm"]:>5}s {r["avg_cycle"]:>6}s {r["cyc_min"]:>7}/min{marker}')

if best:
    print(f'\n  ✅ SWEET SPOT: {best["rest"]}s motor rest')
    print(f'  Full cycle: {best["avg_cycle"]}s')
    print(f'  Throughput: {best["cyc_min"]} cycles/min')
    print(f'  Sustained temp: {best["avg_temp"]}°C')
    print(f'  YOLO: {best["avg_yolo"]}s + Gemma: {best["avg_llm"]}s + rest: {best["rest"]}s')
    print(f'\n  Update main.py:')
    print(f'  CYCLE_MOVE_TIME = {best["rest"]}')
