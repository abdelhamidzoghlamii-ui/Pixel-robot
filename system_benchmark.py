import requests, time, subprocess, os

URL = 'http://127.0.0.1:8080/completion'
PROMPT = '<start_of_turn>user\nExplain in 3 sentences how a robot navigates a room.<end_of_turn>\n<start_of_turn>model\n'
RUNS = 10

def get_temp():
    for zone in range(20):
        try:
            t = int(open(f'/sys/class/thermal/thermal_zone{zone}/temp').read())
            if 30000 < t < 100000:
                return t // 1000
        except:
            pass
    return 0

def get_ram():
    try:
        with open('/proc/meminfo') as f:
            lines = {l.split(':')[0]: int(l.split()[1]) 
                     for l in f if ':' in l}
        total = lines['MemTotal'] // 1024
        avail = lines['MemAvailable'] // 1024
        used  = total - avail
        swap_used = (lines['SwapTotal'] - lines['SwapFree']) // 1024
        return total, used, avail, swap_used
    except:
        return 0,0,0,0

def call():
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            'prompt': PROMPT,
            'n_predict': 60,
            'temperature': 0.1,
            'stop': ['<end_of_turn>']
        }, timeout=30)
        d = r.json()
        tok_s = d['timings']['predicted_per_second']
        prompt_s = d['timings']['prompt_per_second']
        n_tok = d['timings']['predicted_n']
        elapsed = round(time.time()-t0, 2)
        return tok_s, prompt_s, n_tok, elapsed, True
    except:
        return 0, 0, 0, 0, False

# ── START ────────────────────────────────────────────
print('=' * 55)
print('  FULL SYSTEM BENCHMARK — Gemma 4 E2B Q4_K_M')
print('  Running', RUNS, 'inference cycles...')
print('=' * 55)

t_total, used, avail, swap = get_ram()
print(f'\n  RAM before: {used}MB used / {t_total}MB total')
print(f'  Available:  {avail}MB | Swap used: {swap}MB')
print(f'  Temp before: {get_temp()}°C\n')

speeds = []
prompt_speeds = []
temps = []
times = []
throttles = 0
failures = 0
prev_speed = None

for i in range(1, RUNS+1):
    temp = get_temp()
    temps.append(temp)
    tok_s, prompt_s, n_tok, elapsed, ok = call()

    if not ok:
        failures += 1
        print(f'  [{i:2d}] ❌ FAILED | {temp}°C')
        continue

    if prev_speed and tok_s < prev_speed * 0.85:
        throttles += 1
        flag = ' ⚠️ THROTTLE'
    else:
        flag = ''

    speeds.append(tok_s)
    prompt_speeds.append(prompt_s)
    times.append(elapsed)
    prev_speed = tok_s

    bar = '█' * int(tok_s / 1.5)
    print(f'  [{i:2d}] {tok_s:5.1f} tok/s | {temp}°C | {elapsed:.1f}s | {bar}{flag}')
    time.sleep(1)

# ── RAM AFTER ────────────────────────────────────────
t_total2, used2, avail2, swap2 = get_ram()
temp_final = get_temp()

# ── SUMMARY ──────────────────────────────────────────
avg_s   = round(sum(speeds)/len(speeds), 1) if speeds else 0
max_s   = round(max(speeds), 1) if speeds else 0
min_s   = round(min(speeds), 1) if speeds else 0
avg_t   = round(sum(temps)/len(temps)) if temps else 0
max_t   = max(temps) if temps else 0
avg_e   = round(sum(times)/len(times), 1) if times else 0
avg_ps  = round(sum(prompt_speeds)/len(prompt_speeds), 1) if prompt_speeds else 0

print('\n' + '=' * 55)
print('  BENCHMARK SUMMARY')
print('=' * 55)
print(f'  Model:         Gemma 4 E2B Q4_K_M')
print(f'  Config:        4 threads | parallel 1 | cache-ram 0')
print(f'  Runs:          {RUNS} total | {failures} failed | {len(speeds)} successful')
print(f'')
print(f'  SPEED:')
print(f'    Generation:  {avg_s} tok/s avg | {max_s} max | {min_s} min')
print(f'    Prompt eval: {avg_ps} tok/s avg')
print(f'    Response time: {avg_e}s avg per request')
print(f'')
print(f'  THERMAL:')
print(f'    Temp avg:    {avg_t}°C')
print(f'    Temp max:    {max_t}°C')
print(f'    Temp final:  {temp_final}°C')
print(f'    Throttles:   {throttles} detected')
print(f'')
print(f'  RAM:')
print(f'    Before:      {used}MB used | {avail}MB free')
print(f'    After:       {used2}MB used | {avail2}MB free')
print(f'    Swap used:   {swap2}MB')
print(f'')
if avg_s >= 10:
    rating = '✅ EXCELLENT — robot ready'
elif avg_s >= 8:
    rating = '✅ GOOD — robot ready'
elif avg_s >= 6:
    rating = '⚠️  ACCEPTABLE — robot works but slow'
else:
    rating = '❌ TOO SLOW — investigate'
print(f'  RATING:        {rating}')
print(f'  CYCLE TIME:    ~{round(30/avg_s, 1)}s per nav decision (30 tokens)')
print('=' * 55)
