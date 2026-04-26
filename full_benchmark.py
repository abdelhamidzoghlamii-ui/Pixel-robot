import requests, json, time, os

URL = 'http://127.0.0.1:8080/completion'

def get_temp():
    try:
        for zone in range(20):
            try:
                t = int(open(f'/sys/class/thermal/thermal_zone{zone}/temp').read().strip())
                if 30000 < t < 100000:
                    return t // 1000
            except:
                pass
    except:
        pass
    return 0

def call(prompt, max_tokens=50):
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            'prompt': prompt,
            'n_predict': max_tokens,
            'temperature': 0.1,
            'stop': ['<end_of_turn>']
        }, timeout=30)
        d = r.json()
        tok_s = d['timings']['predicted_per_second']
        elapsed = round(time.time()-t0, 2)
        return d['content'].strip(), tok_s, elapsed
    except:
        return '', 0, 0

# ── Prompts ──────────────────────────────────────────────────
NAV_SYS = """You are a robot navigation AI controlling a 4WD mecanum wheel robot.
Reply with ONE word: FORWARD, LEFT, RIGHT, BACK, or STOP.
Then one sentence explanation.

STRICT RULES:
- Person center + distance < 100cm → STOP
- Person center + distance > 100cm → FORWARD
- Person visible LEFT → LEFT
- Person visible RIGHT → RIGHT
- Obstacle center + distance < 45cm → BACK
- Obstacle LEFT → RIGHT
- Obstacle RIGHT → LEFT
- Room signature visible + mission matches → FORWARD
- Room signature + distance < 100cm → STOP
- Same direction 4+ times + empty scene → turn LEFT or RIGHT
- All 5 rooms visited + person not found → STOP"""

PARSE_SYS = """You are a home robot command parser.
Convert voice commands into a JSON array of actions.
Output ONLY a JSON array. No explanation. No markdown.

ACTIONS:
{"action":"find_person","target":"NAME","message":"MSG or empty"}
{"action":"navigate_to","target":"ROOM"}
{"action":"say","message":"TEXT"}
{"action":"patrol","target":"all"}
{"action":"find_object","target":"OBJECT","room":"ROOM or empty"}
{"action":"come_back"}

RULES:
- "Find X and tell Y" = find_person with message
- "Go to X" = navigate_to
- "Say X" = say
- "Patrol" = patrol
- "Come back" or "Go back" or "Return" = come_back
- "Find my X" = find_object"""

NAV_TESTS = [
    ("Mission: find Chiara. I see: person center. Distance: 60cm.", "STOP"),
    ("Mission: find Abdel. I see: person left. Distance: 150cm.", "LEFT"),
    ("Mission: find Chiara. I see: person right. Distance: 200cm.", "RIGHT"),
    ("Mission: find Abdel. I see: person center. Distance: 250cm.", "FORWARD"),
    ("Mission: navigate_to kitchen. I see: refrigerator center. Distance: 180cm.", "FORWARD"),
    ("Mission: navigate_to bedroom. I see: bed center. Distance: 80cm.", "STOP"),
    ("Mission: find Chiara. I see: obstacle center. Distance: 40cm.", "BACK"),
    ("Mission: navigate_to kitchen. I see: chair left. Distance: 30cm.", "RIGHT"),
    ("Mission: find Abdel. I see: obstacle right. Distance: 50cm.", "LEFT"),
    ("Mission: find Chiara. I see: empty hallway. Distance: 999cm. Last moves: FWD,FWD,FWD,FWD,FWD.", "LEFT|RIGHT"),
    ("Mission: patrol apartment. All rooms visited: hallway,kitchen,living_room,bedroom,bathroom.", "STOP"),
]

VOICE_TESTS = [
    ("Go to the kitchen", "navigate_to", "kitchen"),
    ("Come back", "come_back", None),
    ("Find Abdel", "find_person", "abdel"),
    ("Say hello", "say", None),
    ("Patrol the apartment", "patrol", None),
    ("Find my phone", "find_object", "phone"),
    ("Go to the bedroom", "navigate_to", "bedroom"),
    ("Where is Chiara?", "find_person", "chiara"),
    ("Go back", "come_back", None),
    ("Tell Chiara dinner is ready", "find_person", "chiara"),
    ("Find Abdel and say his coffee is ready", "find_person", "abdel"),
    ("Look around the apartment", "patrol", None),
    ("Move to the bathroom", "navigate_to", "bathroom"),
    ("Hurry back", "come_back", None),
    ("Find my glasses in the bedroom", "find_object", "glasses"),
]

def nav_prompt(ctx):
    return f'<start_of_turn>user\n{NAV_SYS}\n\n{ctx}<end_of_turn>\n<start_of_turn>model\n'

def voice_prompt(cmd):
    return f'<start_of_turn>user\n{PARSE_SYS}\nInput: "{cmd}"<end_of_turn>\n<start_of_turn>model\n'

print('='*60)
print('  FULL ROBOT BENCHMARK')
print('  Nav + Voice + Thermal')
print('='*60)
input('\nPress ENTER when server ready (4 threads, parallel 1)...\n')

# ── NAV BENCHMARK ────────────────────────────────────────────
print('\n' + '─'*60)
print('  NAV BENCHMARK (11 tests)')
print('─'*60)

nav_pass = 0
nav_speeds = []
nav_temps = []

for i, (ctx, expected) in enumerate(NAV_TESTS, 1):
    temp = get_temp()
    nav_temps.append(temp)
    reply, tok_s, elapsed = call(nav_prompt(ctx), max_tokens=40)
    direction = reply.split('\n')[0].strip().split('.')[0].split(' ')[0].upper()
    
    if '|' in expected:
        ok = direction in expected.split('|')
    else:
        ok = direction == expected
    
    if ok: nav_pass += 1
    nav_speeds.append(tok_s)
    
    status = '✅' if ok else '❌'
    print(f'[{i:2d}] {status} {temp}°C | {tok_s:.1f}tok/s | Expected:{expected} Got:{direction}')
    time.sleep(0.5)

nav_pct = round(nav_pass/len(NAV_TESTS)*100)
nav_avg_speed = round(sum(nav_speeds)/len(nav_speeds), 1)
nav_avg_temp = round(sum(nav_temps)/len(nav_temps))
nav_max_temp = max(nav_temps)

print(f'\n  NAV: {nav_pass}/{len(NAV_TESTS)} ({nav_pct}%)')
print(f'  Speed: {nav_avg_speed} tok/s avg')
print(f'  Temp: {nav_avg_temp}°C avg | {nav_max_temp}°C max')

# ── VOICE BENCHMARK ──────────────────────────────────────────
print('\n' + '─'*60)
print('  VOICE BENCHMARK (15 tests)')
print('─'*60)

voice_pass = 0
voice_speeds = []
voice_temps = []

for i, (cmd, exp_action, exp_target) in enumerate(VOICE_TESTS, 1):
    temp = get_temp()
    voice_temps.append(temp)
    reply, tok_s, elapsed = call(voice_prompt(cmd), max_tokens=100)
    
    ok = False
    try:
        s = reply.find('[')
        e = reply.rfind(']') + 1
        if s >= 0 and e > 0:
            actions = json.loads(reply[s:e])
            if actions:
                a = actions[0]
                action_ok = a.get('action','') == exp_action
                target_ok = True
                if exp_target:
                    target_ok = exp_target.lower() in str(a.get('target','')).lower()
                ok = action_ok and target_ok
    except:
        pass
    
    if ok: voice_pass += 1
    voice_speeds.append(tok_s)
    
    status = '✅' if ok else '❌'
    print(f'[{i:2d}] {status} {temp}°C | {tok_s:.1f}tok/s | "{cmd[:35]}"')
    time.sleep(0.5)

voice_pct = round(voice_pass/len(VOICE_TESTS)*100)
voice_avg_speed = round(sum(voice_speeds)/len(voice_speeds), 1)
voice_avg_temp = round(sum(voice_temps)/len(voice_temps))
voice_max_temp = max(voice_temps)

print(f'\n  VOICE: {voice_pass}/{len(VOICE_TESTS)} ({voice_pct}%)')
print(f'  Speed: {voice_avg_speed} tok/s avg')
print(f'  Temp: {voice_avg_temp}°C avg | {voice_max_temp}°C max')

# ── THERMAL SUMMARY ──────────────────────────────────────────
all_speeds = nav_speeds + voice_speeds
all_temps = nav_temps + voice_temps

print('\n' + '='*60)
print('  FULL BENCHMARK SUMMARY')
print('='*60)
print(f'  Nav:   {nav_pass}/{len(NAV_TESTS)} ({nav_pct}%) | {nav_avg_speed} tok/s | {nav_avg_temp}°C avg')
print(f'  Voice: {voice_pass}/{len(VOICE_TESTS)} ({voice_pct}%) | {voice_avg_speed} tok/s | {voice_avg_temp}°C avg')
print(f'\n  Overall speed: {round(sum(all_speeds)/len(all_speeds),1)} tok/s')
print(f'  Overall temp:  {round(sum(all_temps)/len(all_temps))}°C avg | {max(all_temps)}°C max')

total = nav_pass + voice_pass
total_tests = len(NAV_TESTS) + len(VOICE_TESTS)
pct = round(total/total_tests*100)
print(f'\n  TOTAL: {total}/{total_tests} ({pct}%)')

# Cycle recommendation
avg_s = round(sum(all_speeds)/len(all_speeds), 1)
if avg_s >= 10:
    rec = '1.3s'
elif avg_s >= 8:
    rec = '1.5s'
else:
    rec = '2.0s'
print(f'\n  Recommended CYCLE_MOVE_TIME: {rec}')
print(f'  (based on {avg_s} tok/s average)')
print('='*60)
