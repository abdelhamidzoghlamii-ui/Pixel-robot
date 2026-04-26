import requests, json, time, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

URL = 'http://127.0.0.1:8080/completion'

GEMMA_SYS = """You are a robot navigation AI controlling a 4WD mecanum wheel robot.
Reply with ONE word: FORWARD, LEFT, RIGHT, BACK, or STOP.
Then one sentence explanation.

STRICT RULES — follow exactly:
- Person center + distance < 100cm → STOP (reached person)
- Person center + distance > 100cm → FORWARD (approach)
- Person visible LEFT → LEFT (turn toward them)
- Person visible RIGHT → RIGHT (turn toward them)
- Obstacle center + distance < 80cm → BACK (reverse away)
- Obstacle LEFT → RIGHT (avoid by going right)
- Obstacle RIGHT → LEFT (avoid by going left)
- Refrigerator/sink visible + mission kitchen → FORWARD
- Bed/wardrobe visible + mission bedroom → FORWARD
- Toilet visible + mission bathroom → FORWARD
- Couch/tv visible + mission living_room → FORWARD
- Room signature visible + distance < 100cm → STOP (arrived)
- Same direction 4+ times + empty scene → turn LEFT or RIGHT
- All 5 rooms visited + person not found → STOP (give up)
- Patrol + all 5 rooms visited → STOP (complete)"""

TESTS = [
    # (input, expected, description)
    # EASY — person detection
    ("Mission: find Chiara. Cycle: 3. I see: person center. Distance: 60cm. Last moves: FWD, FWD. Rooms visited: hallway.",
     "STOP", "person center close"),
    ("Mission: find Abdel. Cycle: 5. I see: person left. Distance: 150cm. Last moves: FWD. Rooms visited: kitchen.",
     "LEFT", "person left"),
    ("Mission: find Chiara. Cycle: 4. I see: person right. Distance: 200cm. Last moves: FWD. Rooms visited: hallway.",
     "RIGHT", "person right"),
    ("Mission: find Abdel. Cycle: 6. I see: person center. Distance: 250cm. Last moves: FWD, FWD. Rooms visited: hallway.",
     "FORWARD", "person center far"),

    # EASY — room navigation
    ("Mission: navigate_to kitchen. Cycle: 3. I see: refrigerator center. Distance: 180cm. Last moves: FWD. Rooms visited: hallway.",
     "FORWARD", "fridge visible heading kitchen"),
    ("Mission: navigate_to bedroom. Cycle: 5. I see: bed center. Distance: 80cm. Last moves: FWD, FWD. Rooms visited: hallway.",
     "STOP", "bed close arrived bedroom"),
    ("Mission: navigate_to bathroom. Cycle: 4. I see: toilet center. Distance: 75cm. Last moves: FWD. Rooms visited: hallway.",
     "STOP", "toilet close arrived bathroom"),
    ("Mission: navigate_to living_room. Cycle: 6. I see: couch center, tv left. Distance: 200cm. Last moves: FWD. Rooms visited: hallway.",
     "FORWARD", "couch visible heading living room"),

    # MEDIUM — obstacle avoidance
    ("Mission: find Chiara. Cycle: 8. I see: obstacle center. Distance: 40cm. Last moves: FWD, FWD. Rooms visited: hallway.",
     "BACK", "obstacle center close"),
    ("Mission: navigate_to kitchen. Cycle: 5. I see: chair left. Distance: 30cm. Last moves: FWD. Rooms visited: hallway.",
     "RIGHT", "obstacle left avoid right"),
    ("Mission: find Abdel. Cycle: 7. I see: obstacle right. Distance: 50cm. Last moves: FWD. Rooms visited: bedroom.",
     "LEFT", "obstacle right avoid left"),

    # HARD — exploration/stuck
    ("Mission: find Chiara. Cycle: 15. I see: empty hallway. Distance: 999cm. Last moves: FWD, FWD, FWD, FWD, FWD. Rooms visited: hallway.",
     "LEFT", "stuck 5x FWD must turn"),
    ("Mission: find Abdel. Cycle: 18. I see: nothing detected. Distance: 999cm. Last moves: FWD, FWD, FWD, FWD, FWD. Rooms visited: hallway.",
     "RIGHT", "nothing detected 5x FWD must turn"),
    ("Mission: find Chiara. Cycle: 20. I see: clear path. Distance: 999cm. Last moves: LEFT, LEFT, LEFT, LEFT. Rooms visited: hallway, kitchen.",
     "FORWARD", "stuck turning must go forward"),

    # HARD — mission complete
    ("Mission: patrol apartment. Cycle: 25. I see: empty hallway. Distance: 999cm. Last moves: FWD, LEFT, FWD, RIGHT, FWD. Rooms visited: hallway, kitchen, living_room, bedroom, bathroom.",
     "STOP", "patrol all rooms visited"),
    ("Mission: find Chiara. Cycle: 30. I see: empty room. Distance: 999cm. Last moves: FWD, LEFT, RIGHT, FWD, LEFT. Rooms visited: hallway, kitchen, living_room, bedroom, bathroom.",
     "STOP", "all rooms searched give up"),
]

def parse(text):
    prompt = (
        '<start_of_turn>user\n' + GEMMA_SYS +
        '\n\n' + text +
        '<end_of_turn>\n<start_of_turn>model\n'
    )
    try:
        r = requests.post(URL, json={
            'prompt': prompt,
            'n_predict': 40,
            'temperature': 0.1,
            'stop': ['<end_of_turn>']
        }, timeout=20)
        raw = r.json()['content'].strip()
        direction = raw.split('\n')[0].strip().split('.')[0].split(' ')[0].upper()
        return direction, raw
    except:
        return "", ""

print('='*58)
print('  NAV BENCHMARK — BASE MODEL WITH NEW RULES PROMPT')
print(f'  {len(TESTS)} tests | Target: 75%+')
print('='*58)
input('\nPress ENTER when server ready...\n')

passed = 0
categories = {'person':[], 'room':[], 'obstacle':[], 'exploration':[], 'complete':[]}

for i, (inp, expected, desc) in enumerate(TESTS, 1):
    direction, raw = parse(inp)
    ok = direction == expected
    if ok: passed += 1

    # Track category
    if i <= 4: categories['person'].append(ok)
    elif i <= 8: categories['room'].append(ok)
    elif i <= 11: categories['obstacle'].append(ok)
    elif i <= 14: categories['exploration'].append(ok)
    else: categories['complete'].append(ok)

    status = '✅' if ok else '❌'
    print(f'\n[{i:2d}] {status} {desc}')
    print(f'      Expected: {expected} | Got: {direction}')
    if not ok:
        print(f'      Full: {raw[:70]}')
    time.sleep(0.3)

pct = round(passed/len(TESTS)*100)
print(f'\n{"="*58}')
print(f'  RESULT: {passed}/{len(TESTS)} ({pct}%)')
print(f'\n  BY CATEGORY:')
for cat, results in categories.items():
    p = sum(results)
    t = len(results)
    bar = '✅' if p==t else '⚠️ ' if p>=t*0.5 else '❌'
    print(f'  {bar} {cat:<12} {p}/{t}')
if pct >= 75:
    print(f'\n  ✅ GOOD ENOUGH FOR ROBOT')
elif pct >= 60:
    print(f'\n  ⚠️  ACCEPTABLE — room for improvement')
else:
    print(f'\n  ❌ NEEDS WORK')
print('='*58)
