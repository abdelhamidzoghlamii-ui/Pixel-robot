import requests, json, base64, os, time, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

HOME = '/data/data/com.termux/files/home'
URL  = 'http://127.0.0.1:8080/completion'

def ask(prompt, image_path=None, max_tokens=80):
    payload = {
        'prompt': prompt,
        'n_predict': max_tokens,
        'temperature': 0.1,
        'stop': ['<end_of_turn>']
    }
    if image_path and os.path.exists(image_path):
        from PIL import Image
        import io
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((320, 320))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload['image_data'] = [{'data': b64, 'id': 1}]
        prompt = prompt.replace('<start_of_turn>user\n',
                                '<start_of_turn>user\n[img-1]\n')
        payload['prompt'] = prompt
    try:
        t0 = time.time()
        r = requests.post(URL, json=payload, timeout=60)
        elapsed = round(time.time()-t0, 2)
        return r.json()['content'].strip(), elapsed
    except Exception as e:
        return f'ERROR: {e}', 0

def build_nav_prompt(context):
    sys = (
        'You are a robot navigation AI. '
        'Reply with ONE word only: FORWARD, LEFT, RIGHT, BACK, or STOP. '
        'Then one sentence explanation.'
    )
    return (
        f'<start_of_turn>user\n{sys}\n\n{context}'
        f'<end_of_turn>\n<start_of_turn>model\n'
    )

def build_json_prompt(command):
    sys = (
        'You are a robot command parser. '
        'Parse the voice command into JSON. '
        'Reply ONLY with valid JSON, no explanation.\n'
        'Format: {"action": "find_person|go_to_room|say|patrol", '
        '"target": "name or room", "message": "optional"}'
    )
    return (
        f'<start_of_turn>user\n{sys}\n\nCommand: "{command}"'
        f'<end_of_turn>\n<start_of_turn>model\n'
    )

def build_photo_prompt(question):
    return (
        f'<start_of_turn>user\n{question}'
        f'<end_of_turn>\n<start_of_turn>model\n'
    )

# ── Tests ─────────────────────────────────────────────

NAV_TESTS = [
    {
        'context': 'Mission: find Chiara. I see: person center close. Last moves: FWD, FWD. Distance: 80cm.',
        'expected': 'STOP',
        'reason': 'Person close → stop'
    },
    {
        'context': 'Mission: find kitchen. I see: chair left, couch right, TV center. Last moves: FWD. Distance: 200cm.',
        'expected': 'FORWARD',
        'reason': 'Path clear, living room, keep going'
    },
    {
        'context': 'Mission: find Chiara. I see: obstacle center. Last moves: FWD. Distance: 20cm.',
        'expected': ['LEFT', 'RIGHT'],
        'reason': 'Obstacle close → turn'
    },
    {
        'context': 'Mission: find kitchen. I see: refrigerator center. Last moves: LEFT, FWD. Distance: 150cm.',
        'expected': 'FORWARD',
        'reason': 'Fridge = kitchen ahead, move toward it'
    },
    {
        'context': 'Mission: find bedroom. I see: bed center. Last moves: FWD. Distance: 180cm.',
        'expected': 'STOP',
        'reason': 'Bed = bedroom = mission complete'
    },
    {
        'context': 'Mission: find Chiara. I see: empty hallway. Last moves: FWD,FWD,FWD,FWD,FWD. Distance: 999cm.',
        'expected': ['LEFT', 'RIGHT'],
        'reason': 'Nothing found after 5 forward moves, turn to explore'
    },
    {
        'context': 'Mission: find Chiara. I see: person left. Last moves: FWD. Distance: 300cm.',
        'expected': 'LEFT',
        'reason': 'Person is to the left, turn toward them'
    },
    {
        'context': 'Mission: patrol apartment. I see: clear path. Last moves: FWD. Distance: 400cm.',
        'expected': 'FORWARD',
        'reason': 'Patrolling, path clear, go forward'
    },
]

JSON_TESTS = [
    {
        'command': 'Go find Chiara and tell her dinner is ready',
        'expected_action': 'find_person',
        'expected_target': 'chiara',
    },
    {
        'command': 'Go to the kitchen',
        'expected_action': 'go_to_room',
        'expected_target': 'kitchen',
    },
    {
        'command': 'Say hello to everyone',
        'expected_action': 'say',
    },
    {
        'command': 'Patrol the apartment',
        'expected_action': 'patrol',
    },
    {
        'command': 'Find Abdel and tell him his coffee is getting cold',
        'expected_action': 'find_person',
        'expected_target': 'abdel',
    },
]

PHOTO_TESTS = [
    {
        'question': 'What room is this? Answer in one word.',
        'type': 'room_id'
    },
    {
        'question': 'Is there a person in this image? Answer YES or NO.',
        'type': 'person_detect'
    },
    {
        'question': 'Describe what you see in 2 sentences.',
        'type': 'description'
    },
]

# ── Run ───────────────────────────────────────────────
print('='*60)
print('  QUALITY BENCHMARK — GEMMA E2B')
print('  Navigation + JSON + Vision')
print('='*60)
input('\nPress ENTER when server is ready...')

total_tests = 0
passed = 0

# ── 1. Navigation reasoning ───────────────────────────
print(f'\n{"─"*60}')
print(f'  TEST 1: NAVIGATION REASONING ({len(NAV_TESTS)} scenarios)')
print(f'{"─"*60}')

nav_pass = 0
for i, test in enumerate(NAV_TESTS, 1):
    prompt = build_nav_prompt(test['context'])
    reply, elapsed = ask(prompt, max_tokens=40)
    first_word = reply.strip().split()[0].upper() if reply else ''

    expected = test['expected']
    if isinstance(expected, list):
        ok = first_word in expected
    else:
        ok = first_word == expected

    status = '✅' if ok else '❌'
    if ok:
        nav_pass += 1
        passed += 1
    total_tests += 1

    print(f'  [{i}] {status} {test["reason"]}')
    print(f'       Expected: {expected} | Got: {first_word} | {elapsed}s')
    if not ok:
        print(f'       Full reply: {reply[:100]}')

print(f'\n  Navigation: {nav_pass}/{len(NAV_TESTS)} correct')

# ── 2. JSON command parsing ───────────────────────────
print(f'\n{"─"*60}')
print(f'  TEST 2: JSON COMMAND PARSING ({len(JSON_TESTS)} commands)')
print(f'{"─"*60}')

json_pass = 0
for i, test in enumerate(JSON_TESTS, 1):
    prompt = build_json_prompt(test['command'])
    reply, elapsed = ask(prompt, max_tokens=60)

    # Try to parse JSON
    ok = False
    parsed = None
    try:
        s = reply.find('{')
        e = reply.rfind('}') + 1
        if s >= 0 and e > 0:
            parsed = json.loads(reply[s:e])
            action_ok = parsed.get('action','').lower() == test['expected_action']
            target_ok = True
            if 'expected_target' in test:
                target = parsed.get('target','').lower()
                target_ok = test['expected_target'] in target
            ok = action_ok and target_ok
    except:
        pass

    status = '✅' if ok else '❌'
    if ok:
        json_pass += 1
        passed += 1
    total_tests += 1

    print(f'  [{i}] {status} "{test["command"][:40]}"')
    if parsed:
        print(f'       Got: {json.dumps(parsed)} | {elapsed}s')
    else:
        print(f'       Raw: {reply[:80]} | {elapsed}s')

print(f'\n  JSON parsing: {json_pass}/{len(JSON_TESTS)} correct')

# ── 3. Vision / photo analysis ────────────────────────
print(f'\n{"─"*60}')
print(f'  TEST 3: PHOTO ANALYSIS')
print(f'{"─"*60}')

photo = HOME + '/robot/test_photos/scene_test.jpg'
if not os.path.exists(photo):
    photos = [f for f in os.listdir(HOME+'/robot/test_photos/') 
              if f.endswith('.jpg')]
    if photos:
        photo = HOME + '/robot/test_photos/' + photos[0]

if os.path.exists(photo):
    print(f'  Using: {photo}')
    for i, test in enumerate(PHOTO_TESTS, 1):
        prompt = build_photo_prompt(test['question'])
        reply, elapsed = ask(prompt, image_path=photo, max_tokens=80)
        print(f'\n  [{i}] Q: {test["question"]}')
        print(f'       A: {reply}')
        print(f'       Time: {elapsed}s')
        total_tests += 1
        passed += 1  # vision tests are qualitative, mark pass if responds
else:
    print('  No test photo found — skipping vision tests')

# ── Final score ───────────────────────────────────────
print(f'\n{"="*60}')
print(f'  QUALITY RESULTS')
print(f'{"="*60}')
print(f'  Navigation:  {nav_pass}/{len(NAV_TESTS)}')
print(f'  JSON:        {json_pass}/{len(JSON_TESTS)}')
print(f'  Vision:      qualitative (check above)')
print(f'\n  Overall score: {nav_pass+json_pass}/{len(NAV_TESTS)+len(JSON_TESTS)} reasoning tests')

pct = round((nav_pass+json_pass)/(len(NAV_TESTS)+len(JSON_TESTS))*100)
print(f'  Accuracy: {pct}%')

if pct >= 85:
    print(f'\n  ✅ Q4_K_M APPROVED for robot use')
    print(f'  Quality is sufficient — switch from Q8_0')
elif pct >= 70:
    print(f'\n  ⚠️  Q4_K_M MARGINAL — check failed cases')
    print(f'  Consider staying on Q8_0 for reliability')
else:
    print(f'\n  ❌ Q4_K_M FAILS quality bar')
    print(f'  Stay on Q8_0')
