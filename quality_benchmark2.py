import requests, json, base64, os, time, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

HOME = '/data/data/com.termux/files/home'
URL  = 'http://127.0.0.1:8080/completion'

def ask(prompt, image_path=None, max_tokens=150):
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
        prompt = prompt.replace(
            '<start_of_turn>user\n',
            '<start_of_turn>user\n[img-1]\n'
        )
        payload['prompt'] = prompt
    try:
        t0 = time.time()
        r = requests.post(URL, json=payload, timeout=60)
        elapsed = round(time.time()-t0, 2)
        return r.json()['content'].strip(), elapsed
    except Exception as e:
        return f'ERROR: {e}', 0

def gemma_prompt(text):
    return (
        f'<start_of_turn>user\n{text}'
        f'<end_of_turn>\n<start_of_turn>model\n'
    )

# ── Test photos ───────────────────────────────────────
PHOTO_DIR = HOME + '/robot/test_photos/'
photos = sorted([
    PHOTO_DIR + f for f in os.listdir(PHOTO_DIR)
    if f.endswith('.jpg')
]) if os.path.exists(PHOTO_DIR) else []

print('='*60)
print('  QUALITY BENCHMARK v2 — WHAT GEMMA ACTUALLY DOES')
print('='*60)
input('\nPress ENTER when server ready...')

passed = 0
total  = 0

# ── TEST 1: High-level navigation strategy ────────────
# Gemma decides WHERE to go, not how to avoid obstacles
print(f'\n{"─"*60}')
print('  TEST 1: HIGH-LEVEL NAVIGATION STRATEGY')
print(f'{"─"*60}')

SYS = (
    'You are a home robot navigation strategist. '
    'Python handles obstacle avoidance and close-range stopping. '
    'Your job is high-level strategy: which direction to explore. '
    'Reply with one word: FORWARD LEFT RIGHT BACK or STOP. '
    'Then one short reason.'
)

NAV_TESTS = [
    {
        'ctx': 'Mission: find Chiara. Explored: hallway, kitchen. '
               'Last 5 moves all FORWARD, no person found. '
               'Currently in living room.',
        'good': ['LEFT', 'RIGHT'],
        'desc': 'Explored forward, should now turn to explore'
    },
    {
        'ctx': 'Mission: find kitchen. I see refrigerator ahead. '
               'Distance 200cm. Last move: FORWARD.',
        'good': ['FORWARD'],
        'desc': 'Fridge = kitchen ahead, keep going'
    },
    {
        'ctx': 'Mission: find Chiara. Person visible to the LEFT. '
               'Distance 300cm. Last move: FORWARD.',
        'good': ['LEFT'],
        'desc': 'Person left, should turn left'
    },
    {
        'ctx': 'Mission: patrol apartment. '
               'Visited: hallway, kitchen, living room. '
               'Not yet visited: bedroom, bathroom.',
        'good': ['FORWARD', 'LEFT', 'RIGHT'],
        'desc': 'Should explore unvisited rooms'
    },
    {
        'ctx': 'Mission: find Abdel. '
               'Last seen: bedroom 10 minutes ago. '
               'Currently in hallway facing bedroom door.',
        'good': ['FORWARD'],
        'desc': 'Should go toward bedroom where Abdel was last seen'
    },
    {
        'ctx': 'Mission: find Chiara. '
               'Searched entire apartment twice, no person found. '
               'Battery getting low.',
        'good': ['STOP'],
        'desc': 'Should stop after exhaustive search'
    },
]

nav_pass = 0
for i, test in enumerate(NAV_TESTS, 1):
    prompt = gemma_prompt(f'{SYS}\n\nSituation: {test["ctx"]}')
    reply, elapsed = ask(prompt, max_tokens=60)
    first = reply.strip().split()[0].upper().rstrip('.,!')
    ok = first in test['good']
    if ok:
        nav_pass += 1
        passed += 1
    total += 1
    status = '✅' if ok else '❌'
    print(f'\n  [{i}] {status} {test["desc"]}')
    print(f'       Expected: {test["good"]} | Got: {first}')
    print(f'       Full: {reply[:120]}')
    print(f'       Time: {elapsed}s')

print(f'\n  Strategy: {nav_pass}/{len(NAV_TESTS)}')

# ── TEST 2: JSON voice command parsing ────────────────
print(f'\n{"─"*60}')
print('  TEST 2: JSON VOICE COMMAND PARSING')
print(f'{"─"*60}')

JSON_SYS = (
    'Parse this voice command into JSON. '
    'Reply ONLY with valid JSON.\n'
    'Format: {"action": "find_person|go_to_room|say|patrol|come_back", '
    '"target": "person name or room", '
    '"message": "message to deliver or empty string"}'
)

JSON_TESTS = [
    # find_person with message
    {'cmd': 'Find Chiara and tell her dinner is ready',
     'action': 'find_person', 'target': 'chiara', 'has_message': True},

    # navigate_to room
    {'cmd': 'Go to the kitchen',
     'action': 'navigate_to', 'target': 'kitchen', 'has_message': False},

    # find_person different phrasing
    {'cmd': 'Tell Abdel his coffee is getting cold',
     'action': 'find_person', 'target': 'abdel', 'has_message': True},

    # patrol
    {'cmd': 'Patrol the apartment',
     'action': 'patrol', 'has_message': False},

    # say
    {'cmd': 'Say good morning to everyone',
     'action': 'say', 'has_message': True},

    # find_object
    {'cmd': 'Find my keys in the bedroom',
     'action': 'find_object', 'target': 'keys', 'has_message': False},

    # come_back
    {'cmd': 'Come back',
     'action': 'come_back', 'has_message': False},

    # chain navigate + say
    {'cmd': 'Go to the living room and say hello',
     'action': 'navigate_to', 'target': 'living_room', 'has_message': False},

    # find person no message
    {'cmd': 'Where is Chiara?',
     'action': 'find_person', 'target': 'chiara', 'has_message': False},

    # navigate bedroom
    {'cmd': 'Go to the bedroom',
     'action': 'navigate_to', 'target': 'bedroom', 'has_message': False},
]

json_pass = 0
for i, test in enumerate(JSON_TESTS, 1):
    prompt = gemma_prompt(f'{JSON_SYS}\n\nCommand: "{test["cmd"]}"')
    reply, elapsed = ask(prompt, max_tokens=80)
    ok = False
    parsed = None
    try:
        s = reply.find('[')
        e = reply.rfind(']') + 1
        if s >= 0:
            actions = json.loads(reply[s:e])
            if actions:
                parsed = actions[0]
                # Check action type
                action_ok = parsed.get('action','').lower() == test['action']
                # Check target if expected
                target_ok = True
                if 'target' in test:
                    target_ok = test['target'] in parsed.get('target','').lower()
                # Check message if expected
                msg_ok = True
                if test.get('has_message'):
                    msg_ok = len(parsed.get('message','')) > 0
                ok = action_ok and target_ok and msg_ok
    except:
        pass
    if ok:
        json_pass += 1
        passed += 1
    total += 1
    status = '✅' if ok else '❌'
    print(f'\n  [{i}] {status} "{test["cmd"][:45]}"')
    if parsed:
        print(f'       {json.dumps(parsed)}')
    else:
        print(f'       Raw: {reply[:80]}')
    print(f'       Time: {elapsed}s')

print(f'\n  JSON: {json_pass}/{len(JSON_TESTS)}')

# ── TEST 3: Vision — room identification ──────────────
print(f'\n{"─"*60}')
print('  TEST 3: VISION — ROOM + SCENE UNDERSTANDING')
print(f'{"─"*60}')

if photos:
    # Use up to 3 photos
    test_photos = photos[:3]
    for i, photo in enumerate(test_photos, 1):
        print(f'\n  Photo {i}: {os.path.basename(photo)}')

        # Q1: Room ID
        prompt = gemma_prompt(
            'You are a robot trying to identify what room you are in. '
            'Look at this image carefully. '
            'What room is this? Name the room and list 3 objects you see.'
        )
        reply, elapsed = ask(prompt, image_path=photo, max_tokens=80)
        print(f'  Room ID ({elapsed}s):\n    {reply}')

        # Q2: Person description
        prompt = gemma_prompt(
            'Is there a person in this image? '
            'If yes: describe their appearance in detail '
            '(hair color, clothing, position, what they are doing). '
            'If no: say NO PERSON.'
        )
        reply, elapsed = ask(prompt, image_path=photo, max_tokens=100)
        print(f'  Person ({elapsed}s):\n    {reply}')

        # Q3: Navigation advice
        prompt = gemma_prompt(
            'You are a robot navigating this room. '
            'Based on what you see: '
            '1. What obstacles should you avoid? '
            '2. Where is safe to drive? '
            'Answer in 2 short sentences.'
        )
        reply, elapsed = ask(prompt, image_path=photo, max_tokens=100)
        print(f'  Nav advice ({elapsed}s):\n    {reply}')
else:
    print('  No photos found in test_photos/')

# ── TEST 4: Descriptiveness with your actual photos ───
print(f'\n{"─"*60}')
print('  TEST 4: DETAILED SCENE DESCRIPTION')
print(f'{"─"*60}')

if photos:
    photo = photos[0]
    print(f'  Photo: {os.path.basename(photo)}')

    prompts = [
        ('General', 'Describe everything you see in this image in detail.'),
        ('People',  'Describe any people you see: appearance, position, expression, clothing.'),
        ('Layout',  'Describe the room layout: furniture positions, doors, windows, obstacles.'),
        ('Robot',   'As a home robot, what is the most important thing to notice in this scene?'),
    ]

    for label, question in prompts:
        prompt = gemma_prompt(question)
        reply, elapsed = ask(prompt, image_path=photo, max_tokens=150)
        print(f'\n  [{label}] ({elapsed}s):')
        print(f'  {reply}')

# ── Final ─────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'  FINAL SCORE')
print(f'{"="*60}')
print(f'  Strategy:  {nav_pass}/{len(NAV_TESTS)}')
print(f'  JSON:      {json_pass}/{len(JSON_TESTS)}')
pct = round((nav_pass+json_pass)/(len(NAV_TESTS)+len(JSON_TESTS))*100)
print(f'  Score:     {pct}%')
if pct >= 80:
    print(f'  ✅ APPROVED')
elif pct >= 65:
    print(f'  ⚠️  MARGINAL')
else:
    print(f'  ❌ FAILS')
