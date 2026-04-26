import requests, json, time

URL = 'http://127.0.0.1:8080/completion'

# Short prompt — matches training format exactly
LORA_SYS = """You are a home robot command parser. Convert this voice command to JSON.
Output ONLY a JSON array. No explanation. No markdown.

ACTIONS:
{"action":"find_person","target":"NAME","message":"MSG or empty"}
{"action":"navigate_to","target":"ROOM"}
{"action":"say","message":"EXACT TEXT"}
{"action":"patrol","target":"all"}
{"action":"find_object","target":"OBJECT","room":"ROOM or empty"}
{"action":"come_back"}

ROOMS: hallway, kitchen, living_room, bedroom, bathroom"""

TESTS = [
    ("Go to the kitchen",                            "navigate_to",  "kitchen",   None,      1),
    ("Come back",                                    "come_back",    None,        None,      1),
    ("Find Abdel",                                   "find_person",  "abdel",     None,      1),
    ("Say hello",                                    "say",          None,        "hello",   1),
    ("Patrol the apartment",                         "patrol",       None,        None,      1),
    ("Find my phone",                                "find_object",  "phone",     None,      1),
    ("Go to the bedroom",                            "navigate_to",  "bedroom",   None,      1),
    ("Where is Chiara?",                             "find_person",  "chiara",    None,      1),
    ("Can you go to the kitchen please",             "navigate_to",  "kitchen",   None,      1),
    ("Hey find Abdel",                               "find_person",  "abdel",     None,      1),
    ("Please come back here",                        "come_back",    None,        None,      1),
    ("Look around the apartment",                    "patrol",       None,        None,      1),
    ("Where are my glasses?",                        "find_object",  "glasses",   None,      1),
    ("Move to the bathroom",                         "navigate_to",  "bathroom",  None,      1),
    ("Go back",                                      "come_back",    None,        None,      1),
    ("Tell Chiara dinner is getting cold",           "find_person",  "chiara",    "cold",    1),
    ("Find Abdel and say his coffee is ready",       "find_person",  "abdel",     "coffee",  1),
    ("Tell Chiara someone is at the door",           "find_person",  "chiara",    "door",    1),
    ("Find Abdel and tell him to come eat",          "find_person",  "abdel",     "eat",     1),
    ("Find Chiara and tell her dinner is ready",     "find_person",  "chiara",    "dinner",  1),
    ("could you head to the living room",            "navigate_to",  "living_room",None,     1),
    ("quickly go find chiara",                       "find_person",  "chiara",    None,      1),
    ("search all the rooms",                         "patrol",       None,        None,      1),
    ("locate my wallet in the bedroom",              "find_object",  "wallet",    None,      1),
    ("hurry back",                                   "come_back",    None,        None,      1),
    ("say good night to everyone",                   "say",          None,        "night",   1),
    ("find chiara and let her know dinner is served","find_person",  "chiara",    "dinner",  1),
    ("check if abdel is in the bedroom",             "find_person",  "abdel",     None,      1),
    ("go to kitchen and say hello",                  "navigate_to",  "kitchen",   None,      2),
    ("go to the bedroom and say good night",         "navigate_to",  "bedroom",   None,      2),
]

def parse(text):
    prompt = (
        '<start_of_turn>user\n' + LORA_SYS +
        '\nCommand: ' + text +
        '<end_of_turn>\n<start_of_turn>model\n'
    )
    try:
        r = requests.post(URL, json={
            'prompt': prompt,
            'n_predict': 200,
            'temperature': 0.05,
            'stop': ['<end_of_turn>', '\n\n']
        }, timeout=20)
        raw = r.json()['content']
        s = raw.find('[')
        e = raw.rfind(']') + 1
        if s >= 0 and e > 0:
            result = json.loads(raw[s:e])
            return result if isinstance(result, list) else [result]
    except:
        pass
    return []

def score(actions, exp_action, exp_target, exp_msg, min_actions):
    if min_actions == 0:
        return len(actions) == 0, "expected empty"
    if not actions:
        return False, "no output"
    if len(actions) < min_actions:
        return False, f"need {min_actions} got {len(actions)}"
    a = actions[0]
    action = a.get('action', '').lower()
    target = (a.get('target', '') or '').lower()
    message = (a.get('message', '') or '').lower()
    if exp_action and action != exp_action:
        return False, f"action={action} expected={exp_action}"
    if exp_target and exp_target.lower() not in target:
        return False, f"target missing '{exp_target}'"
    if exp_msg and exp_msg.lower() not in message:
        return False, f"message missing '{exp_msg}'"
    return True, "ok"

print('='*55)
print('  LORA ADAPTER BENCHMARK — Training prompt format')
print(f'  {len(TESTS)} tests')
print('='*55)
input('\nPress ENTER when server ready...\n')

passed = 0
for i, (cmd, exp_action, exp_target, exp_msg, min_act) in enumerate(TESTS, 1):
    actions = parse(cmd)
    ok, reason = score(actions, exp_action, exp_target, exp_msg, min_act)
    if ok: passed += 1
    status = '✅' if ok else '❌'
    print(f'\n[{i:2d}] {status} "{cmd}"')
    if actions:
        print(f'      → {json.dumps(actions[0])}')
        if len(actions) > 1:
            print(f'      → {json.dumps(actions[1])}')
    else:
        print(f'      → []')
    if not ok:
        print(f'      ✗ {reason}')
    time.sleep(0.3)

pct = round(passed/len(TESTS)*100)
print(f'\n{"="*55}')
print(f'  RESULT: {passed}/{len(TESTS)} ({pct}%)')
if pct >= 90:
    print(f'  ✅ ADAPTER WORKS — use this prompt in production')
elif pct >= 80:
    print(f'  ⚠️  DECENT — adapter partially helps')
else:
    print(f'  ❌ ADAPTER NOT HELPING')
print('='*55)
