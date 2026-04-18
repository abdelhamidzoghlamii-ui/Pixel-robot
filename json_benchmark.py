import requests, json, time, sys
sys.path.insert(0, '/data/data/com.termux/files/home/robot')

URL = 'http://127.0.0.1:8080/completion'

PARSE_SYS = """You are a home robot command parser.
Convert voice commands into a JSON array of actions.
Output ONLY a JSON array. No explanation. No markdown.

ACTIONS:
{"action":"find_person","target":"NAME","message":"MSG or empty"}
{"action":"navigate_to","target":"ROOM"}
{"action":"say","message":"EXACT TEXT"}
{"action":"patrol","target":"all"}
{"action":"find_object","target":"OBJECT","room":"ROOM or empty"}
{"action":"come_back"}

ROOMS: hallway, kitchen, living_room, bedroom, bathroom

RULES:
- "Tell X that Y" or "Find X and tell Y" = find_person with message
- "Go to X" or "Move to X" = navigate_to
- "Say X" = say, message must be EXACT words X
- "Patrol" or "Look around" or "Check rooms" = patrol
- "Come back" or "Return" or "Go back" = come_back
- "Find my X" where X is object = find_object
- "Where is X" person = find_person no message
- "Where are my X" = find_object
- Unknown command = []

EXAMPLES:
Input: "Find Chiara and tell her dinner is ready"
Output: [{"action":"find_person","target":"Chiara","message":"dinner is ready"}]

Input: "Tell Abdel his coffee is cold"
Output: [{"action":"find_person","target":"Abdel","message":"your coffee is cold"}]

Input: "Where is Chiara?"
Output: [{"action":"find_person","target":"Chiara","message":""}]

Input: "Find Abdel"
Output: [{"action":"find_person","target":"Abdel","message":""}]

Input: "Go to the kitchen"
Output: [{"action":"navigate_to","target":"kitchen"}]

Input: "Go to the bedroom"
Output: [{"action":"navigate_to","target":"bedroom"}]

Input: "Go to the living room"
Output: [{"action":"navigate_to","target":"living_room"}]

Input: "Go to the hallway"
Output: [{"action":"navigate_to","target":"hallway"}]

Input: "Say goodbye"
Output: [{"action":"say","message":"goodbye"}]

Input: "Say good morning"
Output: [{"action":"say","message":"good morning"}]

Input: "Say hello"
Output: [{"action":"say","message":"hello"}]

Input: "Patrol the apartment"
Output: [{"action":"patrol","target":"all"}]

Input: "Look around the apartment"
Output: [{"action":"patrol","target":"all"}]

Input: "Check all the rooms"
Output: [{"action":"patrol","target":"all"}]

Input: "Find my keys in the bedroom"
Output: [{"action":"find_object","target":"keys","room":"bedroom"}]

Input: "Find my phone in the living room"
Output: [{"action":"find_object","target":"phone","room":"living_room"}]

Input: "Find my glasses"
Output: [{"action":"find_object","target":"glasses","room":""}]

Input: "Go to the living room and say hello"
Output: [{"action":"navigate_to","target":"living_room"},{"action":"say","message":"hello"}]

Input: "Come back"
Output: [{"action":"come_back"}]

Input: "Return to me"
Output: [{"action":"come_back"}]

Input: "Go back"
Output: [{"action":"come_back"}]
"""

TESTS = [
    ("Find Chiara and tell her dinner is ready",  "find_person",  "chiara",       "dinner",   1),
    ("Tell Abdel his coffee is getting cold",     "find_person",  "abdel",        "coffee",   1),
    ("Where is Chiara?",                          "find_person",  "chiara",       None,       1),
    ("Find Abdel",                                "find_person",  "abdel",        None,       1),
    ("Tell Chiara to come to dinner",             "find_person",  "chiara",       "dinner",   1),
    ("Find Chiara and say the food is ready",     "find_person",  "chiara",       "food",     1),
    ("Go to the kitchen",                         "navigate_to",  "kitchen",      None,       1),
    ("Go to the bedroom",                         "navigate_to",  "bedroom",      None,       1),
    ("Go to the hallway",                         "navigate_to",  "hallway",      None,       1),
    ("Go to the bathroom",                        "navigate_to",  "bathroom",     None,       1),
    ("Go to the living room",                     "navigate_to",  "living_room",  None,       1),
    ("Move to the kitchen",                       "navigate_to",  "kitchen",      None,       1),
    ("Say goodbye",                               "say",          None,           "goodbye",  1),
    ("Say good morning to everyone",              "say",          None,           "morning",  1),
    ("Say hello",                                 "say",          None,           "hello",    1),
    ("Say dinner is ready",                       "say",          None,           "dinner",   1),
    ("Announce that lunch is served",             "say",          None,           None,       1),
    ("Come back",                                 "come_back",    None,           None,       1),
    ("Return to me",                              "come_back",    None,           None,       1),
    ("Go back",                                   "come_back",    None,           None,       1),
    ("Come back here",                            "come_back",    None,           None,       1),
    ("Patrol the apartment",                      "patrol",       None,           None,       1),
    ("Look around the apartment",                 "patrol",       None,           None,       1),
    ("Check all the rooms",                       "patrol",       None,           None,       1),
    ("Find my keys in the bedroom",               "find_object",  "keys",         None,       1),
    ("Find my phone in the living room",          "find_object",  "phone",        None,       1),
    ("Find my glasses",                           "find_object",  "glasses",      None,       1),
    ("Where are my keys?",                        "find_object",  "keys",         None,       1),
    ("Go to the kitchen and say hello",           "navigate_to",  "kitchen",      None,       2),
    ("Go to the living room and say hello",       "navigate_to",  "living_room",  None,       2),
]

def parse(text):
    prompt = (
        '<start_of_turn>user\n' + PARSE_SYS +
        '\nInput: "' + text + '"'
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
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
    except:
        pass
    return []

def score(actions, exp_action, exp_target, exp_msg, min_actions):
    if not actions:
        return False, "no output"
    if len(actions) < min_actions:
        return False, f"got {len(actions)} actions, need {min_actions}"
    a = actions[0]
    action = a.get('action', '').lower()
    target = (a.get('target', '') or '').lower()
    message = (a.get('message', '') or '').lower()
    if action != exp_action:
        return False, f"action={action} expected={exp_action}"
    if exp_target and exp_target.lower() not in target:
        return False, f"target='{target}' missing '{exp_target}'"
    if exp_msg and exp_msg.lower() not in message:
        return False, f"message='{message}' missing '{exp_msg}'"
    return True, "ok"

print('=' * 58)
print('  JSON COMMAND BENCHMARK — 30 TESTS')
print('  Target: 90%+ (27/30)')
print('=' * 58)
input('\nPress ENTER when server ready...\n')

passed = 0
categories = {}

for i, (cmd, exp_action, exp_target, exp_msg, min_act) in enumerate(TESTS, 1):
    actions = parse(cmd)
    ok, reason = score(actions, exp_action, exp_target, exp_msg, min_act)
    if ok:
        passed += 1
    status = '✅' if ok else '❌'
    categories[exp_action] = categories.get(exp_action, [0, 0])
    categories[exp_action][1] += 1
    if ok:
        categories[exp_action][0] += 1
    print(f'\n[{i:2d}] {status} "{cmd}"')
    if actions:
        print(f'     → {json.dumps(actions[0])}')
        if len(actions) > 1:
            print(f'     → {json.dumps(actions[1])}')
    else:
        print(f'     → NO OUTPUT')
    if not ok:
        print(f'     ✗ {reason}')
    time.sleep(0.3)

pct = round(passed / len(TESTS) * 100)
print(f'\n{"=" * 58}')
print(f'  RESULT: {passed}/{len(TESTS)} ({pct}%)')
print(f'{"=" * 58}')
print(f'\n  BY CATEGORY:')
for action, (p, t) in sorted(categories.items()):
    bar = '✅' if p == t else '⚠️ ' if p >= t*0.5 else '❌'
    print(f'  {bar} {action:<15} {p}/{t}')
if pct >= 90:
    print(f'\n  ✅ TARGET REACHED')
elif pct >= 80:
    print(f'\n  ⚠️  CLOSE')
else:
    print(f'\n  ❌ NEEDS WORK')
