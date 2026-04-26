import requests, json, time, subprocess, os

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
- "Say X" = say, message = EXACT words X
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

Input: "Say goodbye"
Output: [{"action":"say","message":"goodbye"}]

Input: "Say good morning"
Output: [{"action":"say","message":"good morning"}]

Input: "Patrol the apartment"
Output: [{"action":"patrol","target":"all"}]

Input: "Find my keys in the bedroom"
Output: [{"action":"find_object","target":"keys","room":"bedroom"}]

Input: "Come back"
Output: [{"action":"come_back"}]

Input: "Return to me"
Output: [{"action":"come_back"}]

Input: "Go back"
Output: [{"action":"come_back"}]

Input: "Go to the living room and say hello"
Output: [{"action":"navigate_to","target":"living_room"},{"action":"say","message":"hello"}]
"""

# (command, action, target, msg_contains, min_actions, difficulty)
TESTS = [
    # EASY — direct simple commands
    ("Go to the kitchen",                           "navigate_to",  "kitchen",      None,       1, "easy"),
    ("Come back",                                   "come_back",    None,           None,       1, "easy"),
    ("Find Abdel",                                  "find_person",  "abdel",        None,       1, "easy"),
    ("Say hello",                                   "say",          None,           "hello",    1, "easy"),
    ("Patrol the apartment",                        "patrol",       None,           None,       1, "easy"),
    ("Find my phone",                               "find_object",  "phone",        None,       1, "easy"),
    ("Go to the bedroom",                           "navigate_to",  "bedroom",      None,       1, "easy"),
    ("Where is Chiara?",                            "find_person",  "chiara",       None,       1, "easy"),

    # MEDIUM — varied phrasing
    ("Can you go to the kitchen please",            "navigate_to",  "kitchen",      None,       1, "medium"),
    ("Hey find Abdel",                              "find_person",  "abdel",        None,       1, "medium"),
    ("Please come back here",                       "come_back",    None,           None,       1, "medium"),
    ("Look around the apartment",                   "patrol",       None,           None,       1, "medium"),
    ("Where are my glasses?",                       "find_object",  "glasses",      None,       1, "medium"),
    ("Move to the bathroom",                        "navigate_to",  "bathroom",     None,       1, "medium"),
    ("Announce that lunch is ready",                "say",          None,           None,       1, "medium"),
    ("Go back to start",                            "come_back",    None,           None,       1, "medium"),

    # MEDIUM — with messages
    ("Tell Chiara dinner is getting cold",          "find_person",  "chiara",       "cold",     1, "medium"),
    ("Find Abdel and say his coffee is ready",      "find_person",  "abdel",        "coffee",   1, "medium"),
    ("Tell Chiara someone is at the door",          "find_person",  "chiara",       "door",     1, "medium"),
    ("Find Abdel and tell him to come eat",         "find_person",  "abdel",        "eat",      1, "medium"),

    # HARD — unusual phrasing
    ("could you head to the living room",           "navigate_to",  "living_room",  None,       1, "hard"),
    ("quickly go find chiara",                      "find_person",  "chiara",       None,       1, "hard"),
    ("search all the rooms",                        "patrol",       None,           None,       1, "hard"),
    ("locate my wallet in the bedroom",             "find_object",  "wallet",       None,       1, "hard"),
    ("hurry back",                                  "come_back",    None,           None,       1, "hard"),
    ("say good night to everyone",                  "say",          None,           "night",    1, "hard"),
    ("find chiara and let her know dinner is served","find_person", "chiara",       "dinner",   1, "hard"),
    ("check if abdel is in the bedroom",            "find_person",  "abdel",        None,       1, "hard"),

    # HARD — chained commands
    ("go to kitchen and say hello",                 "navigate_to",  "kitchen",      None,       2, "hard"),
    ("go to the bedroom and say good night",        "navigate_to",  "bedroom",      None,       2, "hard"),

    # VERY HARD — edge cases
    ("make me a coffee",                            None,           None,           None,       0, "edge"),
    ("kitchen",                                     "navigate_to",  "kitchen",      None,       1, "edge"),
    ("chiara",                                      "find_person",  "chiara",       None,       1, "edge"),
    ("find the find chiara",                        "find_person",  "chiara",       None,       1, "edge"),
    ("don't go to the kitchen",                     None,           None,           None,       0, "edge"),
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
            return result if isinstance(result, list) else [result]
    except:
        pass
    return []

def score(actions, exp_action, exp_target, exp_msg, min_actions):
    # Edge case: expect empty output
    if min_actions == 0:
        return len(actions) == 0, "expected empty"
    if not actions:
        return False, "no output"
    if len(actions) < min_actions:
        return False, f"need {min_actions} actions got {len(actions)}"
    a = actions[0]
    action = a.get('action', '').lower()
    target = (a.get('target', '') or '').lower()
    message = (a.get('message', '') or '').lower()
    if exp_action and action != exp_action:
        return False, f"action={action} expected={exp_action}"
    if exp_target and exp_target.lower() not in target:
        return False, f"target='{target}' missing '{exp_target}'"
    if exp_msg and exp_msg.lower() not in message:
        return False, f"message missing '{exp_msg}'"
    return True, "ok"

def run_benchmark(label):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    input('\nPress ENTER when server ready...\n')

    passed = 0
    by_diff = {}

    for i, (cmd, exp_action, exp_target, exp_msg, min_act, diff) in enumerate(TESTS, 1):
        actions = parse(cmd)
        ok, reason = score(actions, exp_action, exp_target, exp_msg, min_act)
        if ok: passed += 1

        if diff not in by_diff:
            by_diff[diff] = [0, 0]
        by_diff[diff][1] += 1
        if ok: by_diff[diff][0] += 1

        status = '✅' if ok else '❌'
        print(f'\n[{i:2d}] {status} [{diff.upper():<6}] "{cmd}"')
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
    print(f'\n{"="*60}')
    print(f'  RESULT: {passed}/{len(TESTS)} ({pct}%)')
    print(f'\n  BY DIFFICULTY:')
    for d in ['easy', 'medium', 'hard', 'edge']:
        if d in by_diff:
            p, t = by_diff[d]
            bar = '✅' if p == t else '⚠️ ' if p >= t*0.7 else '❌'
            print(f'  {bar} {d:<8} {p}/{t}')
    print(f'{"="*60}')
    return pct

print('VOICE COMMAND BENCHMARK — WITH vs WITHOUT LORA ADAPTER')
print('Run 1: Start server WITH voice adapter')
print('  python3 server_manager.py setup_voice_lora')
r1 = run_benchmark('RUN 1 — WITH LORA ADAPTER (setup_voice_lora)')

print('\nNow kill server and start WITHOUT adapter:')
print('  pkill -f llama-server && python3 server_manager.py setup_q4')
r2 = run_benchmark('RUN 2 — WITHOUT ADAPTER (setup_q4)')

print(f'\n{"="*60}')
print(f'  FINAL COMPARISON')
print(f'  With adapter:    {r1}%')
print(f'  Without adapter: {r2}%')
print(f'  Difference:      {r1-r2:+d}%')
if r1 > r2:
    print(f'  ✅ Adapter helps!')
elif r1 == r2:
    print(f'  ➡️  No difference')
else:
    print(f'  ❌ Adapter hurts!')
print(f'{"="*60}')
