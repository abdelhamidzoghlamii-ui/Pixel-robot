#!/usr/bin/env python3
"""
Local AI diagnostic — compares mechanisms for getting a structured
action out of a local model, on synthetic text-only scenes.
No camera, no motors. Read-only against a running llama-server.

Usage:
  python3 ~/json_mechanism_test.py gemma    # port 8080, Gemma chat format
  python3 ~/json_mechanism_test.py qwen     # port 8081, ChatML format

Writes ~/json_mechanism_results_<model>.json so runs don't overwrite
each other, and prints a summary table.
"""
import json, time, statistics, sys
import requests

ACTIONS = ['FORWARD', 'LEFT', 'RIGHT', 'BACK', 'STOP', 'SPEAK']

# --- Model profiles: prompt wrapper + stop tokens differ per family ---
PROFILES = {
    'gemma': dict(
        port=8080,
        wrap=lambda sysmsg, ctx: (
            f"<start_of_turn>user\n{sysmsg}\n\n{ctx}<end_of_turn>\n"
            f"<start_of_turn>model\n"),
        stop=['<end_of_turn>'],
    ),
    'qwen': dict(
        port=8081,
        # Qwen3.5 = ChatML. System goes in its own turn.
        wrap=lambda sysmsg, ctx: (
            f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
            f"<|im_start|>user\n{ctx}<|im_end|>\n"
            f"<|im_start|>assistant\n"),
        stop=['<|im_end|>', '<|endoftext|>'],
    ),
}

# Verbatim from main.py's GEMMA_SYS rules block (checked against source).
RULES = """STRICT RULES — follow exactly:
- Person center + distance < 100cm -> STOP (reached person)
- Person center + distance > 100cm -> FORWARD (approach)
- Person visible LEFT -> LEFT (turn toward them)
- Person visible RIGHT -> RIGHT (turn toward them)
- Obstacle center + distance < 80cm -> BACK (reverse away)
- Obstacle LEFT -> RIGHT (avoid by going right)
- Obstacle RIGHT -> LEFT (avoid by going left)
- Refrigerator/sink visible + mission kitchen -> FORWARD
- Bed/wardrobe visible + mission bedroom -> FORWARD
- Toilet visible + mission bathroom -> FORWARD
- Couch/tv visible + mission living_room -> FORWARD
- Room signature visible + distance < 100cm -> STOP (arrived)
- Same direction 4+ times + empty scene -> turn LEFT or RIGHT
- All 5 rooms visited + person not found -> STOP (give up)
- Patrol + all 5 rooms visited -> STOP (complete)"""

SCENES = [
    dict(scene="person center", distance=60, mission="find_person", target="person",
         known_rooms=[], last_moves=["FORWARD"]*3, expected={"STOP"}),
    dict(scene="person center", distance=250, mission="find_person", target="person",
         known_rooms=[], last_moves=["FORWARD"]*2, expected={"FORWARD"}),
    dict(scene="person left", distance=180, mission="find_person", target="person",
         known_rooms=[], last_moves=["FORWARD"], expected={"LEFT"}),
    dict(scene="person right", distance=180, mission="find_person", target="person",
         known_rooms=[], last_moves=["FORWARD"], expected={"RIGHT"}),
    dict(scene="chair center", distance=40, mission="patrol", target="none",
         known_rooms=[], last_moves=["FORWARD"], expected={"BACK"}),
    dict(scene="chair left", distance=70, mission="patrol", target="none",
         known_rooms=[], last_moves=["FORWARD"], expected={"RIGHT"}),
    dict(scene="chair right", distance=70, mission="patrol", target="none",
         known_rooms=[], last_moves=["FORWARD"], expected={"LEFT"}),
    dict(scene="refrigerator center", distance=200, mission="kitchen", target="kitchen",
         known_rooms=[], last_moves=["FORWARD"], expected={"FORWARD"}),
    dict(scene="bed center", distance=200, mission="bedroom", target="bedroom",
         known_rooms=[], last_moves=["FORWARD"], expected={"FORWARD"}),
    dict(scene="toilet center", distance=200, mission="bathroom", target="bathroom",
         known_rooms=[], last_moves=["FORWARD"], expected={"FORWARD"}),
    dict(scene="couch center", distance=200, mission="living_room", target="living_room",
         known_rooms=[], last_moves=["FORWARD"], expected={"FORWARD"}),
    dict(scene="refrigerator center", distance=60, mission="kitchen", target="kitchen",
         known_rooms=[], last_moves=["FORWARD"], expected={"STOP"}),
    dict(scene="empty room, nothing detected", distance=300, mission="patrol", target="none",
         known_rooms=["kitchen"], last_moves=["FORWARD"]*4, expected={"LEFT", "RIGHT"}),
]

def build_context(s, scene_log):
    last5 = scene_log[-5:] if scene_log else ['none']
    return (
        f"Mission: {s['mission']}\n"
        f"Target: {s['target']}\n"
        f"Current scene: {s['scene']}\n"
        f"Last 5 scenes:\n" + '\n'.join(f'  - {x}' for x in last5) +
        f"\nLast moves: {', '.join(s['last_moves'][-5:])}\n"
        f"Known rooms: {s['known_rooms']}\n"
        f"Distance ahead: {s['distance']}cm"
    )

BASE_SYSMSG = ("You are a robot navigation AI controlling a 4WD mecanum wheel robot.\n"
               "Reply with ONE word: FORWARD, LEFT, RIGHT, BACK, or STOP.\n"
               "Then one sentence explanation.\n\n" + RULES)

SCHEMA_B = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "reason": {"type": "string"},
        "speak": {"type": ["string", "null"]}
    },
    "required": ["action", "reason", "speak"],
    "additionalProperties": False
}

GRAMMAR_A = (
    'root   ::= action "|" reason "|" speak\n'
    'action ::= "FORWARD" | "LEFT" | "RIGHT" | "BACK" | "STOP" | "SPEAK"\n'
    'reason ::= [^|\\n]+\n'
    'speak  ::= [^|\\n]*\n'
)

GRAMMAR_C = (
    'root   ::= "DECISION:" " "? action "|" reason "|" speak\n'
    'action ::= "FORWARD" | "LEFT" | "RIGHT" | "BACK" | "STOP" | "SPEAK"\n'
    'reason ::= [^|\\n]+\n'
    'speak  ::= [^|\\n]*\n'
)

B_SYSMSG = ("You are a robot navigation AI controlling a 4WD mecanum wheel robot.\n"
            "Respond with a single JSON object: "
            '{"action": <ACTION>, "reason": "<one short sentence>", "speak": <string or null>}\n'
            f"action must be exactly one of: {', '.join(ACTIONS)}\n\n" + RULES)

A_SYSMSG = ("You are a robot navigation AI controlling a 4WD mecanum wheel robot.\n"
            "Respond with exactly one line: ACTION|reason|speak\n"
            f"ACTION must be exactly one of: {', '.join(ACTIONS)}\n"
            "Leave speak empty if there is nothing to say.\n\n" + RULES)

C_SYSMSG = ("You are a robot navigation AI controlling a 4WD mecanum wheel robot.\n"
            "First, think through the situation in one short sentence.\n"
            "Then on a new line write: DECISION:ACTION|reason|speak\n"
            f"ACTION must be exactly one of: {', '.join(ACTIONS)}\n"
            "Leave speak empty if there is nothing to say.\n\n" + RULES)

D_SYSMSG = (BASE_SYSMSG +
            "\n\nFormat your final line exactly as: ACTION|reason|speak "
            "(ACTION is the one word above; reason is your one-sentence "
            "explanation; leave speak empty if there is nothing to say aloud).")

def baseline_parse(content):
    for w in content.upper().split():
        if w in ACTIONS:
            return w
    return None

def b_parse(content):
    try:
        d = json.loads(content.strip())
        a = d.get('action')
        return a if a in ACTIONS else None
    except Exception:
        return None

def a_parse(content):
    line = content.strip().splitlines()[0] if content.strip() else ''
    parts = line.split('|')
    if len(parts) == 3 and parts[0].strip() in ACTIONS:
        return parts[0].strip()
    return None

def c_parse(content):
    idx = content.find('DECISION:')
    if idx == -1:
        return None
    parts = content[idx + len('DECISION:'):].strip().split('|')
    if len(parts) == 3 and parts[0].strip() in ACTIONS:
        return parts[0].strip()
    return None

METHODS = {
    'baseline':       dict(sysmsg=BASE_SYSMSG, extra=None, n_predict=40, parse=baseline_parse),
    'B_json_schema':  dict(sysmsg=B_SYSMSG, extra={'json_schema': SCHEMA_B}, n_predict=60, parse=b_parse),
    'A_gbnf_pipe':    dict(sysmsg=A_SYSMSG, extra={'grammar': GRAMMAR_A}, n_predict=60, parse=a_parse),
    'C_lazy_trigger': dict(sysmsg=C_SYSMSG,
                           extra={'grammar': GRAMMAR_C, 'grammar_lazy': True,
                                  'grammar_triggers': [{'type': 1, 'value': 'DECISION:'}]},
                           n_predict=100, parse=c_parse),
    'D_grammar_only': dict(sysmsg=D_SYSMSG, extra={'grammar': GRAMMAR_A}, n_predict=60, parse=a_parse),
}

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'gemma'
    if model not in PROFILES:
        print(f"Unknown model '{model}'. Use one of: {', '.join(PROFILES)}")
        sys.exit(1)
    prof = PROFILES[model]
    url = f"http://127.0.0.1:{prof['port']}"

    try:
        requests.get(url + '/health', timeout=5)
    except Exception as e:
        print(f"[FAIL] Cannot reach {url} — is the server for '{model}' running? ({e})")
        sys.exit(1)

    print(f"Model profile: {model}  (port {prof['port']})\n")

    def call(prompt, extra, n_predict):
        payload = {'prompt': prompt, 'n_predict': n_predict,
                   'temperature': 0.1, 'stop': prof['stop']}
        if extra:
            payload.update(extra)
        t0 = time.time()
        try:
            r = requests.post(url + '/completion', json=payload, timeout=60)
            return r.json().get('content', ''), time.time() - t0, {}
        except Exception as e:
            return None, time.time() - t0, {'error': str(e)}

    scene_log = []
    results = {name: [] for name in METHODS}

    for s in SCENES:
        ctx = build_context(s, scene_log)
        scene_log.append(s['scene'])
        for name, m in METHODS.items():
            prompt = prof['wrap'](m['sysmsg'], ctx)
            content, elapsed, raw = call(prompt, m['extra'], m['n_predict'])
            action = m['parse'](content) if content is not None else None
            results[name].append({
                'scene': s['scene'], 'distance': s['distance'],
                'expected': sorted(s['expected']), 'raw_content': content,
                'action': action, 'valid': action is not None,
                'correct': action in s['expected'] if action else False,
                'latency_s': round(elapsed, 3), 'error': raw.get('error'),
            })
            print(f"  [{name}] {s['scene'][:24]:24s} action={str(action):8s} "
                  f"{'OK' if action in s['expected'] else 'X'} ({elapsed:.2f}s)")

    out = f"/data/data/com.termux/files/home/json_mechanism_results_{model}.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== SUMMARY ({model}) ===")
    print(f"{'method':16s} {'valid%':>7s} {'correct%':>9s} {'mean_s':>7s} {'median_s':>9s}")
    for name, rows in results.items():
        n = len(rows)
        v = sum(r['valid'] for r in rows)
        c = sum(r['correct'] for r in rows)
        lat = [r['latency_s'] for r in rows if r['error'] is None]
        mean_s = statistics.mean(lat) if lat else float('nan')
        med_s = statistics.median(lat) if lat else float('nan')
        print(f"{name:16s} {100*v/n:6.1f}% {100*c/n:8.1f}% {mean_s:7.2f} {med_s:9.2f}")

    print(f"\nFull results: {out}")

if __name__ == '__main__':
    main()
