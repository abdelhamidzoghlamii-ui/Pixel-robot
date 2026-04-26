"""
Standalone dual-rate navigation test.
Run this on the phone to test the new logic WITHOUT touching main.py.
Python handles all deterministic rules.
Gemma only handles: exploration direction + mission complete.
"""
import requests, json, time

URL = 'http://127.0.0.1:8080/completion'

# Gemma only needs to answer TWO questions:
# 1. Which direction to explore when stuck?
# 2. Should I give up on this mission?
GEMMA_EXPLORE_SYS = """You are a robot exploration AI.
The robot is stuck or has no useful objects in view.
Choose ONE direction to explore: LEFT, RIGHT, FORWARD, or STOP.

RULES:
- If same move repeated 4+ times → choose different direction
- If all 5 rooms visited and person not found → STOP
- If patrol and all 5 rooms visited → STOP
- Otherwise choose LEFT or RIGHT to explore new area
- Prefer direction not recently tried

Reply with ONE word only: LEFT, RIGHT, FORWARD, or STOP"""

# ── Python navigation rules ─────────────────────────────
ROOM_SIGNATURES = {
    'refrigerator': 'kitchen', 'oven': 'kitchen', 'sink': 'kitchen',
    'bed': 'bedroom', 'wardrobe': 'bedroom',
    'couch': 'living_room', 'tv': 'living_room', 'dining table': 'living_room',
    'toilet': 'bathroom',
}
ALL_ROOMS = {'hallway', 'kitchen', 'bedroom', 'living_room', 'bathroom'}

def python_navigate(scene_text, distance, last_moves, mission, target,
                    known_rooms, person_close_dist=100, obstacle_dist=25):
    """
    Pure Python navigation — fast, deterministic, no LLM.
    Returns (direction, reason, needs_gemma)
    needs_gemma=True means Python couldn't decide → ask Gemma
    """
    scene = scene_text.lower()
    objects = [o.strip() for o in scene.split(',')]

    # ── Safety: obstacle very close ──────────────────────
    if distance < 15:
        return 'BACK', 'obstacle critically close', False
    if distance < obstacle_dist:
        # Check which side
        if any('left' in o for o in objects if 'obstacle' in o or 'chair' in o):
            return 'RIGHT', 'obstacle left → go right', False
        if any('right' in o for o in objects if 'obstacle' in o or 'chair' in o):
            return 'LEFT', 'obstacle right → go left', False
        return 'BACK', f'obstacle at {distance}cm → reverse', False

    # ── Person detection ─────────────────────────────────
    if 'person' in scene:
        if 'person left' in scene:
            return 'LEFT', 'person visible left', False
        if 'person right' in scene:
            return 'RIGHT', 'person visible right', False
        # Person center
        if distance < person_close_dist:
            return 'STOP', f'person close at {distance}cm — reached!', False
        return 'FORWARD', f'person ahead at {distance}cm — approaching', False

    # ── Room signature detection ──────────────────────────
    for obj, room in ROOM_SIGNATURES.items():
        if obj in scene:
            # Found a room signature
            if mission == 'navigate_to' and target == room:
                if distance < 100:
                    return 'STOP', f'reached {room} — {obj} close', False
                return 'FORWARD', f'{obj} visible → heading to {room}', False
            elif mission == 'patrol':
                if room not in known_rooms:
                    return 'FORWARD', f'new room {room} visible — exploring', False

    # ── Obstacle on side (medium range) ──────────────────
    for obj in objects:
        if any(x in obj for x in ['chair', 'obstacle', 'table', 'couch']):
            if 'left' in obj and distance < 80:
                return 'RIGHT', f'obstacle left at {distance}cm → right', False
            if 'right' in obj and distance < 80:
                return 'LEFT', f'obstacle right at {distance}cm → left', False

    # ── Python can't decide → ask Gemma ──────────────────
    return None, 'no clear signal', True


def gemma_explore(mission, target, last_moves, known_rooms, scene, distance):
    """Ask Gemma ONLY for exploration strategy."""
    rooms_visited = list(known_rooms.keys()) if known_rooms else []
    all_visited = ALL_ROOMS.issubset(set(rooms_visited))
    repeated = len(last_moves) >= 4 and len(set(last_moves[-4:])) == 1

    context = (
        f"Mission: {mission} {target}\n"
        f"Scene: {scene}\n"
        f"Distance: {distance}cm\n"
        f"Last moves: {', '.join(last_moves[-5:])}\n"
        f"Rooms visited: {', '.join(rooms_visited) if rooms_visited else 'none'}\n"
        f"All rooms visited: {all_visited}\n"
        f"Stuck (same move 4x): {repeated}"
    )
    prompt = (
        '<start_of_turn>user\n' + GEMMA_EXPLORE_SYS +
        '\n\n' + context +
        '<end_of_turn>\n<start_of_turn>model\n'
    )
    try:
        r = requests.post(URL, json={
            'prompt': prompt,
            'n_predict': 10,
            'temperature': 0.1,
            'stop': ['<end_of_turn>', '\n']
        }, timeout=15)
        raw = r.json()['content'].strip().upper()
        for d in ['STOP', 'LEFT', 'RIGHT', 'FORWARD', 'BACK']:
            if d in raw:
                return d
    except:
        pass
    return 'LEFT'  # safe default


def navigate(scene_text, distance, last_moves, mission, target,
             known_rooms, use_gemma=True):
    """Full navigation decision."""
    direction, reason, needs_gemma = python_navigate(
        scene_text, distance, last_moves, mission, target, known_rooms
    )
    if needs_gemma and use_gemma:
        direction = gemma_explore(
            mission, target, last_moves, known_rooms, scene_text, distance
        )
        reason = f'Gemma exploration → {direction}'
    elif needs_gemma:
        direction = 'FORWARD'
        reason = 'fallback forward'
    return direction, reason


# ── Benchmark tests ──────────────────────────────────────
TESTS = [
    # (scene, dist, last_moves, mission, target, known_rooms, expected, desc)

    # EASY — person
    ("person center", 60, ["FWD","FWD"], "find", "Chiara", {}, "STOP", "person center close"),
    ("person left", 150, ["FWD"], "find", "Abdel", {}, "LEFT", "person left"),
    ("person right", 200, ["FWD"], "find", "Chiara", {}, "RIGHT", "person right"),
    ("person center", 250, ["FWD","FWD"], "find", "Abdel", {}, "FORWARD", "person center far"),

    # EASY — room
    ("refrigerator center", 180, ["FWD"], "navigate_to", "kitchen", {}, "FORWARD", "fridge heading kitchen"),
    ("bed center", 80, ["FWD","FWD"], "navigate_to", "bedroom", {}, "STOP", "bed close arrived bedroom"),
    ("toilet center", 75, ["FWD"], "navigate_to", "bathroom", {}, "STOP", "toilet close arrived bathroom"),
    ("couch center, tv left", 200, ["FWD"], "navigate_to", "living_room", {}, "FORWARD", "couch heading living room"),

    # MEDIUM — obstacles
    ("obstacle center", 40, ["FWD","FWD"], "find", "Chiara", {}, "BACK", "obstacle center close"),
    ("chair left", 30, ["FWD"], "navigate_to", "kitchen", {}, "RIGHT", "obstacle left → right"),
    ("obstacle right", 50, ["FWD"], "find", "Abdel", {}, "LEFT", "obstacle right → left"),

    # HARD — exploration (Gemma)
    ("empty hallway", 999, ["FWD","FWD","FWD","FWD","FWD"], "find", "Chiara",
     {"hallway": "10:00"}, None, "stuck 5x FWD → Gemma turn"),
    ("nothing detected", 999, ["FWD","FWD","FWD","FWD","FWD"], "find", "Abdel",
     {"hallway": "10:00"}, None, "nothing detected → Gemma turn"),

    # HARD — mission complete (Gemma)
    ("empty hallway", 999, ["FWD","LEFT","FWD","RIGHT","FWD"], "patrol", "apartment",
     {"hallway":"10:00","kitchen":"10:01","living_room":"10:02","bedroom":"10:03","bathroom":"10:04"},
     "STOP", "patrol all rooms → STOP"),
    ("empty room", 999, ["FWD","LEFT","RIGHT","FWD","LEFT"], "find", "Chiara",
     {"hallway":"10:00","kitchen":"10:01","living_room":"10:02","bedroom":"10:03","bathroom":"10:04"},
     "STOP", "all rooms searched → STOP"),
]

print('='*60)
print('  DUAL-RATE NAV BENCHMARK')
print('  Python handles deterministic | Gemma handles exploration')
print(f'  {len(TESTS)} tests')
print('='*60)
input('\nPress ENTER when server ready...\n')

passed = 0
cats = {'person':[], 'room':[], 'obstacle':[], 'exploration':[]}

for i, (scene, dist, moves, mission, target, rooms, expected, desc) in enumerate(TESTS, 1):
    direction, reason = navigate(scene, dist, moves, mission, target, rooms)
    
    if expected is None:
        # Gemma test — accept LEFT or RIGHT (both valid exploration)
        ok = direction in ['LEFT', 'RIGHT']
        expected_str = 'LEFT or RIGHT'
    else:
        ok = direction == expected
        expected_str = expected

    if ok: passed += 1
    status = '✅' if ok else '❌'

    if i <= 4: cats['person'].append(ok)
    elif i <= 8: cats['room'].append(ok)
    elif i <= 11: cats['obstacle'].append(ok)
    else: cats['exploration'].append(ok)

    print(f'\n[{i:2d}] {status} {desc}')
    print(f'      Expected: {expected_str} | Got: {direction}')
    if not ok:
        print(f'      Reason: {reason}')
    time.sleep(0.2)

pct = round(passed/len(TESTS)*100)
print(f'\n{"="*60}')
print(f'  RESULT: {passed}/{len(TESTS)} ({pct}%)')
print(f'\n  BY CATEGORY:')
for cat, results in cats.items():
    if results:
        p, t = sum(results), len(results)
        bar = '✅' if p==t else '⚠️ ' if p>=t*0.5 else '❌'
        print(f'  {bar} {cat:<12} {p}/{t}')
if pct >= 85:
    print(f'\n  ✅ READY — merge into main.py')
elif pct >= 70:
    print(f'\n  ⚠️  GOOD — minor tweaks needed')
else:
    print(f'\n  ❌ NEEDS WORK')
print('='*60)
