import requests, time

URL = 'http://127.0.0.1:8080/completion'

SYS = """You are a robot navigation AI controlling a 4WD mecanum wheel robot. Reply with ONE word: FORWARD, LEFT, RIGHT, BACK, or STOP. Then one sentence explanation. STRICT RULES: Person center distance less than 100cm means STOP. Person center distance more than 100cm means FORWARD. Person visible LEFT means LEFT. Person visible RIGHT means RIGHT. Obstacle center distance less than 45cm means BACK. Obstacle LEFT means RIGHT. Obstacle RIGHT means LEFT. Refrigerator visible and mission kitchen means FORWARD. Bed visible and mission bedroom means FORWARD. Room signature visible distance less than 100cm means STOP. Same direction 4 times and empty scene means turn LEFT or RIGHT. All 5 rooms visited and person not found means STOP."""

# Simulated robot cycles — same system prompt, changing scene (like real operation)
SCENES = [
    "Mission: find Chiara. Cycle 1. I see: person center. Distance: 300cm. Last moves: FWD.",
    "Mission: find Chiara. Cycle 2. I see: person center. Distance: 200cm. Last moves: FWD, FWD.",
    "Mission: find Chiara. Cycle 3. I see: person center. Distance: 90cm. Last moves: FWD, FWD, FWD.",
    "Mission: find Chiara. Cycle 4. I see: obstacle left. Distance: 40cm. Last moves: STOP.",
    "Mission: find Chiara. Cycle 5. I see: empty hallway. Distance: 999cm. Last moves: RIGHT.",
    "Mission: find Chiara. Cycle 6. I see: refrigerator center. Distance: 150cm. Last moves: FWD.",
    "Mission: find Chiara. Cycle 7. I see: person right. Distance: 120cm. Last moves: FWD, RIGHT.",
    "Mission: find Chiara. Cycle 8. I see: person center. Distance: 70cm. Last moves: RIGHT.",
]

def call(scene):
    prompt = f"<start_of_turn>user\n{SYS}\n\n{scene}<end_of_turn>\n<start_of_turn>model\n"
    t0 = time.time()
    r = requests.post(URL, json={
        'prompt': prompt, 'n_predict': 30, 'temperature': 0.1,
        'cache_prompt': True, 'stop': ['<end_of_turn>']
    }, timeout=40)
    d = r.json()
    t = d['timings']
    total = time.time() - t0
    direction = d['content'].strip().split('\n')[0].split('.')[0].split()[0].upper()
    return t['prompt_ms'], t['prompt_n'], t['predicted_per_second'], total, direction

print('='*58)
print('  SWA-FULL REAL ROBOT SIMULATION')
print('  Same system prompt, 8 changing scenes (like real op)')
print('='*58)

times = []
for i, scene in enumerate(SCENES, 1):
    p_ms, p_n, gen, total, direction = call(scene)
    times.append(total)
    tag = 'COLD' if i == 1 else 'cached'
    print(f'  Cycle {i} [{tag:6}] prompt:{p_ms:6.0f}ms({p_n:3d}tok) gen:{gen:4.1f}t/s -> {direction:8} | {total:.1f}s')
    time.sleep(0.5)

print('='*58)
print(f'  SUMMARY')
print(f'    Cycle 1 (cold):    {times[0]:.1f}s')
print(f'    Cycles 2-8 (warm): {sum(times[1:])/len(times[1:]):.1f}s avg')
print(f'    Speedup after warmup: {times[0]/(sum(times[1:])/len(times[1:])):.1f}x')
print(f'    Real robot cycle time: ~{sum(times[1:])/len(times[1:]):.1f}s')
print('='*58)
